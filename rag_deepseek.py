import json
import os
import pickle
from pathlib import Path

import faiss
import requests
from sentence_transformers import SentenceTransformer

INDEX_PATH = "C:\\Users\garan\PycharmProjects\RAGshtain\data\e5-large\merged.index"
META_PATH = "C:\\Users\garan\PycharmProjects\RAGshtain\data\e5-large\merged_meta.pkl"
EMB_MODEL = os.getenv("EMB_MODEL", "intfloat/multilingual-e5-large")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))


def chunk_text(text: str, chunk_size=512, overlap=64):
    words = (text or "").split()
    step = max(1, chunk_size - overlap)
    out = []
    for i in range(0, len(words), step):
        s = " ".join(words[i : i + chunk_size]).strip()
        if s:
            out.append(s)
    return out


class MergedFaissIndex:
    """
    Cosine similarity with FAISS:
      - IndexFlatIP (inner product)
      - L2-normalize all vectors so dot product == cosine similarity.
    """
    def __init__(self, model_name=EMB_MODEL, batch_size=32):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.index = None
        self.meta = []

    def build(
        self,
        merged_json_path="examples/merged.json",
        index_path=INDEX_PATH,
        meta_path=META_PATH,
        chunk_size=512,
        overlap=64,
    ):
        items = json.loads(Path(merged_json_path).read_text(encoding="utf-8"))
        if not isinstance(items, list):
            raise TypeError("merged.json must be a JSON array (list of objects)")

        records = []
        for obj in items:
            text = obj.get("text") or ""
            for j, ch in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
                ch = "" if ch is None else str(ch)
                ch = ch.strip()
                if not ch:
                    continue

                records.append(
                    {
                        "document_title": obj.get("document_title"),
                        "url": obj.get("url"),
                        "section_title": obj.get("section_title"),
                        "useful_links": obj.get("useful_links") or [],
                        "chunk_id": j,
                        "chunk": ch,
                    }
                )

        if not records:
            raise RuntimeError("No chunks were produced from merged.json (nothing to index).")

        d = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(d)  # cosine via normalized inner product [web:40]
        self.meta = []

        Path(index_path).parent.mkdir(parents=True, exist_ok=True)

        for i in range(0, len(records), self.batch_size):
            batch = records[i : i + self.batch_size]
            texts = []
            cleaned_batch = []

            # extra safety: guarantee list[str] only
            for x in batch:
                t = x.get("chunk")
                if t is None:
                    continue
                t = str(t).strip()
                if not t:
                    continue
                texts.append(t)
                cleaned_batch.append(x)

            if not texts:
                continue

            emb = self.model.encode(texts, convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(emb)  # in-place normalization [web:31]
            self.index.add(emb)
            self.meta.extend(cleaned_batch)

        if self.index.ntotal == 0:
            raise RuntimeError("FAISS index is empty after build(). Check your merged.json content.")

        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.meta, f)

    def load(self, index_path=INDEX_PATH, meta_path=META_PATH):
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.meta = pickle.load(f)

    def search(self, query: str, k=3):
        if query is None:
            query = ""
        query = str(query).strip()
        if not query:
            return []

        q = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q)  # normalize query as well [web:31]
        scores, idx = self.index.search(q, k)  # inner product == cosine [web:40]

        out = []
        for rank, i in enumerate(idx[0]):
            if i < 0:
                continue
            m = self.meta[int(i)]
            out.append(
                {
                    "score_cosine": float(scores[0][rank]),
                    "document_title": m.get("document_title"),
                    "url": m.get("url"),
                    "section_title": m.get("section_title"),
                    "chunk": m.get("chunk"),
                    "useful_links": m.get("useful_links"),
                }
            )
        return out


class DeepSeekRag:
    def __init__(self, indexer: MergedFaissIndex):
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        self.indexer = indexer

    def _call(self, messages, max_tokens=MAX_TOKENS, temperature=0.2) -> str:
        r = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def answer(self, question: str, k=3):
        hits = self.indexer.search(question, k=k)
        context = "\n\n".join(
            f"[{h['document_title']} | {h['section_title']}]\n{h['chunk']}"
            for h in hits
            if h.get("chunk")
        )
        messages = [
            {"role": "system", "content": "Answer using only the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"},
        ]
        return self._call(messages), hits


def main():
    ix = MergedFaissIndex()

    if Path(INDEX_PATH).exists() and Path(META_PATH).exists():
        ix.load()
    else:
        ix.build("examples/merged.json")

    rag = DeepSeekRag(ix)

    while True:
        q = input("Q> ").strip()
        if q in ("/exit", "/quit"):
            break
        ans, hits = rag.answer(q, k=3)
        print("\nA>", ans)
        print("\nTop-3 sources:")
        for h in hits:
            print(f"- {h['document_title']} | {h['section_title']}")
            print(f"  {h['url']}")
            print(f"  score_cosine={h['score_cosine']:.4f}\n")


if __name__ == "__main__":
    main()
