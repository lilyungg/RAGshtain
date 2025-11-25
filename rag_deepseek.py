import os
import pickle
import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import requests

INDEX_PATH = "data/wiki.index"
META_PATH = "data/meta.pkl"
EMB_MODEL = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))

class WikiIndexer:
    def __init__(self, model_name=EMB_MODEL, batch_size=32):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.index = None
        self.metadata = []

    def chunk_text(self, text, chunk_size=512, overlap=50):
        words = text.split()
        step = max(1, chunk_size - overlap)
        chunks = []
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i+chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def process_dataset(self, dataset, max_articles=None):
        processed = []
        articles = dataset[:max_articles] if max_articles else dataset
        for art in tqdm(articles, desc="Processing"):
            chunks = self.chunk_text(art["text"])
            for c in chunks:
                processed.append({
                    "title": art["title"],
                    "url": art.get("url", ""),
                    "chunk": c
                })
        return processed

    def build_index(self, data, index_path=INDEX_PATH, meta_path=META_PATH):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        d = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(d)
        for i in tqdm(range(0, len(data), self.batch_size), desc="Indexing"):
            batch = data[i:i+self.batch_size]
            texts = [x["chunk"] for x in batch]
            emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            self.index.add(emb.astype("float32"))
            self.metadata.extend(batch)
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, index_path=INDEX_PATH, meta_path=META_PATH):
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query, k=5):
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        dist, idx = self.index.search(q_emb, k)
        return [(self.metadata[i], dist[0][j]) for j, i in enumerate(idx[0])]

class DeepSeekRag:
    def __init__(self, indexer: WikiIndexer):
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        self.indexer = indexer
        self.api_key = DEEPSEEK_API_KEY
        self.model = DEEPSEEK_MODEL

    def _call_deepseek(self, prompt: str) -> str:
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.7,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()

    def query(self, question, k=5):
        docs = self.indexer.search(question, k=k)
        context = "\n\n".join(f"[{d['title']}]\n{d['chunk']}" for d, _ in docs)
        prompt = (
            "Answer the question based only on the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )
        answer = self._call_deepseek(prompt)
        return answer, docs

def main():
    indexer = WikiIndexer()
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        print("Loading existing index...")
        indexer.load()
    else:
        print("Building index from EPSTEIN_FILES_20K...")
        ds = load_dataset("tensonaut/EPSTEIN_FILES_20K", split="train")
        # Этот датасет: поля file_name, text, возможно "length" и др.
        records = []
        for row in ds:
            text = row.get("text") or row.get("content") or ""
            if not text.strip():
                continue
            title = row.get("file_name") or row.get("id") or "EPSTEIN_DOC"
            url = ""  # URL нет
            records.append({"title": title, "url": url, "text": text})

        data = indexer.process_dataset(records)
        indexer.build_index(data)
        print("Index built.")

    rag = DeepSeekRag(indexer)
    print("RAG ready. Type your question (or /exit):")
    while True:
        try:
            q = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not q:
            continue
        if q in ("/exit", "/quit"):
            print("Bye.")
            break
        answer, docs = rag.query(q)
        print("\nA>", answer)
        print("\nSources:")
        for d, score in docs[:3]:
            print(f"- {d['title']} (score={score:.4f})")
            print(f"Chunk:\n{d['chunk']}\n")
        print("-" * 60)

if __name__ == "__main__":
    main()