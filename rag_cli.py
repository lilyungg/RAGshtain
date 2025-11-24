import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import faiss, pickle
import numpy as np
from vllm import LLM, SamplingParams

# --------- настройки ----------
MODEL_NAME = os.getenv("MODEL_NAME", "facebook/opt-125m")
EMB_MODEL = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
INDEX_PATH = os.getenv("INDEX_PATH", "data/wiki.index")
META_PATH = os.getenv("META_PATH", "data/meta.pkl")
# ------------------------------

class WikiIndexer:
    def __init__(self, model_name=EMB_MODEL, batch_size=32):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.index = None
        self.metadata = []

    def chunk_text(self, text, chunk_size=512, overlap=50):
        words = text.split()
        chunks = []
        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + chunk_size])
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
                    "url": art["url"],
                    "chunk": c,
                })
        return processed

    def build_index(self, data, index_path=INDEX_PATH, meta_path=META_PATH):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        d = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(d)
        for i in tqdm(range(0, len(data), self.batch_size), desc="Indexing"):
            batch = data[i:i + self.batch_size]
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

class LocalVLLMRag:
    def __init__(self, model_name=MODEL_NAME):
        print(f"Loading LLM: {model_name}")
        import os
        os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")


        self.llm = LLM(
                model=model_name,
                dtype="float32",          # на CPU надёжнее всего
            )
        self.sampling = SamplingParams(
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        self.indexer = WikiIndexer()
        self._ensure_index()

    def _ensure_index(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            print("Loading existing index...")
            self.indexer.load()
            return
        print("Building index from Wikipedia (1000 articles)...")
        ds = load_dataset("wikipedia", "20220301.simple", split="train", streaming=True)
        data = self.indexer.process_dataset(list(ds.take(1000)))
        self.indexer.build_index(data)
        print("Index built.")

    def query(self, question, k=5):
        docs = self.indexer.search(question, k=k)
        context = "\n\n".join(f"[{d['title']}]\n{d['chunk']}" for d, _ in docs)
        prompt = (
            "Answer the question based only on the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )
        out = self.llm.generate([prompt], self.sampling)[0]
        answer = out.outputs[0].text.strip()
        return answer, docs

def main():
    rag = LocalVLLMRag(MODEL_NAME)
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
        answer, docs = rag.query(q, k=5)
        print("\nA>", answer)
        print("\nSources:")
        for d, score in docs[:3]:
            print(f"- {d['title']} (score={score:.4f})")
        print("-" * 60)

if __name__ == "__main__":
    main()
