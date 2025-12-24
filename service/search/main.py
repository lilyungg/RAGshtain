import os
import pickle
import requests
import faiss

from fastapi import FastAPI
from pydantic import BaseModel
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

# ================= CONFIG =================

EMB_MODEL = os.getenv("EMB_MODEL", "intfloat/multilingual-e5-large")

DATA_DIR = "e5-large"
INDEX_PATH = f"{DATA_DIR}/merged.index"
META_PATH = f"{DATA_DIR}/merged_meta.pkl"

CHUNK_SIZE = 512
OVERLAP = 64

# ================= APP =================

app = FastAPI(title="Habr Parser + FAISS")

model = SentenceTransformer(EMB_MODEL)
index = None
metadata: list[dict] = []

# ================= UTILS =================

def chunk_text(text: str, chunk_size=512, overlap=64):
    words = text.split()
    step = max(1, chunk_size - overlap)
    chunks = []

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def save_index():
    os.makedirs(DATA_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)


def load_index():
    global index, metadata
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            metadata = pickle.load(f)
        print(f"✅ Loaded FAISS index ({index.ntotal} vectors)")
    else:
        dim = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)
        metadata = []
        print("🆕 Created new FAISS index")

# ================= MODELS =================

class ParseRequest(BaseModel):
    url: str


class SearchRequest(BaseModel):
    query: str
    k: int = 5

# ================= EVENTS =================

@app.on_event("startup")
def startup():
    load_index()

# ================= API =================

@app.post("/parse")
def parse_habr(req: ParseRequest):
    global index, metadata

    r = requests.get(
        req.url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    soup = BeautifulSoup(r.text, "html.parser")

    title_tag = soup.select_one("h1.tm-title span")
    title = title_tag.text.strip() if title_tag else ""

    body = soup.select_one("div.article-formatted-body")
    if not body:
        return {"error": "Article content not found"}

    text = body.get_text(" ", strip=True)
    chunks = chunk_text(text)
    if not chunks:
        return {"error": "No text chunks produced"}

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    start_id = len(metadata)

    for i, chunk in enumerate(chunks):
        metadata.append({
            "source": "habr",
            "url": req.url,
            "document_title": title,
            "chunk_id": i,
            "text": chunk
        })

    save_index()

    return {
        "status": "ok",
        "title": title,
        "chunks_added": len(chunks),
        "total_vectors": index.ntotal,
        "meta_range": [start_id, len(metadata) - 1]
    }


@app.post("/search")
def search(req: SearchRequest):
    if index.ntotal == 0:
        return []
    q = model.encode([req.query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)
    scores, idx = index.search(q, req.k)
    print(idx)
    results = []
    for rank, i in enumerate(idx[0]):
        if i < 0:
            continue
        results.append({
            "score_cosine": float(scores[0][rank]),
            **metadata[int(i)]
        })
    print(results)
    return results