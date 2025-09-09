import json
import os
import pickle
import re
from collections import Counter
from typing import List, Dict, Any
import torch
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama

# Constants for file paths
JSON_PATH = "comments.json"
IDX_PATH = "faiss.idx"
CHUNKS_PATH = "chunks.pkl"

# Only initialize embedder ONCE (avoid repeated loads)
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")


def load_data(path: str = JSON_PATH) -> List[Dict[str, Any]]:
    """Load JSON data from file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_data(data: List[Dict[str, Any]], path: str = JSON_PATH) -> None:
    """Save JSON data to file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def chunk_data(data: List[Dict[str, Any]], chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Chunk list of comments into smaller pieces for embedding."""
    texts = [item.get("comment") or item.get("feedback") or item.get("text") or "" for item in data]
    texts = [t for t in texts if t]
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text("\n\n".join(texts))
    return chunks

def build_index(chunks: List[str], index_path: str = IDX_PATH, chunks_path: str = CHUNKS_PATH):
    """Build FAISS index from chunks, save index/chunks to disk."""
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    return index

def load_index(index_path: str = IDX_PATH, chunks_path: str = CHUNKS_PATH):
    """Load FAISS index and chunks from disk."""
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def retrieve(query: str, index, embedder, chunks: List[str], k: int = 5) -> List[str]:
    """Retrieve top-k relevant chunks to the query."""
    query_vec = embedder.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]

def generate(prompt: str, model_name: str = "mistral:7b") -> str:
    """Generate response from Ollama."""
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return response.message.content.strip()

def answer(query: str, index, embedder, chunks: List[str], model_name: str = "mistral:7b") -> str:
    """Complete RAG answer workflow for a question."""
    context = retrieve(query, index, embedder, chunks)
    prompt = (
        "You are an assistant who ONLY answers from these consultation comments.\n\n"
        + "\n".join(context)
        + f"\n\nQuestion: {query}\nAnswer:"
    )
    return generate(prompt, model_name)

def extract_entities(texts: List[str]) -> Dict[str, int]:
    """Extract and count company names in texts."""
    pattern = r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+(?:Inc|Corp|Corporation|LLC|Ltd|Limited|Group|Company|Bank|Systems|Technologies|Electronics))"
    counts = Counter(re.findall(pattern, " ".join(texts)))
    return dict(counts)

topics = {
    "finance": ["financial", "inflation", "budget", "tax", "funding"],
    "technology": ["technology", "tech", "digital", "ai", "software"],
    "healthcare": ["health", "medical", "hospital", "patient"],
    "environment": ["environment", "climate", "renewable", "carbon"],
}

def detect_topics(texts: List[str]) -> Dict[str, int]:
    """Detect topics by keyword frequency."""
    txt = " ".join(texts).lower()
    return {t: sum(txt.count(w) for w in ws) for t, ws in topics.items() if any(w in txt for w in ws)}
