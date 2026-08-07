import pickle
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.config import KB_DIR, TFIDF_CACHE_PATH
from backend.llm_client import generate


def _chunk_document(path: Path) -> list[dict]:
    text = path.read_text()
    parts = re.split(r"\n(?=\d+\.\s)|\n(?=#)", text)
    chunks = []
    for part in parts:
        part = part.strip()
        if len(part) > 15:
            chunks.append({"source": path.stem, "text": part})
    return chunks


def _load_all_chunks() -> list[dict]:
    chunks = []
    for path in sorted(KB_DIR.glob("*.md")):
        chunks.extend(_chunk_document(path))
    return chunks


class KnowledgeBase:
    def __init__(self):
        self.chunks = []
        self.vectorizer = None
        self.matrix = None
        self._load_or_build()

    def _load_or_build(self):
        if TFIDF_CACHE_PATH.exists():
            with open(TFIDF_CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            self.chunks = cached["chunks"]
            self.vectorizer = cached["vectorizer"]
            self.matrix = cached["matrix"]
            return
        self._build()

    def _build(self):
        self.chunks = _load_all_chunks()
        texts = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(texts)
        with open(TFIDF_CACHE_PATH, "wb") as f:
            pickle.dump({"chunks": self.chunks, "vectorizer": self.vectorizer, "matrix": self.matrix}, f)

    def rebuild(self):
        if TFIDF_CACHE_PATH.exists():
            TFIDF_CACHE_PATH.unlink()
        self._build()

    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        if self.matrix is None or len(self.chunks) == 0:
            return []
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": float(sims[i])} for i in top_idx]


_kb = None


def get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb


def answer_from_knowledge_base(query: str) -> dict:
    kb = get_kb()
    hits = kb.retrieve(query, k=4)
    if not hits:
        return {"answer": "No relevant institutional documents found.", "sources": []}
    context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    prompt = (
        f"Answer the question using ONLY the context below, from official campus documents. "
        f"Be concise and cite which document each fact comes from in brackets.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )
    answer = generate(prompt, system_instruction="You are the Knowledge Agent for a campus assistant. Only use the provided context.")
    return {"answer": answer, "sources": sorted(set(h["source"] for h in hits))}