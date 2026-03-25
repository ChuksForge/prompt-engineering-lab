"""
chunker.py + indexer.py
========================
Document Intelligence System — Chunking & Indexing
Project: P9 · prompt-engineering-lab

Chunker:
  - Sentence-aware splitting (no mid-sentence breaks)
  - Section detection (headings, numbered items)
  - Configurable size + overlap
  - Preserves section context in chunk metadata

Indexer:
  - TF-IDF cosine similarity (zero-dependency)
  - Optional: sentence-transformers dense embeddings
  - Retrieves top-k chunks for a query
"""

import re
import math
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# CHUNKER
# ──────────────────────────────────────────────

@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    start_char: int
    end_char: int
    section: str = ""       # detected heading/section context
    chunk_index: int = 0
    total_chunks: int = 0

    @property
    def word_count(self) -> int:
        return len(re.findall(r'\b\w+\b', self.text))


HEADING_PATTERN = re.compile(
    r'^(#{1,3}\s+.+|[A-Z][A-Z\s]{4,}:?$|\d+\.\s+[A-Z].{5,})',
    re.MULTILINE
)

def detect_section(text: str, position: int) -> str:
    """Find the most recent section heading before a given character position."""
    before = text[:position]
    headings = list(HEADING_PATTERN.finditer(before))
    if headings:
        return headings[-1].group().strip()[:80]
    return ""


def split_into_sentences(text: str) -> list:
    """Split text into sentences preserving structure."""
    # Split on sentence-ending punctuation followed by whitespace/newline
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\-\d"])', text)
    # Also split on double newlines (paragraph breaks)
    result = []
    for sent in sentences:
        parts = re.split(r'\n{2,}', sent)
        result.extend([p.strip() for p in parts if p.strip()])
    return result


class Chunker:
    """
    Sentence-aware document chunker.

    Args:
        chunk_size:    Target words per chunk
        chunk_overlap: Words of overlap between adjacent chunks
        min_chunk_size: Minimum words to create a chunk
    """

    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 40,
        min_chunk_size: int = 30,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk(self, doc_id: str, text: str) -> list:
        """
        Chunk a document into overlapping text segments.
        Returns list of Chunk objects.
        """
        sentences  = split_into_sentences(text)
        chunks     = []
        current    = []
        current_words = 0
        char_pos   = 0

        for sent in sentences:
            sent_words = len(re.findall(r'\b\w+\b', sent))

            if current_words + sent_words > self.chunk_size and current:
                # Save current chunk
                chunk_text = " ".join(current)
                start_char = text.find(current[0]) if current else 0
                end_char   = start_char + len(chunk_text)
                section    = detect_section(text, start_char)

                chunks.append(Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_chunk_{len(chunks)}",
                    text=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                    section=section,
                    chunk_index=len(chunks),
                ))

                # Overlap: keep last N words
                overlap_words = []
                total = 0
                for s in reversed(current):
                    sw = re.findall(r'\b\w+\b', s)
                    if total + len(sw) <= self.chunk_overlap:
                        overlap_words.insert(0, s)
                        total += len(sw)
                    else:
                        break
                current       = overlap_words + [sent]
                current_words = sum(len(re.findall(r'\b\w+\b', s)) for s in current)
            else:
                current.append(sent)
                current_words += sent_words

        # Final chunk
        if current and current_words >= self.min_chunk_size:
            chunk_text = " ".join(current)
            start_char = text.find(current[0]) if current else 0
            section    = detect_section(text, start_char)
            chunks.append(Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}_chunk_{len(chunks)}",
                text=chunk_text,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                section=section,
                chunk_index=len(chunks),
            ))

        # Set total_chunks
        for c in chunks:
            c.total_chunks = len(chunks)

        return chunks


# ──────────────────────────────────────────────
# INDEXER
# ──────────────────────────────────────────────

def _tokenize(text: str) -> list:
    stops = {
        "the","a","an","and","or","but","in","on","at","to","for","of",
        "with","by","from","is","are","was","were","be","been","has",
        "have","had","it","its","this","that","as","not","also","which"
    }
    return [t for t in re.findall(r"\b[a-z]+\b", text.lower()) if t not in stops and len(t) > 1]

def _tf(tokens: list) -> dict:
    total = max(1, len(tokens))
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return {t: c/total for t, c in tf.items()}

def _cosine(a: dict, b: dict) -> float:
    dot  = sum(a.get(t, 0) * b.get(t, 0) for t in a)
    na   = math.sqrt(sum(v**2 for v in a.values()))
    nb   = math.sqrt(sum(v**2 for v in b.values()))
    return dot / (na * nb) if na * nb else 0.0


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int


class Indexer:
    """
    TF-IDF document index with optional dense embedding upgrade.

    Usage:
        indexer = Indexer()
        indexer.add_chunks(chunks)
        results = indexer.retrieve("What is the monthly fee?", top_k=3)
    """

    def __init__(self, use_embeddings: bool = False):
        self.use_embeddings = use_embeddings
        self._chunks: list[Chunk] = []
        self._tfidf_vectors: list[dict] = []
        self._idf: dict = {}
        self._embedding_index = None
        self._embeddings = None

    def add_chunks(self, chunks: list):
        """Add chunks to the index."""
        self._chunks.extend(chunks)
        self._build_tfidf()
        if self.use_embeddings:
            self._build_embeddings()

    def _build_tfidf(self):
        n = len(self._chunks)
        if n == 0:
            return

        tf_vectors = [_tf(_tokenize(c.text)) for c in self._chunks]
        df = {}
        for tv in tf_vectors:
            for t in tv:
                df[t] = df.get(t, 0) + 1

        self._idf = {t: math.log(n / (1 + df[t])) for t in df}
        self._tfidf_vectors = [
            {t: v * self._idf.get(t, 0) for t, v in tv.items()}
            for tv in tf_vectors
        ]

    def _build_embeddings(self):
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            model = SentenceTransformer("all-MiniLM-L6-v2")
            texts = [c.text for c in self._chunks]
            self._embeddings = model.encode(texts, show_progress_bar=False)
            logger.info(f"  Embeddings built: {len(self._embeddings)} vectors")
        except ImportError:
            logger.warning("sentence-transformers not installed — falling back to TF-IDF")
            self.use_embeddings = False
        except Exception as e:
            logger.warning(f"Embedding build failed: {e} — falling back to TF-IDF")
            self.use_embeddings = False

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """
        Retrieve the top_k most relevant chunks for a query.
        Uses dense embeddings if available, else TF-IDF cosine.
        """
        if not self._chunks:
            return []

        if self.use_embeddings and self._embeddings is not None:
            return self._retrieve_dense(query, top_k)
        return self._retrieve_tfidf(query, top_k)

    def _retrieve_tfidf(self, query: str, top_k: int) -> list:
        q_tokens = _tokenize(query)
        q_tf     = _tf(q_tokens)
        q_vec    = {t: v * self._idf.get(t, 0) for t, v in q_tf.items()}

        scores = [(i, _cosine(q_vec, cv)) for i, cv in enumerate(self._tfidf_vectors)]
        scores.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievedChunk(chunk=self._chunks[i], score=round(s, 4), rank=rank+1)
            for rank, (i, s) in enumerate(scores[:top_k])
        ]

    def _retrieve_dense(self, query: str, top_k: int) -> list:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        q_emb = model.encode([query])[0]
        norms = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(q_emb)
        scores_arr = np.dot(self._embeddings, q_emb) / np.maximum(norms, 1e-8)
        top_idx = scores_arr.argsort()[::-1][:top_k]
        return [
            RetrievedChunk(chunk=self._chunks[i], score=round(float(scores_arr[i]),4), rank=rank+1)
            for rank, i in enumerate(top_idx)
        ]

    def retrieve_as_context(self, query: str, top_k: int = 3) -> str:
        """Format retrieved chunks as a context string for LLM prompts."""
        chunks = self.retrieve(query, top_k=top_k)
        parts  = []
        for rc in chunks:
            section = f" [{rc.chunk.section}]" if rc.chunk.section else ""
            parts.append(f"[Source{rc.rank}{section}]\n{rc.chunk.text}")
        return "\n\n".join(parts)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def clear(self):
        self._chunks.clear()
        self._tfidf_vectors.clear()
        self._idf.clear()
        self._embeddings = None
