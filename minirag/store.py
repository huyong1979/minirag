"""
store.py -- in-memory vector store with hybrid search (BM25 + cosine).

Dense vector search captures semantic meaning but misses exact terms like
hostnames, acronyms, and IDs.  BM25 captures exact-term matches but ignores
semantics.  Reciprocal Rank Fusion (RRF) of both gives robust retrieval.
"""

from __future__ import annotations

import json
import re
import numpy as np
from pathlib import Path


def _tokenize(text: str) -> list[str]:
    """Simple lowercased tokenizer."""
    return re.findall(r'[a-z0-9_.\-]+', text.lower())


class VectorStore:
    """
    Hybrid vector + BM25 store with Reciprocal Rank Fusion.

    Example::

        store = VectorStore()
        store.add(docs, vectors)
        results = store.search("Which server runs PVA Gateway?",
                               query_vector=vec)
        store.save("my_index")
        store2 = VectorStore.load("my_index")
    """

    def __init__(self) -> None:
        self._docs: list[dict] = []
        self._matrix = None           # np.ndarray shape (N, D)
        self._bm25 = None             # BM25Okapi, rebuilt lazily
        self._corpus: list[list[str]] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add(self, docs: list[dict], vectors: list[list[float]]) -> None:
        """Add documents and their embedding vectors to the store."""
        if len(docs) != len(vectors):
            raise ValueError("docs and vectors must have the same length.")

        new_matrix = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(new_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        new_matrix /= norms

        if self._matrix is None:
            self._matrix = new_matrix
        else:
            self._matrix = np.vstack([self._matrix, new_matrix])

        self._docs.extend(docs)
        self._corpus.extend(_tokenize(d.get("text", "")) for d in docs)
        self._bm25 = None   # invalidate; will be rebuilt lazily

    def _get_bm25(self):
        if self._bm25 is None:
            try:
                from rank_bm25 import BM25Okapi
            except ImportError:
                raise ImportError(
                    "BM25 support requires rank-bm25.  Install with:\n"
                    "    pip install rank-bm25"
                )
            self._bm25 = BM25Okapi(self._corpus)
        return self._bm25

    # ------------------------------------------------------------------
    # Candidate filter (for date/ID pre-filtering)
    # ------------------------------------------------------------------

    def _candidate_indices(self, keyword_filter: str | None) -> list[int]:
        if keyword_filter:
            kf = keyword_filter.lower()
            idx = [i for i, d in enumerate(self._docs)
                   if kf in d.get("text", "").lower()]
            return idx if idx else list(range(len(self._docs)))
        return list(range(len(self._docs)))

    # ------------------------------------------------------------------
    # Individual search methods (exposed for advanced use)
    # ------------------------------------------------------------------

    def search_vector(
        self,
        query_vector: list[float],
        top_k: int = 5,
        keyword_filter: str | None = None,
    ) -> list[dict]:
        """Pure cosine-similarity search."""
        if self._matrix is None or not self._docs:
            return []
        indices = self._candidate_indices(keyword_filter)
        q = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q /= norm
        scores = self._matrix[indices] @ q
        k = min(top_k, len(indices))
        top_pos = np.argpartition(scores, -k)[-k:]
        top_pos = top_pos[np.argsort(scores[top_pos])[::-1]]
        results = []
        for pos in top_pos:
            doc = dict(self._docs[indices[pos]])
            doc["score"] = float(scores[pos])
            results.append(doc)
        return results

    def search_bm25(
        self,
        query: str,
        top_k: int = 5,
        keyword_filter: str | None = None,
    ) -> list[dict]:
        """Pure BM25 keyword search."""
        if not self._docs:
            return []
        bm25 = self._get_bm25()
        indices = self._candidate_indices(keyword_filter)
        tokens = _tokenize(query)
        all_scores = bm25.get_scores(tokens)
        scores = np.array(all_scores)[indices]
        k = min(top_k, len(indices))
        top_pos = np.argpartition(scores, -k)[-k:]
        top_pos = top_pos[np.argsort(scores[top_pos])[::-1]]
        results = []
        for pos in top_pos:
            doc = dict(self._docs[indices[pos]])
            doc["bm25_score"] = float(scores[pos])
            results.append(doc)
        return results

    # ------------------------------------------------------------------
    # Hybrid search: Reciprocal Rank Fusion (default)
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        query_vector: list[float] | None = None,
        top_k: int = 5,
        keyword_filter: str | None = None,
        rrf_k: int = 60,
    ) -> list[dict]:
        """
        Hybrid search via Reciprocal Rank Fusion of BM25 + vector results.

        RRF score = sum(1 / (rrf_k + rank)) across both ranked lists.
        Being rank-based, scale differences between BM25 and cosine scores
        do not matter -- no normalisation needed.

        Falls back to BM25-only if query_vector is None.

        Args:
            query:          Query string (used for BM25 and keyword filter).
            query_vector:   Pre-computed embedding vector (optional).
            top_k:          Number of results to return.
            keyword_filter: Optional exact-match pre-filter string.
            rrf_k:          RRF smoothing constant (default 60).
        """
        if not self._docs:
            return []

        fetch_k = max(top_k * 4, 20)

        bm25_list = self.search_bm25(query, top_k=fetch_k,
                                     keyword_filter=keyword_filter)
        vec_list: list[dict] = []
        if query_vector is not None:
            vec_list = self.search_vector(query_vector, top_k=fetch_k,
                                          keyword_filter=keyword_filter)

        def _key(doc: dict) -> tuple:
            return (doc.get("source", ""), doc.get("chunk_index", -1),
                    doc.get("text", "")[:40])

        rrf: dict[tuple, float] = {}
        doc_map: dict[tuple, dict] = {}

        for rank, doc in enumerate(bm25_list):
            k_ = _key(doc)
            rrf[k_] = rrf.get(k_, 0.0) + 1.0 / (rrf_k + rank + 1)
            doc_map[k_] = doc

        for rank, doc in enumerate(vec_list):
            k_ = _key(doc)
            rrf[k_] = rrf.get(k_, 0.0) + 1.0 / (rrf_k + rank + 1)
            if k_ not in doc_map:
                doc_map[k_] = doc

        sorted_keys = sorted(rrf, key=lambda k: rrf[k], reverse=True)
        results = []
        for k_ in sorted_keys[:top_k]:
            doc = dict(doc_map[k_])
            doc["score"] = rrf[k_]
            doc.pop("bm25_score", None)
            results.append(doc)
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        """Save to <path>.npz (matrix) and <path>.json (docs)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._matrix is not None:
            np.savez_compressed(str(path) + ".npz", matrix=self._matrix)
        jpath = str(path) + ".json"
        fh = open(jpath, "w", encoding="utf-8")
        json.dump(self._docs, fh, ensure_ascii=False, indent=2)
        fh.close()

    @classmethod
    def load(cls, path) -> "VectorStore":
        """Load a previously saved store from <path>.npz + <path>.json."""
        path = Path(path)
        store = cls()
        npz_path = str(path) + ".npz"
        json_path = str(path) + ".json"
        if Path(npz_path).exists():
            data = np.load(npz_path)
            store._matrix = data["matrix"]
        fh = open(json_path, "r", encoding="utf-8")
        store._docs = json.load(fh)
        fh.close()
        store._corpus = [_tokenize(d.get("text", "")) for d in store._docs]
        return store

    def __len__(self) -> int:
        return len(self._docs)
