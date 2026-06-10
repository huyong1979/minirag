"""
retriever.py -- find the most relevant document chunks for a query.

Uses hybrid search (BM25 + vector cosine) by default via VectorStore.search().
Automatic keyword extraction anchors date/ID queries as a pre-filter.
"""

from __future__ import annotations

import re
from .embedder import Embedder
from .store import VectorStore


_DATE_PATTERNS = [
    r"\d{2}/\d{2}/\d{4}",   # 04/10/2025
    r"\d{4}-\d{2}-\d{2}",   # 2025-04-10
    r"\d{2}-\d{2}-\d{4}",   # 10-04-2025
]


def _extract_keyword(query: str) -> str | None:
    """Return the first date-like token found in query, or None."""
    for pattern in _DATE_PATTERNS:
        m = re.search(pattern, query)
        if m:
            return m.group(0)
    return None


class Retriever:
    """
    Retrieves the most relevant document chunks using hybrid search.

    Strategy:
    - BM25 handles exact-term recall (hostnames, acronyms, IDs, dates).
    - Vector search handles semantic similarity.
    - Reciprocal Rank Fusion merges both ranked lists.
    - Date/ID tokens are auto-extracted from the query as a pre-filter.

    Args:
        embedder:     An Embedder instance.
        store:        A VectorStore instance (must already have docs added).
        top_k:        Default number of results to return.
        auto_keyword: Auto-extract date/ID tokens as a keyword pre-filter.

    Example::

        retriever = Retriever(embedder, store, top_k=5)
        results = retriever.retrieve("Which server runs PVA Gateway?")
        results = retriever.retrieve("What happened on 04/10/2025?")
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        top_k: int = 5,
        auto_keyword: bool = True,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.auto_keyword = auto_keyword

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        keyword_filter: str | None = None,
    ) -> list[dict]:
        """
        Return the most relevant document chunks for a query string.

        Args:
            query:          The question or search string.
            top_k:          Override the default number of results.
            keyword_filter: Explicit keyword pre-filter. If None and
                            auto_keyword=True, a date/ID is auto-extracted.

        Returns:
            List of document dicts with an added "score" key.
        """
        k = top_k if top_k is not None else self.top_k

        kf = keyword_filter
        if kf is None and self.auto_keyword:
            kf = _extract_keyword(query)

        query_vec = self.embedder.embed(query)
        return self.store.search(
            query=query,
            query_vector=query_vec,
            top_k=k,
            keyword_filter=kf,
        )
