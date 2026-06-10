"""
tests/test_keyword_filter.py -- tests for keyword pre-filter and date extraction.
"""

import pytest
from minirag.store import VectorStore
from minirag.retriever import _extract_keyword


# ------------------------------------------------------------------
# _extract_keyword
# ------------------------------------------------------------------

def test_extract_mm_dd_yyyy():
    assert _extract_keyword("What happened on 04/10/2025?") == "04/10/2025"

def test_extract_iso_date():
    assert _extract_keyword("Activities on 2025-04-10 please") == "2025-04-10"

def test_extract_no_date():
    assert _extract_keyword("Tell me about synchrotron light") is None


# ------------------------------------------------------------------
# VectorStore keyword_filter via search_bm25 and search_vector
# ------------------------------------------------------------------

def make_dated_store():
    store = VectorStore()
    docs = [
        {"text": "04/10/2025: Departed from EWR at 4am.", "source": "daily.md", "chunk_index": 0},
        {"text": "04/11/2025: Arrived in Hong Kong.", "source": "daily.md", "chunk_index": 1},
        {"text": "04/12/2025: Visited Shenzhen.", "source": "daily.md", "chunk_index": 2},
    ]
    vectors = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0],
    ]
    store.add(docs, vectors)
    return store


def test_keyword_filter_narrows_to_correct_date():
    store = make_dated_store()
    results = store.search_vector([1.0, 0.0, 0.0], top_k=3,
                                  keyword_filter="04/10/2025")
    assert len(results) == 1
    assert "04/10/2025" in results[0]["text"]


def test_keyword_filter_fallback_when_no_match():
    store = make_dated_store()
    results = store.search_vector([1.0, 0.0, 0.0], top_k=2,
                                  keyword_filter="01/01/2000")
    assert len(results) == 2


def test_keyword_filter_case_insensitive():
    store = VectorStore()
    docs = [{"text": "Event on 04/10/2025: conference.", "source": "t", "chunk_index": 0}]
    store.add(docs, [[1.0, 0.0]])
    results = store.search_vector([1.0, 0.0], top_k=1,
                                  keyword_filter="04/10/2025")
    assert len(results) == 1


def test_hybrid_search_exact_hostname():
    """BM25 should surface the exact hostname even when vector is not close."""
    store = VectorStore()
    docs = [
        {"text": "ioc-server1.example.com runs the PVA Gateway service.",
         "source": "work.md", "chunk_index": 0},
        {"text": "The weather in New York is sunny today.",
         "source": "notes.md", "chunk_index": 1},
        {"text": "EPICS channel access uses UDP broadcast on port 5064.",
         "source": "epics.md", "chunk_index": 2},
    ]
    # All vectors are close to each other -- vector search alone is unreliable
    vectors = [
        [0.9, 0.1, 0.0],
        [0.85, 0.15, 0.0],
        [0.88, 0.12, 0.0],
    ]
    store.add(docs, vectors)
    # Hybrid search -- BM25 should rank the pvagw doc first
    results = store.search("Which server is PVA Gateway deployed on?",
                           query_vector=[0.9, 0.1, 0.0], top_k=1)
    assert len(results) == 1
    assert "ioc-server1.example.com" in results[0]["text"]
