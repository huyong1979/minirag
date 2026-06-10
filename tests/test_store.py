"""
tests/test_store.py -- unit tests for VectorStore (no Ollama required).
"""

import numpy as np
import pytest
from minirag.store import VectorStore


def make_store():
    store = VectorStore()
    docs = [
        {"text": "The sky is blue.", "source": "test", "chunk_index": 0},
        {"text": "The grass is green.", "source": "test", "chunk_index": 1},
        {"text": "The sun is yellow.", "source": "test", "chunk_index": 2},
    ]
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    store.add(docs, vectors)
    return store


def test_add_and_len():
    store = make_store()
    assert len(store) == 3


def test_search_vector_returns_correct_top():
    store = make_store()
    results = store.search_vector([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0]["text"] == "The sky is blue."
    assert results[0]["score"] > 0.99


def test_search_vector_top_k_limit():
    store = make_store()
    results = store.search_vector([1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2


def test_search_vector_empty_store():
    store = VectorStore()
    results = store.search_vector([1.0, 0.0, 0.0], top_k=3)
    assert results == []


def test_search_bm25_exact_term():
    store = make_store()
    results = store.search_bm25("yellow sun", top_k=1)
    assert len(results) == 1
    assert "yellow" in results[0]["text"]


def test_hybrid_search_returns_results():
    store = make_store()
    # Hybrid: query for "sky" -- BM25 should surface "sky is blue" highly
    results = store.search("sky", query_vector=[1.0, 0.0, 0.0, 0.0], top_k=3)
    assert len(results) >= 1
    texts = [r["text"] for r in results]
    assert any("sky" in t or "blue" in t for t in texts)


def test_save_load(tmp_path):
    store = make_store()
    idx_path = tmp_path / "test_index"
    store.save(str(idx_path))

    store2 = VectorStore.load(str(idx_path))
    assert len(store2) == 3
    results = store2.search_vector([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert results[0]["text"] == "The sky is blue."


def test_add_mismatch_raises():
    store = VectorStore()
    with pytest.raises(ValueError):
        store.add([{"text": "a"}], [[1.0, 0.0], [0.0, 1.0]])
