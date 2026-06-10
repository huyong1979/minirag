"""
pipeline.py -- the main entry point: RAGPipeline ties everything together.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .embedder import Embedder
from .generator import Generator
from .loader import load_documents
from .retriever import Retriever
from .store import VectorStore


class RAGPipeline:
    """
    A complete RAG pipeline in one class.

    Steps:
    1. Load documents from files.
    2. Embed them and store in an in-memory VectorStore.
    3. For each query: embed -> retrieve -> generate.

    Args:
        embed_model:  Ollama model for embeddings (default: "nomic-embed-text").
        chat_model:   Ollama model for generation (default: "llama3.2").
        base_url:     Ollama server URL (default: http://localhost:11434).
        top_k:        Number of chunks to retrieve per query (default: 5).
        chunk_size:   Characters per text chunk when loading documents (default: 500).
        chunk_overlap: Overlap between consecutive chunks (default: 50).

    Example::

        from minirag import RAGPipeline

        rag = RAGPipeline()
        rag.add_documents("my_notes.txt")
        answer = rag.ask("What does NSLS-2 stand for?")
        print(answer)
    """

    def __init__(
        self,
        embed_model: str = "nomic-embed-text",
        chat_model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        top_k: int = 5,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self.embedder = Embedder(model=embed_model, base_url=base_url)
        self.store = VectorStore()
        self.retriever = Retriever(self.embedder, self.store, top_k=top_k)
        self.generator = Generator(model=chat_model, base_url=base_url)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    def add_documents(
        self,
        paths,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> int:
        """
        Load and index one or more documents.

        Args:
            paths:        File path, directory, or list of paths.
            chunk_size:   Override default chunk size.
            chunk_overlap: Override default chunk overlap.

        Returns:
            Number of chunks added.

        Example::

            count = rag.add_documents(["doc1.txt", "doc2.md"])
            print(f"Indexed {count} chunks.")
        """
        docs = load_documents(
            paths,
            chunk_size=chunk_size or self.chunk_size,
            chunk_overlap=chunk_overlap or self.chunk_overlap,
        )
        if not docs:
            return 0

        print(f"Embedding {len(docs)} chunks...", flush=True)
        vectors = self.embedder.embed_batch([d["text"] for d in docs])
        self.store.add(docs, vectors)
        print(f"Done. Total chunks in store: {len(self.store)}", flush=True)
        return len(docs)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def ask(self, question: str, top_k: int | None = None, keyword_filter: str | None = None) -> str:
        """
        Answer a question using retrieval-augmented generation.

        Args:
            question:       The question to answer.
            top_k:          Override the default number of retrieved chunks.
            keyword_filter: Optional explicit keyword to pre-filter chunks
                            (e.g. a date string). Auto-extracted when None.

        Returns:
            The generated answer as a string.

        Example::

            answer = rag.ask("Summarize the safety procedures.")
            print(answer)
        """
        context_docs = self.retriever.retrieve(question, top_k=top_k, keyword_filter=keyword_filter)
        return self.generator.generate(question, context_docs)

    def retrieve(self, query: str, top_k: int | None = None, keyword_filter: str | None = None) -> list[dict]:
        """
        Return the raw retrieved chunks without generating an answer.
        Useful for debugging or inspecting what the retriever finds.
        """
        return self.retriever.retrieve(query, top_k=top_k, keyword_filter=keyword_filter)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_index(self, path) -> None:
        """Save the vector index to disk (see VectorStore.save)."""
        self.store.save(path)
        print(f"Index saved to {path}.npz / {path}.json")

    def load_index(self, path) -> None:
        """Load a previously saved index from disk (see VectorStore.load)."""
        self.store = VectorStore.load(path)
        self.retriever.store = self.store
        print(f"Index loaded from {path}: {len(self.store)} chunks")
