"""
minirag — a minimal RAG toolkit for local Ollama LLMs.
"""

from .pipeline import RAGPipeline
from .loader import load_documents
from .embedder import Embedder
from .store import VectorStore
from .retriever import Retriever
from .generator import Generator

__version__ = "0.1.0"
__all__ = [
    "RAGPipeline",
    "load_documents",
    "Embedder",
    "VectorStore",
    "Retriever",
    "Generator",
]
