"""
embedder.py — generate text embeddings via the Ollama /api/embeddings endpoint.
"""

from __future__ import annotations

import requests
from typing import Sequence


class Embedder:
    """
    Wraps the Ollama embedding API.

    Args:
        model:    The Ollama embedding model to use (default: "nomic-embed-text").
                  Run `ollama pull nomic-embed-text` once before using.
        base_url: Base URL of the Ollama server (default: http://localhost:11434).
        timeout:  HTTP request timeout in seconds.

    Example:
        embedder = Embedder()
        vector = embedder.embed("Hello, world!")
        vectors = embedder.embed_batch(["Hello", "World"])
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._endpoint = f"{self.base_url}/api/embeddings"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for a single string."""
        return self._call(text)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Return embedding vectors for a list of strings (one request each)."""
        return [self._call(t) for t in texts]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call(self, text: str) -> list[float]:
        try:
            resp = requests.post(
                self._endpoint,
                json={"model": self.model, "prompt": text},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Make sure Ollama is running (`ollama serve`)."
            )
        data = resp.json()
        if "embedding" not in data:
            raise ValueError(f"Unexpected Ollama response: {data}")
        return data["embedding"]
