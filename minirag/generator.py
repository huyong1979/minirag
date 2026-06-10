"""
generator.py -- send a question + retrieved context to an Ollama chat model.
"""

from __future__ import annotations

import requests


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question based on the provided context. "
    "Use the context as your primary source. If the context contains relevant information, "
    "synthesize and summarize it even if it is incomplete or spread across multiple sources. "
    "Only say you don't know if the context contains no relevant information at all. "
    "When citing sources, always use the filename (e.g. timing.md), not the reference number. "
    "Be concise and factual."
)


class Generator:
    """
    Calls the Ollama /api/chat endpoint with a RAG-style prompt.

    Args:
        model:         Ollama chat model (default: "llama3.2").
        base_url:      Ollama server URL (default: http://localhost:11434).
        system_prompt: Override the default system instruction.
        timeout:       HTTP timeout in seconds.

    Example::

        gen = Generator(model="llama3.2")
        answer = gen.generate("What is synchrotron light?", context_docs)
        print(answer)
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.system_prompt = system_prompt
        self.timeout = timeout
        self._endpoint = f"{self.base_url}/api/chat"

    def generate(self, question: str, context_docs: list[dict]) -> str:
        """
        Generate an answer to the question using retrieved context documents.

        Args:
            question:     The user's question.
            context_docs: List of document dicts returned by Retriever.retrieve().

        Returns:
            The assistant's answer as a plain string.
        """
        context = self._build_context(context_docs)
        user_message = f"Context:\n{context}\n\nQuestion: {question}"

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
        }

        try:
            resp = requests.post(self._endpoint, json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Make sure Ollama is running (`ollama serve`)."
            )

        data = resp.json()
        return data["message"]["content"].strip()

    @staticmethod
    def _build_context(docs: list[dict]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.get("source", "unknown")
            text = doc.get("text", "").strip()
            parts.append(f"[{i}] (from {source})\n{text}")
        return "\n\n".join(parts)
