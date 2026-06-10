# minirag

A minimal, beginner-friendly Python package for building a local RAG (Retrieval-Augmented Generation) system.
It uses a local [Ollama](https://ollama.com) LLM — no OpenAI API key, no cloud, no heavy dependencies.

## What is RAG?

RAG stands for **Retrieval-Augmented Generation**. Instead of asking an LLM a question cold, you first
retrieve relevant passages from your own documents, then pass those as context to the LLM. This grounds
the answer in your data and reduces hallucination.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) installed and running
- The embedding and chat models pulled in Ollama

Works on **Linux, macOS, and Windows**.

## Install

```bash
# Clone or download this repo, then:
pip install -e .

# For PDF support:
pip install -e ".[pdf]"
```

## Quick start

```bash
# 1. Install and start Ollama
ollama serve

# 2. Pull the required models (one-time setup)
ollama pull nomic-embed-text   # embedding model
ollama pull llama3.2           # chat model

# 3. Run the example
python examples/basic_usage.py
```

## Usage

```python
from minirag import RAGPipeline

# Create the pipeline
rag = RAGPipeline(
    embed_model="nomic-embed-text",  # any Ollama embedding model
    chat_model="llama3.2",           # any Ollama chat model
)

# Index your documents (txt, md, or pdf)
rag.add_documents("my_notes.txt")
rag.add_documents("docs/")           # whole directory

# Ask questions
answer = rag.ask("What is the main topic of these documents?")
print(answer)

# Save the index so you don't have to re-embed next time
rag.save_index("my_index")

# Later: reload without re-embedding
rag.load_index("my_index")
answer = rag.ask("Another question")
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_model` | `nomic-embed-text` | Ollama embedding model |
| `chat_model` | `llama3.2` | Ollama chat model |
| `base_url` | `http://localhost:11434` | Ollama server URL |
| `top_k` | `5` | Chunks retrieved per query |
| `chunk_size` | `500` | Characters per chunk |
| `chunk_overlap` | `50` | Overlap between chunks |

## Package layout

```
minirag/
    __init__.py     -- public API
    pipeline.py     -- RAGPipeline (main class)
    loader.py       -- file loading + chunking
    embedder.py     -- Ollama embeddings
    store.py        -- in-memory vector store (NumPy cosine similarity)
    retriever.py    -- query -> top-K chunks
    generator.py    -- context + question -> Ollama answer
examples/
    basic_usage.py  -- end-to-end demo
```

## Dependencies

- `requests` -- HTTP calls to Ollama
- `numpy` -- vector math (cosine similarity)
- `pypdf` *(optional)* -- PDF text extraction

## License

MIT
