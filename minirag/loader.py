"""
loader.py -- ingest plain text, Markdown, PDF, and RTF files into chunks.

A "document" is just a dict:
    {"text": str, "source": str, "chunk_index": int}
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable


def load_documents(
    paths: str | Path | Iterable[str | Path],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """
    Load one or more files and split them into overlapping text chunks.

    Supported formats: .txt, .md, .pdf (needs pypdf), .rtf

    Args:
        paths:         A single file path, directory path, or list of paths.
                       Directories are scanned recursively.
        chunk_size:    Approximate number of characters per chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        A list of document dicts, each with keys: "text", "source", "chunk_index".

    Example:
        docs = load_documents("my_notes.txt", chunk_size=400)
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    all_docs: list[dict] = []
    for p in paths:
        p = Path(p).expanduser()
        if p.is_dir():
            for ext in ("*.txt", "*.md", "*.pdf", "*.rtf"):
                for fp in sorted(p.rglob(ext)):
                    all_docs.extend(_load_file(fp, chunk_size, chunk_overlap))
        else:
            all_docs.extend(_load_file(p, chunk_size, chunk_overlap))

    return all_docs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_file(path: Path, chunk_size: int, chunk_overlap: int) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _read_pdf(path)
    elif suffix == ".rtf":
        text = _read_rtf(path)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    return _chunk_text(text, str(path), chunk_size, chunk_overlap)


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # optional dependency
    except ImportError:
        raise ImportError(
            "PDF support requires pypdf. Install it with:\n"
            "    pip install 'minirag[pdf]'\n"
            "or:  pip install pypdf"
        )
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_rtf(path: Path) -> str:
    """
    Extract plain text from an RTF file using regex stripping.

    Handles the most common RTF constructs well enough for notes and
    how-to documents. Not a full RTF parser.
    """
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    # Expand \uNNNN Unicode escapes (RTF \uN? format -- ? is ASCII fallback)
    def _uni(m):
        n = int(m.group(1))
        if n < 0:
            n += 65536
        try:
            return chr(n)
        except (ValueError, OverflowError):
            return ""
    text = re.sub(r"\\u(-?\d+)\\?.", _uni, text)

    # Decode \\'XX hex escapes (e.g. \'92 = right single quote)
    def _hex(m):
        try:
            return bytes.fromhex(m.group(1)).decode("latin-1")
        except Exception:
            return ""
    text = re.sub(r"\\' ([0-9a-fA-F]{2})", _hex, text)
    text = re.sub(r"\\'([0-9a-fA-F]{2})", _hex, text)

    # Drop destination groups we don't want: {\fonttbl ...} etc.
    text = re.sub(
        r"\{\\(?:fonttbl|colortbl|info|stylesheet|listtable"
        r"|listoverridetable|pict|object|fldinst|header|footer)[^}]*\}",
        "", text, flags=re.DOTALL
    )

    # Replace paragraph/line break control words with newlines
    text = re.sub(r"\\(?:par|line|pard|sect)\b", "\n", text)

    # Replace tab
    text = re.sub(r"\\tab\b", "\t", text)

    # Drop the \* destination marker and asterisk-prefixed ignored groups
    text = re.sub(r"\\\*", "", text)

    # Remove all remaining RTF control words (\word or \word-123 or \word123)
    text = re.sub(r"\\[a-zA-Z]+[-]?\d*[ ]?", "", text)

    # Remove remaining braces
    text = re.sub(r"[{}]", "", text)

    # Normalise whitespace
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    # Remove trailing backslashes that are RTF line-continuation artefacts
    text = re.sub(r"\\\s*\n", "\n", text)

    return text.strip()


def _chunk_text(
    text: str, source: str, chunk_size: int, chunk_overlap: int
) -> list[dict]:
    # Normalise whitespace but preserve paragraph breaks.
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    chunks: list[dict] = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({"text": chunk, "source": source, "chunk_index": idx})
            idx += 1
        start = end - chunk_overlap  # slide window with overlap

    return chunks
