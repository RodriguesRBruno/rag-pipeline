"""Stage 1: Document ingestion and chunking.

Loads the Enter the Gungeon document corpus and splits it into chunks using
two independent strategies (semantic and sentence-based) so that retrieval
quality can be compared across strategies in later stages.
"""

from __future__ import annotations

import html
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DOCUMENTS_CSV = "dataset/documents.csv"
DEFAULT_DATA_DIR = "data"

REQUIRED_COLUMNS = {"index", "source_url", "text"}

SEMANTIC_CHUNK_CONFIG = {
    "target_size": 800,
    "min_size": 600,
    "max_size": 1000,
    "overlap_ratio": 0.1,
}

SENTENCE_CHUNK_CONFIG = {
    "target_size": 400,
    "min_size": 300,
    "max_size": 500,
    "overlap": 1,
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
_WHITESPACE_RE = re.compile(r"[ \t]+")


@dataclass
class Document:
    """A single cleaned document from the corpus."""

    index: int
    source_url: str
    text: str
    tokens: int


@dataclass
class Chunk:
    """A chunk of text produced by a chunking strategy."""

    chunk_id: str
    text: str
    document_index: int
    source_url: str
    tokens: int
    strategy: str


def count_tokens(text: str) -> int:
    """Conservative word-split token estimate."""
    return len(text.split())


def clean_text(text: str) -> str:
    """Unescape HTML entities and normalize whitespace while preserving paragraphs."""
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _PARAGRAPH_SPLIT_RE.sub("\n\n", text)
    return text.strip()


def load_documents(csv_path: str = DEFAULT_DOCUMENTS_CSV) -> List[Document]:
    """Load, validate, and clean documents from the corpus CSV.

    Args:
        csv_path: Path to documents.csv.

    Returns:
        List of cleaned Document objects.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Documents CSV not found: {csv_path}")

    df = pd.read_csv(path, encoding="utf-8")
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"documents.csv missing required columns: {missing_columns}")

    documents: List[Document] = []
    dropped = 0
    for row in df.itertuples(index=False):
        source_url = str(row.source_url).strip() if not pd.isna(row.source_url) else ""
        raw_text = str(row.text) if not pd.isna(row.text) else ""
        text = clean_text(raw_text)

        if not text or not source_url:
            dropped += 1
            logger.warning("Dropping document index=%s: empty text or source_url", row.index)
            continue

        documents.append(
            Document(
                index=int(row.index),
                source_url=source_url,
                text=text,
                tokens=count_tokens(text),
            )
        )

    total_tokens = sum(doc.tokens for doc in documents)
    avg_tokens = total_tokens / len(documents) if documents else 0
    logger.info(
        "Loaded %d documents (%d dropped) | total_tokens=%d avg_tokens=%.1f",
        len(documents),
        dropped,
        total_tokens,
        avg_tokens,
    )
    return documents


def _split_words_with_overlap(words: List[str], max_size: int, overlap: int) -> List[List[str]]:
    """Split a list of words into fixed-size windows with overlap."""
    if len(words) <= max_size:
        return [words]

    step = max(max_size - overlap, 1)
    windows = []
    start = 0
    while start < len(words):
        end = min(start + max_size, len(words))
        windows.append(words[start:end])
        if end == len(words):
            break
        start += step
    return windows


def _paragraphs(text: str) -> List[str]:
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
    return paragraphs or [text]


def chunk_semantic(
    documents: List[Document],
    target_size: int = SEMANTIC_CHUNK_CONFIG["target_size"],
    min_size: int = SEMANTIC_CHUNK_CONFIG["min_size"],
    max_size: int = SEMANTIC_CHUNK_CONFIG["max_size"],
    overlap_ratio: float = SEMANTIC_CHUNK_CONFIG["overlap_ratio"],
) -> List[Chunk]:
    """Chunk documents by semantic (paragraph/section) boundaries.

    Falls back to fixed-size windows with overlap when a document has no
    paragraph structure or a single paragraph exceeds max_size.
    """
    overlap = max(int(round(target_size * overlap_ratio)), 1)
    chunks: List[Chunk] = []

    for doc in documents:
        paragraphs = _paragraphs(doc.text)
        segments: List[str] = []
        buffer_words: List[str] = []

        def flush_buffer():
            if buffer_words:
                segments.append(" ".join(buffer_words))
                buffer_words.clear()

        for paragraph in paragraphs:
            para_words = paragraph.split()

            if len(para_words) > max_size:
                flush_buffer()
                for window in _split_words_with_overlap(para_words, max_size, overlap):
                    segments.append(" ".join(window))
                continue

            if len(buffer_words) + len(para_words) > max_size:
                flush_buffer()

            buffer_words.extend(para_words)

            if len(buffer_words) >= target_size:
                flush_buffer()

        flush_buffer()

        # Merge trailing tiny segments into the previous one so we don't emit
        # near-empty chunks, unless it's the only segment for the document.
        merged: List[str] = []
        for segment in segments:
            if (
                merged
                and count_tokens(segment) < min_size
                and count_tokens(merged[-1]) + count_tokens(segment) <= max_size
            ):
                merged[-1] = merged[-1] + " " + segment
            else:
                merged.append(segment)

        for i, segment_text in enumerate(merged):
            chunks.append(
                Chunk(
                    chunk_id=f"semantic_{doc.index}_{i}",
                    text=segment_text,
                    document_index=doc.index,
                    source_url=doc.source_url,
                    tokens=count_tokens(segment_text),
                    strategy="semantic",
                )
            )

    return chunks


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, preferring NLTK and falling back to regex."""
    try:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize

        sentences = sent_tokenize(text)
        if sentences:
            return sentences
    except Exception:
        logger.debug("Falling back to regex sentence splitting", exc_info=True)

    sentences = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences.extend(s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip())
    return sentences or [text]


def chunk_sentence(
    documents: List[Document],
    target_size: int = SENTENCE_CHUNK_CONFIG["target_size"],
    min_size: int = SENTENCE_CHUNK_CONFIG["min_size"],
    max_size: int = SENTENCE_CHUNK_CONFIG["max_size"],
    overlap: int = SENTENCE_CHUNK_CONFIG["overlap"],
) -> List[Chunk]:
    """Chunk documents by grouping complete sentences into fixed-size chunks."""
    chunks: List[Chunk] = []

    for doc in documents:
        sentences = _split_sentences(doc.text)

        # Split any pathologically long single sentence on word boundaries.
        normalized_sentences: List[str] = []
        for sentence in sentences:
            words = sentence.split()
            if len(words) > max_size:
                for window in _split_words_with_overlap(words, max_size, 0):
                    normalized_sentences.append(" ".join(window))
            else:
                normalized_sentences.append(sentence)
        sentences = normalized_sentences

        groups: List[List[str]] = []
        buffer: List[str] = []
        buffer_tokens = 0
        new_since_flush = 0

        def flush():
            nonlocal buffer, buffer_tokens, new_since_flush
            groups.append(list(buffer))
            buffer = buffer[-overlap:] if overlap else []
            buffer_tokens = sum(count_tokens(s) for s in buffer)
            new_since_flush = 0

        for sentence in sentences:
            sentence_tokens = count_tokens(sentence)

            if buffer and buffer_tokens + sentence_tokens > max_size:
                flush()

            buffer.append(sentence)
            buffer_tokens += sentence_tokens
            new_since_flush += 1

            if buffer_tokens >= target_size:
                flush()

        if new_since_flush > 0:
            groups.append(list(buffer))

        # Merge a too-small trailing group into the previous one when it fits,
        # dropping the sentences it repeats from the overlap. Otherwise leave
        # it as a short final chunk (allowed for the last chunk in a document).
        if len(groups) >= 2:
            last_tokens = sum(count_tokens(s) for s in groups[-1])
            if last_tokens < min_size:
                skip = min(overlap, len(groups[-1]))
                new_content = groups[-1][skip:]
                combined = groups[-2] + new_content
                if sum(count_tokens(s) for s in combined) <= max_size:
                    groups[-2] = combined
                    groups.pop()

        for i, group in enumerate(groups):
            text = " ".join(group)
            chunks.append(
                Chunk(
                    chunk_id=f"sentence_{doc.index}_{i}",
                    text=text,
                    document_index=doc.index,
                    source_url=doc.source_url,
                    tokens=count_tokens(text),
                    strategy="sentence",
                )
            )

    return chunks


def save_chunks(chunks: List[Chunk], strategy: str, path: str) -> None:
    """Save chunks to JSON with summary metadata."""
    total_tokens = sum(c.tokens for c in chunks)
    avg_chunk_size = total_tokens / len(chunks) if chunks else 0

    payload = {
        "metadata": {
            "strategy": strategy,
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "avg_chunk_size": avg_chunk_size,
        },
        "chunks": [asdict(c) for c in chunks],
    }

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    logger.info(
        "Saved %d %s chunks (%d tokens, avg %.1f/chunk) -> %s",
        len(chunks),
        strategy,
        total_tokens,
        avg_chunk_size,
        path,
    )


def load_chunks(path: str) -> List[Chunk]:
    """Load previously saved chunks from JSON."""
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return [Chunk(**c) for c in payload["chunks"]]


def run_ingestion(
    documents_csv: str = DEFAULT_DOCUMENTS_CSV,
    data_dir: str = DEFAULT_DATA_DIR,
) -> None:
    """Load documents and produce both chunk sets, saving them to data_dir."""
    documents = load_documents(documents_csv)

    semantic_chunks = chunk_semantic(documents)
    save_chunks(semantic_chunks, "semantic", str(Path(data_dir) / "chunks_semantic.json"))

    sentence_chunks = chunk_sentence(documents)
    save_chunks(sentence_chunks, "sentence", str(Path(data_dir) / "chunks_sentence.json"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_ingestion()
