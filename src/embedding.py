"""Stage 2: Embedding generation for chunked documents.

Generates dense vector embeddings for chunks produced by src.ingestion using
two pre-trained Sentence-Transformers models, so retrieval quality can later
be compared across models.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.ingestion import Chunk, load_chunks

logger = logging.getLogger(__name__)

MODEL_NAMES: Dict[str, str] = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}

CHUNK_FILES: Dict[str, str] = {
    "semantic": "chunks_semantic.json",
    "sentence": "chunks_sentence.json",
}

DEFAULT_BATCH_SIZE = 32

_model_cache: Dict[str, "SentenceTransformer"] = {}


def load_embedding_model(model_key: str):
    """Load (and cache) a SentenceTransformer model by key ('minilm' or 'mpnet')."""
    if model_key not in MODEL_NAMES:
        raise ValueError(f"Unknown model key: {model_key}. Expected one of {list(MODEL_NAMES)}")

    if model_key not in _model_cache:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s (%s)", model_key, MODEL_NAMES[model_key])
        _model_cache[model_key] = SentenceTransformer(MODEL_NAMES[model_key])

    return _model_cache[model_key]


def embed_texts(
    texts: List[str],
    model,
    batch_size: int = DEFAULT_BATCH_SIZE,
    normalize: bool = True,
) -> np.ndarray:
    """Embed a list of texts with the given model, in batches."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=len(texts) > batch_size,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def generate_embeddings(
    chunks_json: str,
    model_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    normalize: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Generate embeddings for all chunks in a chunks JSON file.

    Args:
        chunks_json: Path to a chunks_*.json file produced by src.ingestion.
        model_name: Model key ("minilm" or "mpnet").
        batch_size: Batch size for encoding.
        normalize: Whether to L2-normalize embeddings.

    Returns:
        (embeddings_array of shape (n_chunks, dim), chunk_ids list)
    """
    chunks: List[Chunk] = load_chunks(chunks_json)
    model = load_embedding_model(model_name)

    chunk_ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]

    embeddings = embed_texts(texts, model, batch_size=batch_size, normalize=normalize)
    logger.info(
        "Generated %d embeddings (dim=%d) with model=%s from %s",
        embeddings.shape[0],
        embeddings.shape[1] if embeddings.ndim == 2 else 0,
        model_name,
        chunks_json,
    )
    return embeddings, chunk_ids


def save_embeddings(embeddings: np.ndarray, chunk_ids: List[str], path: str) -> None:
    """Save embeddings and chunk_ids to a compressed .npz file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        chunk_ids=np.array(chunk_ids, dtype=object),
    )
    logger.info("Saved embeddings -> %s", path)


def load_embeddings(path: str) -> Tuple[np.ndarray, List[str]]:
    """Load embeddings and chunk_ids from a .npz file."""
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], list(data["chunk_ids"])


def embed_all_strategies(
    data_dir: str = "data",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """Generate embeddings for every (strategy, model) combination.

    Reads chunks_semantic.json / chunks_sentence.json from data_dir and writes
    embeddings_{strategy}_{model}.npz plus embedding_metadata.json to data_dir.
    """
    data_path = Path(data_dir)
    metadata: Dict[str, dict] = {}

    for strategy, chunk_filename in CHUNK_FILES.items():
        chunks_path = data_path / chunk_filename
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Missing chunks file for strategy '{strategy}': {chunks_path}. "
                "Run src.ingestion first."
            )

        for model_key in MODEL_NAMES:
            combo_name = f"{strategy}_{model_key}"
            embeddings, chunk_ids = generate_embeddings(
                str(chunks_path), model_key, batch_size=batch_size
            )

            npz_path = data_path / f"embeddings_{combo_name}.npz"
            save_embeddings(embeddings, chunk_ids, str(npz_path))

            metadata[f"embeddings_{combo_name}"] = {
                "chunk_id_map": {cid: i for i, cid in enumerate(chunk_ids)},
                "model": MODEL_NAMES[model_key],
                "dimensions": int(embeddings.shape[1]),
                "total_chunks": int(embeddings.shape[0]),
            }

    metadata_path = data_path / "embedding_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    logger.info("Saved embedding metadata -> %s", metadata_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    embed_all_strategies()
