"""Stage 3: Vector store abstraction and Chroma-backed implementation.

`VectorStore` is the abstract interface that retrieval/generation code
depends on. `ChromaVectorStore` is the only concrete implementation today;
future backends (Qdrant, Pinecone, ...) can be added by subclassing
`VectorStore` without touching retrieval or generation code.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

COLLECTION_NAMES = [
    "collection_semantic_minilm",
    "collection_semantic_mpnet",
    "collection_sentence_minilm",
    "collection_sentence_mpnet",
]


@dataclass
class RetrievedChunk:
    """Result from a vector store search."""

    chunk_id: str
    text: str
    document_index: int
    source_url: str
    similarity_score: float
    metadata: dict


class VectorStore(ABC):
    """Abstract interface for vector stores.

    `collection_name` is optional: implementations backing a single logical
    index can ignore it, while implementations managing multiple named
    indexes (like ChromaVectorStore, which holds one collection per
    chunking-strategy/embedding-model combination) require it.
    """

    @abstractmethod
    def add_chunks(
        self,
        chunk_ids: List[str],
        texts: List[str],
        embeddings: np.ndarray,
        metadata: List[dict],
        collection_name: Optional[str] = None,
    ) -> None:
        """Add embedded chunks to the store."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        similarity_threshold: float = 0.0,
        collection_name: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Search for the top-K most similar chunks."""
        raise NotImplementedError

    @abstractmethod
    def delete_all(self, collection_name: Optional[str] = None) -> None:
        """Delete all data from the store (or a single collection)."""
        raise NotImplementedError

    @abstractmethod
    def persist(self, path: Optional[str] = None) -> None:
        """Save the store to disk."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Optional[str] = None) -> None:
        """Load the store from disk."""
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    """Chroma-backed implementation of VectorStore, one collection per
    (chunking strategy, embedding model) combination."""

    def __init__(self, persist_dir: str = "data/chroma_db"):
        import chromadb

        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        self._collections: Dict[str, "chromadb.Collection"] = {}

    def _require_collection_name(self, collection_name: Optional[str]) -> str:
        if not collection_name:
            raise ValueError("collection_name is required for ChromaVectorStore")
        return collection_name

    def _get_collection(self, collection_name: str):
        if collection_name not in self._collections:
            self._collections[collection_name] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[collection_name]

    def add_chunks(
        self,
        chunk_ids: List[str],
        texts: List[str],
        embeddings: np.ndarray,
        metadata: List[dict],
        collection_name: Optional[str] = None,
    ) -> None:
        name = self._require_collection_name(collection_name)
        if not (len(chunk_ids) == len(texts) == len(embeddings) == len(metadata)):
            raise ValueError("chunk_ids, texts, embeddings, and metadata must have equal length")

        collection = self._get_collection(name)

        batch_size = 5000
        for start in range(0, len(chunk_ids), batch_size):
            end = start + batch_size
            collection.upsert(
                ids=chunk_ids[start:end],
                embeddings=np.asarray(embeddings[start:end]).tolist(),
                documents=texts[start:end],
                metadatas=metadata[start:end],
            )

        logger.info("Added %d chunks to collection '%s'", len(chunk_ids), name)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        similarity_threshold: float = 0.0,
        collection_name: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        name = self._require_collection_name(collection_name)
        collection = self._get_collection(name)

        count = collection.count()
        if count == 0:
            return []

        query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        result = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(k, count),
        )

        retrieved: List[RetrievedChunk] = []
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
            similarity = 1.0 - distance
            if similarity < similarity_threshold:
                continue
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    document_index=int(meta.get("document_index", -1)),
                    source_url=meta.get("source_url", ""),
                    similarity_score=float(similarity),
                    metadata=meta,
                )
            )

        retrieved.sort(key=lambda r: r.similarity_score, reverse=True)
        return retrieved

    def delete_all(self, collection_name: Optional[str] = None) -> None:
        if collection_name:
            names = [collection_name]
        else:
            names = [c.name for c in self.client.list_collections()]

        for name in names:
            self.client.delete_collection(name)
            self._collections.pop(name, None)
        logger.info("Deleted collections: %s", names)

    def persist(self, path: Optional[str] = None) -> None:
        # PersistentClient writes through automatically; nothing extra needed.
        logger.info("Chroma collections persisted at %s", path or self.persist_dir)

    def load(self, path: Optional[str] = None) -> None:
        import chromadb

        self.persist_dir = path or self.persist_dir
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self._collections = {}


def build_vectorstore(
    data_dir: str = "data",
    persist_dir: str = "data/chroma_db",
) -> ChromaVectorStore:
    """Load pre-computed chunks + embeddings from data_dir and populate all
    four Chroma collections."""
    from src.ingestion import load_chunks
    from src.embedding import MODEL_NAMES, load_embeddings

    data_path = Path(data_dir)
    store = ChromaVectorStore(persist_dir=persist_dir)

    chunk_files = {
        "semantic": data_path / "chunks_semantic.json",
        "sentence": data_path / "chunks_sentence.json",
    }

    for strategy, chunks_path in chunk_files.items():
        chunks_by_id = {c.chunk_id: c for c in load_chunks(str(chunks_path))}

        for model_key in MODEL_NAMES:
            combo_name = f"{strategy}_{model_key}"
            npz_path = data_path / f"embeddings_{combo_name}.npz"
            embeddings, chunk_ids = load_embeddings(str(npz_path))

            texts = [chunks_by_id[cid].text for cid in chunk_ids]
            metadata = [
                {
                    "chunk_id": cid,
                    "document_index": chunks_by_id[cid].document_index,
                    "source_url": chunks_by_id[cid].source_url,
                    "strategy": strategy,
                    "model": model_key,
                }
                for cid in chunk_ids
            ]

            store.add_chunks(
                chunk_ids=list(chunk_ids),
                texts=texts,
                embeddings=embeddings,
                metadata=metadata,
                collection_name=f"collection_{combo_name}",
            )

    store.persist()
    return store


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_vectorstore()
