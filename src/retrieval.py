"""Stage 3 (query side): retrieval of top-K relevant chunks for a query.

Embeds a user query with the same model used to embed the target collection,
then delegates the similarity search to a VectorStore implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.vectorstore import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrieverConfig:
    """Configuration for retrieval."""

    top_k: int = 5
    similarity_threshold: float = 0.15
    dedupe_by_document: bool = True
    candidate_pool_size: int = 20
    max_chunks_per_document: int = 2


class Retriever:
    """Retrieval engine that queries the vector store."""

    def __init__(
        self,
        vectorstore: VectorStore,
        embedding_models: Dict[str, object],
        config: Optional[RetrieverConfig] = None,
    ):
        """Initialize retriever with a vector store and one embedding model per model key.

        Args:
            vectorstore: Initialized VectorStore (e.g. ChromaVectorStore).
            embedding_models: Mapping of model key ("minilm"/"mpnet") to a
                loaded SentenceTransformer instance.
            config: Optional default retrieval configuration.
        """
        self.vectorstore = vectorstore
        self.embedding_models = embedding_models
        self.config = config or RetrieverConfig()

    def _model_key_for_collection(self, collection_name: str) -> str:
        for model_key in self.embedding_models:
            if collection_name.endswith(f"_{model_key}"):
                return model_key
        raise ValueError(
            f"Could not infer embedding model from collection_name='{collection_name}'"
        )

    def _embed_query(self, query: str, model_key: str) -> np.ndarray:
        """Embed a query using the model matching the target collection."""
        model = self.embedding_models[model_key]
        embedding = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(embedding, dtype=np.float32)

    def _validate_query_embedding(self, embedding: np.ndarray) -> None:
        if embedding is None or embedding.size == 0:
            raise ValueError("Query embedding is empty")
        if not np.any(embedding):
            raise ValueError("Query embedding is all zeros")
        if not np.all(np.isfinite(embedding)):
            raise ValueError("Query embedding contains non-finite values")

    @staticmethod
    def _cap_chunks_per_document(
        results: List[RetrievedChunk], max_per_document: int
    ) -> List[RetrievedChunk]:
        """Keep at most `max_per_document` best-scoring chunks per source document.

        Corpora with a wide spread of document lengths can have one document
        contribute disproportionately many chunks (e.g. a huge changelog),
        which crowds out the correct document across unrelated queries purely
        because it has more shots at the top-K. Capping chunks per document
        (still ranked by similarity, no separate reranking model) restores
        document-level diversity while still allowing a couple of chunks from
        the same document through for queries that need multiple sections of
        one document to answer (e.g. multi-passage synthesis questions).
        """
        counts: Dict[int, int] = {}
        capped: List[RetrievedChunk] = []
        for chunk in sorted(results, key=lambda r: r.similarity_score, reverse=True):
            count = counts.get(chunk.document_index, 0)
            if count >= max_per_document:
                continue
            counts[chunk.document_index] = count + 1
            capped.append(chunk)
        return capped

    def retrieve(
        self,
        query: str,
        collection_name: str,
        k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve top-K relevant chunks for a query.

        Args:
            query: User query string.
            collection_name: Name of the collection to search
                (e.g. "collection_semantic_minilm").
            k: Number of results (defaults to config.top_k).
            similarity_threshold: Minimum similarity score (defaults to
                config.similarity_threshold).

        Returns:
            List of RetrievedChunk objects sorted by similarity (descending).
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        model_key = self._model_key_for_collection(collection_name)
        query_embedding = self._embed_query(query, model_key)
        self._validate_query_embedding(query_embedding)

        target_k = k if k is not None else self.config.top_k
        threshold = (
            similarity_threshold if similarity_threshold is not None else self.config.similarity_threshold
        )
        pool_k = max(target_k, self.config.candidate_pool_size) if self.config.dedupe_by_document else target_k

        results = self.vectorstore.search(
            query_embedding=query_embedding,
            k=pool_k,
            similarity_threshold=threshold,
            collection_name=collection_name,
        )

        if self.config.dedupe_by_document:
            results = self._cap_chunks_per_document(results, self.config.max_chunks_per_document)

        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:target_k]
