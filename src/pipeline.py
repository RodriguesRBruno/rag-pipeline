"""Stage orchestration: RAGPipeline ties ingestion output, embeddings, the
vector store, retrieval, and generation together behind a single interface.

Assumes chunks, embeddings, and the Chroma vector store have already been
built on disk (see src.ingestion, src.embedding, src.vectorstore). This
module only loads/queries them; it does not (re)build them.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from tqdm import tqdm

from src.embedding import MODEL_NAMES, load_embedding_model
from src.generation import Generator, GroundedResponse
from src.retrieval import Retriever, RetrieverConfig
from src.vectorstore import ChromaVectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """High-level RAG pipeline orchestrator."""

    VALID_STRATEGIES = ["semantic", "sentence"]
    VALID_MODELS = list(MODEL_NAMES)

    def __init__(
        self,
        strategy: str = "semantic",
        model: str = "minilm",
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        data_dir: str = "data",
        persist_dir: Optional[str] = None,
        log_level: str = "INFO",
    ):
        """Initialize the RAG pipeline.

        Args:
            strategy: Chunking strategy to query ("semantic" or "sentence").
            model: Embedding model to query ("minilm" or "mpnet").
            top_k: Default number of chunks to retrieve per query.
            similarity_threshold: Default minimum similarity for retrieval.
            data_dir: Directory holding chunks/embeddings artifacts.
            persist_dir: Chroma persistence directory (defaults to data_dir/chroma_db).
            log_level: Logging level for pipeline events.
        """
        logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")
        self._validate_config(strategy, model)

        self.data_dir = data_dir
        self.persist_dir = persist_dir or f"{data_dir}/chroma_db"
        self.strategy = strategy
        self.model = model

        self.vectorstore = ChromaVectorStore(persist_dir=self.persist_dir)
        self.embedding_models: Dict[str, object] = {
            key: load_embedding_model(key) for key in MODEL_NAMES
        }
        self.retriever = Retriever(
            vectorstore=self.vectorstore,
            embedding_models=self.embedding_models,
            config=RetrieverConfig(top_k=top_k, similarity_threshold=similarity_threshold),
        )
        self.generator = Generator()

        logger.info("RAGPipeline initialized with strategy=%s model=%s", strategy, model)

    @classmethod
    def _validate_config(cls, strategy: str, model: str) -> None:
        if strategy not in cls.VALID_STRATEGIES:
            raise ValueError(f"Invalid strategy '{strategy}'. Expected one of {cls.VALID_STRATEGIES}")
        if model not in cls.VALID_MODELS:
            raise ValueError(f"Invalid model '{model}'. Expected one of {cls.VALID_MODELS}")

    @property
    def collection_name(self) -> str:
        return f"collection_{self.strategy}_{self.model}"

    def set_strategy(self, strategy: str, model: str) -> None:
        """Switch to a different chunking-strategy/embedding-model combination."""
        self._validate_config(strategy, model)
        self.strategy = strategy
        self.model = model
        logger.info("Switched pipeline to strategy=%s model=%s", strategy, model)

    def query(self, user_query: str) -> GroundedResponse:
        """Process a single query end-to-end: retrieve, then generate."""
        chunks = self.retriever.retrieve(user_query, collection_name=self.collection_name)
        logger.info("Retrieved %d chunks for query: %r", len(chunks), user_query)

        response = self.generator.generate(user_query, chunks)
        logger.info(
            "Generated response (answerable=%s, confidence=%.2f) for query: %r",
            response.is_answerable,
            response.confidence,
            user_query,
        )
        return response

    def query_batch(
        self,
        queries: List[str],
        show_progress: bool = True,
    ) -> List[GroundedResponse]:
        """Process multiple queries, continuing past individual failures."""
        results = []
        iterator = tqdm(queries, desc="Querying") if show_progress else queries
        for q in iterator:
            try:
                results.append(self.query(q))
            except Exception:
                logger.exception("Query failed, skipping: %r", q)
                results.append(
                    GroundedResponse(question=q, answer="", sources=[], is_answerable=False, confidence=0.0)
                )
        return results

    def get_config(self) -> dict:
        """Return the current pipeline configuration."""
        return {
            "strategy": self.strategy,
            "model": self.model,
            "collection_name": self.collection_name,
            "data_dir": self.data_dir,
            "persist_dir": self.persist_dir,
            "top_k": self.retriever.config.top_k,
            "similarity_threshold": self.retriever.config.similarity_threshold,
        }

    def health_check(self) -> bool:
        """Verify all components are initialized and the active collection is queryable."""
        try:
            self.retriever.retrieve("health check", collection_name=self.collection_name, k=1)
            return True
        except Exception:
            logger.exception("Health check failed")
            return False
