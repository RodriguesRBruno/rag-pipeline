"""Stage orchestration: RAGPipeline ties ingestion output, embeddings, the
vector store, retrieval, and generation together behind a single interface.

`build_pipeline` builds whichever of chunks/embeddings/Chroma vector store
are missing from disk (see src.ingestion, src.embedding, src.vectorstore),
reusing anything already present, and returns a ready-to-query RAGPipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from src.embedding import MODEL_NAMES, embed_all_strategies, load_embedding_model
from src.generation import Generator, GroundedResponse
from src.ingestion import run_ingestion
from src.retrieval import Retriever, RetrieverConfig
from src.vectorstore import ChromaVectorStore, build_vectorstore

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data")
CHUNK_FILENAMES = ["chunks_semantic.json", "chunks_sentence.json"]
STRATEGIES = ["semantic", "sentence"]


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


def ensure_ingestion(data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Run ingestion unless both chunk files already exist in data_dir."""
    if all((data_dir / name).exists() for name in CHUNK_FILENAMES):
        logger.info("Ingestion cache found in %s, skipping.", data_dir)
        return
    logger.info("No ingestion cache found, running ingestion...")
    run_ingestion(data_dir=str(data_dir))


def ensure_embeddings(data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Generate embeddings unless every (strategy, model) .npz file already exists."""
    expected = [
        data_dir / f"embeddings_{strategy}_{model}.npz"
        for strategy in STRATEGIES
        for model in MODEL_NAMES
    ]
    if all(path.exists() for path in expected):
        logger.info("Embedding cache found in %s, skipping.", data_dir)
        return
    logger.info("No embedding cache found, generating embeddings...")
    embed_all_strategies(data_dir=str(data_dir))


def ensure_vectorstore(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Build the Chroma vector store unless it's already persisted on disk."""
    persist_dir = data_dir / "chroma_db"
    if (persist_dir / "chroma.sqlite3").exists():
        logger.info("Vector store found at %s, skipping build.", persist_dir)
        return persist_dir
    logger.info("No vector store found, building...")
    build_vectorstore(data_dir=str(data_dir), persist_dir=str(persist_dir))
    return persist_dir


def build_pipeline(data_dir: Path = DEFAULT_DATA_DIR, log_level: str = "INFO") -> RAGPipeline:
    """Build any missing pipeline artifacts under data_dir, then return a ready RAGPipeline."""
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")
    ensure_ingestion(data_dir)
    ensure_embeddings(data_dir)
    ensure_vectorstore(data_dir)
    return RAGPipeline(data_dir=str(data_dir), log_level=log_level)
