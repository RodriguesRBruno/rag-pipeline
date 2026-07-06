"""Entry point: builds pipeline artifacts if missing, then runs an interactive Q&A loop."""

from __future__ import annotations

import logging
from pathlib import Path

from src.embedding import MODEL_NAMES, embed_all_strategies
from src.ingestion import run_ingestion
from src.pipeline import RAGPipeline
from src.vectorstore import build_vectorstore

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
CHUNK_FILENAMES = ["chunks_semantic.json", "chunks_sentence.json"]
STRATEGIES = ["semantic", "sentence"]


def ensure_ingestion(data_dir: Path = DATA_DIR) -> None:
    if all((data_dir / name).exists() for name in CHUNK_FILENAMES):
        logger.info("Ingestion cache found in %s, skipping.", data_dir)
        return
    logger.info("No ingestion cache found, running ingestion...")
    run_ingestion(data_dir=str(data_dir))


def ensure_embeddings(data_dir: Path = DATA_DIR) -> None:
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


def ensure_vectorstore(data_dir: Path = DATA_DIR) -> Path:
    persist_dir = data_dir / "chroma_db"
    if (persist_dir / "chroma.sqlite3").exists():
        logger.info("Vector store found at %s, skipping build.", persist_dir)
        return persist_dir
    logger.info("No vector store found, building...")
    build_vectorstore(data_dir=str(data_dir), persist_dir=str(persist_dir))
    return persist_dir


def build_pipeline(data_dir: Path = DATA_DIR) -> RAGPipeline:
    ensure_ingestion(data_dir)
    ensure_embeddings(data_dir)
    ensure_vectorstore(data_dir)
    return RAGPipeline(data_dir=str(data_dir))


def qa_loop(pipeline: RAGPipeline) -> None:
    print("\nRAG pipeline ready. Ask a question about Enter the Gungeon (or type 'quit' to exit).\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break

        response = pipeline.query(question)
        print(f"\n{response.answer}\n")
        if response.sources:
            print("Sources:")
            for source in response.sources:
                print(f"  - {source.url} (document #{source.document_index})")
        print(f"[answerable={response.is_answerable}, confidence={response.confidence:.2f}]\n")


def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    pipeline = build_pipeline()
    qa_loop(pipeline)


if __name__ == "__main__":
    main()
