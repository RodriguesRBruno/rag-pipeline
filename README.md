# RAG Pipeline

RAG project for learning using the Kaggle dataset available at https://www.kaggle.com/datasets/samuelmatsuoharris/single-topic-rag-evaluation-dataset

## Setup

```bash
uv sync
```

Create a `.env` file (see `.env.example`) with your Claude credentials:

```
ANTHROPIC_AUTH_TOKEN=<your token>
# Optional, defaults to claude-haiku-4-5
# GENERATION_MODEL=claude-haiku-4-5
```

## Running

```bash
uv run --env-file .env main.py
```

This builds any pipeline artifacts missing from `data/` (chunked documents, embeddings, and the Chroma vector store), reusing whatever is already there from a previous run instead of recomputing it, then drops you into an interactive prompt where you can ask questions and get grounded, cited answers. Type `quit` (or Ctrl-D) to exit.

To force a stage to rebuild, delete its output under `data/` (or delete `data/` entirely) and rerun `main.py`, or invoke a single stage directly:

```bash
uv run --env-file .env python -m src.ingestion    # (re)chunk documents
uv run --env-file .env python -m src.embedding    # (re)generate embeddings
uv run --env-file .env python -m src.vectorstore  # (re)build the vector store
```

Note: these stage modules import via `src.*` package paths, so they must be run with `python -m src.<module>` rather than as a bare file path (`python src/<module>.py`).
