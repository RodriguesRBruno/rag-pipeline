# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **single-day RAG (Retrieval-Augmented Generation) pipeline** learning project. The system ingests Enter the Gungeon video game documentation, chunks it using two strategies, embeds it with two models, stores it in a vector database, and generates grounded, cited answers to user questions.

**Key Constraint**: All implementation must be completed in a single day (8-10 hours).

## Project Structure

```
rag-pipeline/
├── main.py                       # Entry point: builds cached artifacts if missing, runs Q&A loop
├── src/                          # Core implementation modules
│   ├── ingestion.py              # Load docs, implement two chunking strategies
│   ├── embedding.py              # Generate embeddings with MiniLM and MPNet
│   ├── vectorstore.py            # Abstract interface + Chroma implementation
│   ├── retrieval.py              # Query vector store, return top-K results
│   ├── generation.py             # Call Claude API, format responses with citations
│   └── pipeline.py               # Orchestrate all stages
├── dataset/                      # CSV files: documents.csv, question sets
├── data/                         # Generated artifacts (gitignored, rebuilt on demand)
│   ├── chunks_*.json             # Chunked documents per strategy
│   ├── embeddings_*.npz          # Embeddings per strategy/model combination
│   └── chroma_db/                # Persisted vector store (created at runtime)
├── eval/                         # Evaluation scripts (to be created)
│   ├── evaluate.py               # Compute metrics (Recall, Precision, MRR, NDCG, etc.)
│   └── comparison_report.py       # Generate comparison table across strategies
├── spec/
│   └── SPEC.md                   # Full specification document
├── pyproject.toml                # Dependencies via uv (add packages here)
├── README.md                     # Project overview
└── CLAUDE.md                     # This file
```

## Architecture: 4-Stage Pipeline

The RAG system flows through 4 modular stages:

### Stage 1: Ingestion & Chunking
- Load 9,374 documents from `dataset/documents.csv`
- Implement two distinct chunking strategies:
  1. **Semantic Splitting**: Split by topic/section boundaries (500-800 tokens, 10% overlap)
  2. **Sentence-based**: Split on complete sentences (300-500 tokens, minimal overlap)
- Output: `chunks_semantic.json` and `chunks_sentence.json` with metadata (document_index, source_url, tokens)

### Stage 2: Embedding & Vectorization
- Use two Hugging Face embedding models:
  1. **`all-MiniLM-L6-v2`**: 384 dims, fast, baseline
  2. **`all-mpnet-base-v2`**: 768 dims, higher quality
- Generate embeddings for all chunks from both strategies
- Output: 4 embedding files + metadata (one per strategy/model combination)

### Stage 3: Vector Store & Retrieval
- **Implementation**: Chroma (local, in-memory with disk persistence)
- **Critical Design**: Use abstraction interface (`VectorStore` base class) to allow future swaps (Chroma → Qdrant/Pinecone)
- Create 4 collections: one per (strategy × model) combination for fair comparison
- Implement `search(query, k=5)` that returns top-K chunks with similarity scores

### Stage 4: Generation & Citation
- **LLM**: Claude (Anthropic API) via `anthropic` SDK
- Format retrieved chunks as context, pass query to Claude
- Extract and validate citations (ensure they match source_urls)
- Output: JSON with question, answer, sources, is_answerable, confidence

## Development Commands

### Setup & Dependencies
```bash
# Install all dependencies (uv manages pyproject.toml)
uv sync

# Add a new dependency
uv add <package-name>

# Update dependencies
uv lock
```

### Running the Pipeline
```bash
# End-to-end: builds any missing cached artifacts under data/, then starts
# an interactive Q&A loop. Reuses caches from prior runs instead of
# recomputing them.
uv run --env-file .env main.py

# Force-rebuild a single stage (module form is required: these files import
# via `src.*` package paths, so they can't be run as bare file paths)
uv run --env-file .env python -m src.ingestion      # Chunk documents
uv run --env-file .env python -m src.embedding      # Generate embeddings
uv run --env-file .env python -m src.vectorstore    # Build the vector store
```

### Evaluation
```bash
# Run all metrics on all question sets
uv run eval.evaluate

# Generate comparison table
uv run eval.comparison_report
```

### Quick Testing
```bash
# Test a single query (strategy/model are constructor args, not query() args)
uv run --env-file .env python -c "from src.pipeline import RAGPipeline; p = RAGPipeline(strategy='semantic', model='minilm'); print(p.query('What do keybullet kin drop?'))"
```

## Key Implementation Details

### Chunking Strategies
- **Semantic**: Detect section breaks in documents (look for headers, newline patterns). Fallback to fixed 800-token windows with 100-token overlap if no headers found.
- **Sentence-based**: Use NLTK/spaCy for sentence tokenization, then group sentences to reach 300-500 token chunks.

### Embedding Models
Both from `sentence-transformers` on Hugging Face:
```python
from sentence_transformers import SentenceTransformer

model_minilm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model_mpnet = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
```

### Vector Store Abstraction Pattern
```python
from abc import ABC
class VectorStore(ABC):
    def add_chunks(self, chunks, embeddings): pass
    def search(self, query_embedding, k=5): pass
    def persist(self, path): pass
    def load(self, path): pass

class ChromaVectorStore(VectorStore):
    # Chroma-specific implementation
    pass

# Future: QdrantVectorStore, PineconeVectorStore inherit same interface
```

This allows swapping vector stores without changing retrieval/generation code.

### Citation Extraction
Responses must include sources. Format:
```json
{
  "question": "...",
  "answer": "...",
  "sources": [
    {"url": "source_url", "document_index": 42, "chunk_id": "semantic_mpnet_42_1"}
  ],
  "is_answerable": true,
  "confidence": 0.85
}
```

Validate that cited URLs are real (exist in dataset) and contain claimed information.

### Handling Unanswerable Questions
- If retrieval returns very low similarity (< 0.5), mark as unanswerable
- If LLM explicitly states info not in corpus, respect that
- **Never hallucinate**: don't answer from external knowledge if corpus doesn't contain answer

## Evaluation Metrics

The system is evaluated on 159 questions across 3 difficulty tiers:

### Single-Passage QA (62 questions)
- Answerable from a single document
- Target: ≥ 80% correctness

### Multi-Passage QA (58 questions)
- Require combining info from multiple documents
- Target: ≥ 70% correctness

### No-Answer QA (39 questions)
- Cannot be answered from corpus
- Target: ≥ 85% correct rejection (avoid hallucination)

### Metrics to Compute
For each (strategy × model) combination:
- **Recall@5**: % of relevant docs in top-5 results
- **Precision@5**: % of top-5 results that are relevant
- **MRR**: Mean Reciprocal Rank (inverse rank of first relevant doc)
- **NDCG@5**: Normalized Discounted Cumulative Gain
- **Correctness**: % of answers matching ground truth
- **Citation Accuracy**: 100% (no false citations allowed)
- **Hallucination Rate**: % of answers with info outside retrieved context
- **Unanswerable Detection**: % of no-answer questions correctly identified

**Comparison Table**: Compare metrics across all 4 (strategy × model) combinations. Show which performs best and why.

## Environment & Dependencies

### Python Version
- Python ≥ 3.12 (specified in `.python-version`)

### Key Dependencies (add to pyproject.toml via `uv add`)
```
sentence-transformers>=2.2.0    # Embedding models
chromadb>=0.4.0                 # Vector database
anthropic>=0.7.0                # Claude API
pandas>=2.0.0                   # CSV loading
numpy>=1.24.0                   # Numerical ops
nltk or spacy                   # Sentence tokenization
scikit-learn>=1.3.0             # Metrics (MRR, NDCG)
```

### Environment Variables
No `.env` loading happens in-app (no `python-dotenv`); variables must be real
process env vars. Put them in `.env` (gitignored, see `.env.example`) and
supply it via `--env-file`:
```bash
uv run --env-file .env main.py
```
```
ANTHROPIC_AUTH_TOKEN=sk-...   # Required for LLM generation
GENERATION_MODEL=claude-haiku-4-5  # Optional, this is the default
```

## Single-Day Execution Strategy

### Time Budget: 8-10 hours

**Morning (4-5 hours)**:
1. Set up dependencies in `pyproject.toml` (uv sync)
2. Implement ingestion (chunking strategies)
3. Implement embedding (both models)
4. Implement vector store (Chroma + abstraction interface)
5. Test basic retrieval on sample queries

**Afternoon (4-5 hours)**:
1. Implement LLM generation with citations
2. End-to-end pipeline integration
3. Implement evaluation metrics
4. Run evaluation on all 3 question sets
5. Generate comparison table and results

**Key for Speed**:
- Use pre-trained models (no fine-tuning)
- Don't over-engineer; focus on correctness
- Single consolidated evaluation script (run all metrics at once)
- Automate everything; minimal manual testing

## Deliverables

By end of day, these must exist:

1. **Working Pipeline**: 4 modular stages that chain together
2. **Comparison Table**: Real metrics across all 4 (strategy × model) combinations in a markdown table
3. **Evaluation Report**: Results on all 159 questions with analysis of successes/failures
4. **Larger Corpus Write-up**: One paragraph on what would change for 10x/100x larger corpus (distributed processing, better indexing, etc.)

All code must be in `src/`, evaluation scripts in `eval/`, results in `RESULTS.md`.

## Debugging & Common Issues

### ImportError with sentence_transformers
- Ensure `uv sync` ran successfully
- Models auto-download on first use; check internet connection
- Models cache to `~/.cache/huggingface/`

### Chroma persistence issues
- Collections saved to `./data/chroma_db/`
- If persistence fails, check file permissions
- Delete `chroma_db/` to start fresh if corrupted

### Claude API timeout
- Set `timeout=30` on API calls
- Check `ANTHROPIC_AUTH_TOKEN` is set correctly
- Monitor rate limits (shouldn't hit with 159 questions)

### Chunking produces too many/too few chunks
- Semantic: adjust target_size (800), overlap_ratio (0.1)
- Sentence-based: adjust target_size (400), min/max sizes
- Verify tokenization is working (print first chunk to inspect)

## Notes for Future Implementations

- **Abstraction design** in vector store is intentionally strict to enable future swaps. Don't bypass it.
- **Metadata preservation** on chunks is critical for citations. Every chunk must carry document_index and source_url.
- **Evaluation against ground truth** is automated; ground truth comes from the 3 CSV files (single/multi/no-answer).
- **Single-day constraint** means prioritize working over perfect. A 70% correct system delivered is better than a 90% correct system that's incomplete.
