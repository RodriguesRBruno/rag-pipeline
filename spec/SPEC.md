# RAG Pipeline Specification

## 1. Project Overview

**Title**: RAG Pipeline (Mixed-Topic Small Corpus)
**Type**: Retrieval-Augmented Generation (RAG) System
**Purpose**: A learning project to build an end-to-end RAG pipeline that ingests a small, mixed-topic document corpus, performs semantic search across multiple embedding models, and generates grounded, cited answers using an LLM.

This project implements a modular RAG system that retrieves relevant documents from the corpus described in Section 2 and uses them to answer user questions with proper source citations. The system demonstrates how to build a retrieval pipeline while experimenting with different chunking and embedding strategies to compare retrieval quality.

---

## 2. Dataset & Corpus

### 2.1 Data Source
- **Source**: Kaggle "Single Topic RAG Evaluation Dataset" (MIT License). The full upstream dataset is a 9,374-document, single-topic (Enter the Gungeon) corpus with 159 evaluation questions.
- **What's actually in this repository**: `dataset/` contains a much smaller sample pulled from that same source structure, not the full upstream dataset. All figures in this spec describe that sample, confirmed by inspection in `src/ingestion.py` and `eval/evaluate.py`.

### 2.2 Domain
- **Nominal topic**: Enter the Gungeon (indie roguelike video game) — this is the origin and namesake of the source dataset.
- **Actual content**: Only 1 of the 20 documents (`index=0`, "Bullet Kin") is about Enter the Gungeon. The other 19 documents cover unrelated topics — D&D campaign notes, RAG/LLM tooling, cooking, films, GPUs, other video game wikis (e.g. Stardew Valley), and similar. In practice the corpus is a **generic multi-topic retrieval sample**, not a single-topic Gungeon corpus.
- **Content Type**: Wikipedia/Fandom-style and blog-style documentation, mixed in subject matter as described above.

### 2.3 Dataset Files

#### `documents.csv`
- **Records**: 20 documents
- **Columns**:
  - `index`: Unique document identifier (0-indexed)
  - `source_url`: Original source URL
  - `text`: Full document text content
- **Purpose**: Primary corpus for ingestion and retrieval
- **Note**: Document sizes vary widely — one document (`index=16`, a game changelog) is ~211KB and dominates naive top-K retrieval by chunk count alone unless mitigated (see `src/retrieval.py`'s per-document cap in Section 5.4).

#### `single_passage_answer_questions.csv`
- **Records**: 40 questions
- **Columns**:
  - `document_index`: Reference to documents.csv index
  - `question`: User question
  - `answer`: Ground truth answer sourced from a single document
- **Purpose**: Evaluation set for single-document retrieval (baseline difficulty)
- **Example**: "What do keybullet kin drop?" → "Keybullet kin drop a key upon death."

#### `multi_passage_answer_questions.csv`
- **Records**: 40 questions
- **Columns**:
  - `document_index`: Ground truth references a single document index (see note below)
  - `question`: User question
  - `answer`: Ground truth answer requiring synthesis from multiple passages
- **Purpose**: Evaluation set for multi-passage retrieval (intermediate difficulty)
- **Example**: "Which enemy types wield an AK-47?" → Answer requires combining info from multiple enemy sub-sections
- **Note**: Despite the "multi-passage" label, ground truth in this file is a single `document_index` per question, same as the single-passage set. Retrieval relevance for this set is evaluated against that one document (see `eval/evaluate.py`); the "multi-passage" difficulty comes from needing multiple sections/sentences of that document to fully answer, not multiple documents.

#### `no_answer_questions.csv`
- **Records**: 40 questions
- **Columns**:
  - `document_index`: N/A (no correct answer exists in corpus)
  - `question`: User question that cannot be answered from the corpus
- **Purpose**: Evaluation set for detecting unanswerable questions (advanced difficulty)
- **Example**: "How much health does the Mutant Bullet Kin have?" (not documented in corpus)

### 2.4 Evaluation Strategy
The three question sets provide tiered evaluation:
- **Single-passage**: Tests basic retrieval accuracy (40 questions)
- **Multi-passage**: Tests synthesis across sections of a document (40 questions)
- **No-answer**: Tests hallucination prevention (40 questions)
- **Total**: 120 evaluation questions

None of the implementation in `src/` hard-codes the dataset sizes described above; ingestion, chunking, embedding, and retrieval all operate on whatever is actually present in `dataset/`, so the pipeline works unmodified if the corpus is later expanded.

---

## 3. Functional Requirements

### 3.1 Core Requirements
- **FR-1**: Ingest documents from CSV and chunk them using at least two distinct strategies
- **FR-2**: Generate embeddings for all chunks using two different embedding models
- **FR-3**: Store embeddings and metadata in a vector database with semantic search capability
- **FR-4**: Retrieve top-K relevant chunks given a user query
- **FR-5**: Generate natural language answers using an LLM based on retrieved context
- **FR-6**: Include source citations in all generated responses
- **FR-7**: Explicitly state when a question cannot be answered from the corpus
- **FR-8**: Compare retrieval quality across chunking strategies and embedding models with quantitative metrics

### 3.2 Non-Functional Requirements
- **NF-1**: Code must be modular with clear separation of ingestion, retrieval, and generation stages
- **NF-2**: Vector store layer must use abstraction/interface to support swapping implementations (Chroma → Qdrant/Pinecone)
- **NF-3**: System should handle the full corpus present in `dataset/` without performance degradation, and scale to a larger corpus without code changes
- **NF-4**: Results must be reproducible and comparable across runs

### 3.3 Out of Scope
- Fine-tuning embedding or LLM models
- Web UI or REST API service layer
- Advanced retrieval techniques (hybrid search, re-ranking, query expansion)
- Distributed/scalable implementation (addressed conceptually in corpus size write-up)

---

## 4. System Architecture

### 4.1 Pipeline Overview
The RAG pipeline consists of four modular stages that can be executed independently or as an orchestrated workflow:

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG PIPELINE STAGES                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  STAGE 1: INGESTION & CHUNKING                              │
│  ├─ Load documents.csv                                      │
│  ├─ Clean and preprocess text                               │
│  ├─ Split into chunks (Strategy 1: Semantic)                │
│  ├─ Split into chunks (Strategy 2: Sentence-based)          │
│  └─ Output: List[Chunk] with metadata                       │
│                                                              │
│  STAGE 2: EMBEDDING & VECTORIZATION                         │
│  ├─ Model 1: all-MiniLM-L6-v2 (384 dims)                    │
│  ├─ Model 2: all-mpnet-base-v2 (768 dims)                    │
│  ├─ Generate embeddings for each chunk                      │
│  ├─ Store metadata: source_url, chunk_id, document_index    │
│  └─ Output: List[EmbeddedChunk]                             │
│                                                              │
│  STAGE 3: VECTOR STORE & RETRIEVAL                          │
│  ├─ Store embedded chunks in Chroma                         │
│  ├─ Maintain abstraction interface for swappability         │
│  ├─ Implement semantic search (cosine similarity)           │
│  ├─ Support multi-model querying                            │
│  └─ Output: List[TopKResults] sorted by similarity          │
│                                                              │
│  STAGE 4: GENERATION & CITATION                             │
│  ├─ Format retrieval context with sources                   │
│  ├─ Call Claude API with context + question                 │
│  ├─ Extract citations from response                         │
│  ├─ Verify citations match retrieved documents               │
│  └─ Output: GroundedResponse with citations                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Stage 1: Ingestion & Chunking

**Responsibility**: Load raw documents and chunk them using two distinct strategies.

**Inputs**:
- `dataset/documents.csv`

**Processing**:

1. **Load & Preprocess**
   - Read CSV file
   - Clean HTML entities and special characters
   - Remove extra whitespace
   - Preserve document_index and source_url for all chunks

2. **Chunking Strategy 1: Semantic Splitting**
   - Split documents by semantic boundaries (topic/section headers)
   - Preserve sentence completeness
   - Target chunk size: 500-800 tokens
   - Allow semantic overlap: chunks may share boundary content
   - Implementation: Use libraries like `semantic-text-splitter` or custom section-based splitting

3. **Chunking Strategy 2: Sentence-based Splitting**
   - Split documents into complete sentences
   - Group consecutive sentences to reach target chunk size: 300-500 tokens
   - Ensure no sentence is split
   - Minimal overlap: only adjacent chunks share sentence boundaries
   - Implementation: Use NLTK or spaCy for sentence tokenization

**Outputs**:
- `chunks_semantic.json`: Semantic chunks with metadata
- `chunks_sentence.json`: Sentence-based chunks with metadata
- Each chunk record: `{chunk_id, text, document_index, source_url, tokens, strategy}`

**Metadata Preservation**:
- Every chunk must include original `document_index` to enable source tracking
- Every chunk must include `source_url` for citation purposes
- Token count for each chunk (for embedding cost calculation)

---

### 4.3 Stage 2: Embedding & Vectorization

**Responsibility**: Generate embeddings for all chunks using two embedding models.

**Inputs**:
- `chunks_semantic.json`
- `chunks_sentence.json`

**Embedding Models**:

| Model | Dimensions | Provider | Speed | Quality | Use Case |
|-------|-----------|----------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | Hugging Face | Fast | Good | Baseline comparison, speed testing |
| `all-mpnet-base-v2` | 768 | Hugging Face | Slower | Better | High-quality retrieval |

**Processing for Each Model**:
1. Load pretrained model from Hugging Face
2. Tokenize chunk text (batch size: 32 for efficiency)
3. Generate embeddings with model
4. Normalize embeddings (L2 normalization)
5. Store embedding vector with metadata

**Outputs** (per strategy per model, on the actual 20-document corpus):
- `embeddings_semantic_minilm.npz`: ~144 chunks × 384 dims
- `embeddings_semantic_mpnet.npz`: ~144 chunks × 768 dims
- `embeddings_sentence_minilm.npz`: ~307 chunks × 384 dims (more chunks due to smaller target size)
- `embeddings_sentence_mpnet.npz`: ~307 chunks × 768 dims
- `embedding_metadata.json`: {chunk_id: {model, embedding_shape, norm}}

These chunk counts scale linearly with corpus size and are not hard-coded anywhere in `src/`.

**Storage Format**:
- NumPy .npz format for efficiency
- JSON metadata for chunk information
- SQLite or JSON index for chunk_id → embedding lookup

---

### 4.4 Stage 3: Vector Store & Retrieval

**Responsibility**: Store embeddings in vector database and implement semantic search.

**Vector Store Choice**: Chroma (local, in-memory with persistence)

**Architecture Requirement - Abstraction Interface**:
```python
# Abstract base class for vector store compatibility
class VectorStore(ABC):
    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> None
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[RetrievedChunk]
    def delete_all(self) -> None
    def persist(self, path: str) -> None
    def load(self, path: str) -> None

# Implementations
class ChromaVectorStore(VectorStore): ...
class QdrantVectorStore(VectorStore): ...  # Future swap
class PineconeVectorStore(VectorStore): ...  # Future swap
```

This design allows switching vector stores without modifying retrieval or generation code.

**Inputs**:
- Embedded chunks from Stage 2
- Metadata (source_url, document_index, etc.)

**Processing**:

1. **Initialize Chroma Collections** (one per strategy/model combination):
   - `collection_semantic_minilm`
   - `collection_semantic_mpnet`
   - `collection_sentence_minilm`
   - `collection_sentence_mpnet`

2. **Add Embeddings to Collections**:
   - Store embedding vector
   - Store metadata: `{chunk_id, source_url, document_index, text, strategy, model}`

3. **Implement Search**:
   - Accept query string and user choice of (strategy, model)
   - Embed query using same model that embedded the chunks
   - Query a larger candidate pool than K, cap results per source document, then take the final top-K (see Section 5.4 rationale)
   - Apply a similarity threshold to filter out near-zero matches
   - Return ranked results with scores

**Outputs**:
- 4 persistent Chroma collections (one per strategy/model pair)
- Retrieved chunks with similarity scores

**Key Design Decision**: Each combination of chunking strategy and embedding model gets its own collection to enable fair comparison.

---

### 4.5 Stage 4: Generation & Citation

**Responsibility**: Use retrieved context to generate grounded answers with citations.

**Inputs**:
- User query
- Top-K retrieved chunks from Stage 3
- Ground truth (for evaluation)

**LLM Model**: Claude (Anthropic API) - via `anthropic` SDK

**Processing**:

1. **Format Context**:
   ```
   <context>
   Source 1: [source_url]
   [chunk text 1]

   Source 2: [source_url]
   [chunk text 2]

   Source 3: [source_url]
   [chunk text 3]
   </context>

   Question: [user question]
   ```

2. **Call LLM** with a system prompt that does not assume a single topic, since the corpus is mixed-topic (Section 2.2):
   ```
   You are a helpful assistant that answers questions using ONLY the provided
   context below - the context may cover any topic, not just Enter the Gungeon.
   Answer the question if the context contains the information, regardless of
   subject matter. Include citations to source documents by referencing their
   source URL. If the specific answer is not present in the context, respond
   exactly: "I don't have this information in my corpus."
   ```
   An earlier version of this prompt was scoped to "answering questions about Enter the Gungeon," which caused the LLM to refuse to answer using perfectly good retrieved context simply because the topic wasn't Gungeon (19 of 20 documents aren't). See `RESULTS.md` for the measured impact of generalizing it.

3. **Process Response**:
   - Extract generated text
   - Parse citations from response (matching either raw source URLs or "Source N" labels against the numbered context blocks)
   - Verify citations reference valid source_urls
   - Flag potential hallucinations (claims outside retrieved context)

4. **Format Output**:
   ```json
   {
     "question": "user question",
     "answer": "generated answer",
     "sources": [
       {"url": "source_url", "document_index": 123, "chunk_id": "xyz"},
       ...
     ],
     "is_answerable": true/false,
     "confidence": 0.85
   }
   ```

**Unanswerable Question Handling**:
- If retrieval returns very low similarity scores (below the configured threshold, Section 5.4)
- Or if the LLM explicitly states information is not in corpus
- Mark response as unanswerable
- Do NOT attempt to answer from external knowledge

---

## 5. Implementation Decisions

### 5.1 Embedding Models

**Model 1: `all-MiniLM-L6-v2`**
- Source: Sentence-Transformers (Hugging Face)
- Dimensions: 384
- Strengths: Fast, low memory, reasonable quality
- Purpose: Baseline for speed comparison
- Download: `sentence-transformers/all-MiniLM-L6-v2`

**Model 2: `all-mpnet-base-v2`**
- Source: Sentence-Transformers (Hugging Face)
- Dimensions: 768
- Strengths: Higher quality embeddings, better for complex queries
- Purpose: High-quality retrieval baseline
- Download: `sentence-transformers/all-mpnet-base-v2`

### 5.2 Chunking Strategies

**Strategy 1: Semantic Splitting**
- Split on topic/section boundaries where available
- Use header patterns or doc structure to identify boundaries
- Fallback to fixed 800-token chunks with 100-token overlap
- Preserves semantic meaning of content
- Actual result on the 20-document corpus: ~144 chunks

**Strategy 2: Sentence-based Splitting**
- Split on sentence boundaries (use NLTK/spaCy)
- Group sentences to reach 300-500 token chunks
- Minimal overlap (last sentence of previous chunk may repeat)
- Ensures no half-sentences
- Actual result on the 20-document corpus: ~307 chunks

### 5.3 Vector Store

**Primary Choice**: Chroma
- In-memory vector database with file persistence
- Simple Python API
- Good for learning and prototyping
- Comfortably handles the corpus's few hundred embedded chunks; would need re-evaluation at much larger scale (see Section 8.3)

**Persistence**: Store embeddings to disk for reproducibility
- Collections saved to `./data/chroma_db/`
- Reload on pipeline restart without re-embedding

**Modularity for Future**: Abstraction layer (see Section 4.4) allows:
- Easy migration to Qdrant (more production-ready)
- Easy migration to Pinecone (cloud-hosted)
- Zero changes to retrieval/generation code

### 5.4 Retrieval Strategy

- **Top-K Retrieval**: K=5 (retrieve 5 most similar chunks per query)
- **Similarity Threshold**: 0.15 (cosine similarity)
  - Calibrated empirically against this corpus's real embeddings: top-1 similarity for genuinely correct matches averages ~0.4 and overlaps heavily with no-answer questions' scores (~0.39), so a stricter threshold (e.g. 0.5) causes the system to refuse to answer almost everything. 0.15 filters only near-zero noise; unanswerable detection instead relies primarily on the LLM reading the retrieved text and stating explicitly when the specific fact isn't present.
- **Per-document cap**: retrieval pulls a larger candidate pool (20) and caps results to at most 2 chunks per source document before taking the final top-K, to prevent one oversized document (e.g. the ~211KB changelog at `document_index=16`) from crowding out relevant results for unrelated queries.
- **Aggregation**: If result is from multiple chunks, include all with high similarity, subject to the per-document cap above

### 5.5 LLM Configuration

**Model**: Claude (Anthropic API)
- **Authentication**: token via environment variable `ANTHROPIC_AUTH_TOKEN`
- **Model ID**: `claude-haiku-4-5` by default, overridable via the `GENERATION_MODEL` environment variable
- **Parameters**:
  - `temperature`: 0.3 (low creativity, focus on facts)
  - `max_tokens`: 500
  - `system_prompt`: Instruction to use only provided context, answer regardless of topic, and cite sources (Section 4.5)

### 5.6 Citation Format

**In Response Text**:
```
According to the documentation [Source], Enter the Gungeon is...
```

**Structured Output**:
```json
"sources": [
  {
    "url": "https://enterthegungeon.fandom.com/wiki/Bullet_Kin",
    "document_index": 0,
    "chunk_id": "semantic_mpnet_0_1"
  }
]
```

---

## 6. Evaluation & Metrics

### 6.1 Retrieval Quality Metrics

These metrics are computed for comparison across the four (strategy × model) combinations. None carries a fixed pass/fail target — see Section 6.4 for the metrics that do.

**Metric 1: Recall@K**
- Definition: Fraction of relevant documents in top-K results
- Calculation: `|relevant_docs ∩ top_k| / |relevant_docs|`
- Per question: Check if ground truth document_index appears in retrieved results

**Metric 2: Mean Reciprocal Rank (MRR)**
- Definition: Average of inverse rank of first relevant document
- Calculation: `(1/rank_of_first_relevant)` averaged over all queries
- Penalizes when correct doc appears late in ranking

**Metric 3: Normalized Discounted Cumulative Gain (NDCG@K)**
- Definition: Discounted relevance with normalization
- Calculation: `DCG@K / IDCG@K`
- Accounts for graded relevance (how relevant, not just binary)

**Precision@K is intentionally not computed.** With exactly one relevant document per question and K=5, Precision@5 is structurally capped at 0.2 for any system that retrieves the correct document at all — it cannot distinguish a good retriever from a mediocre one on this dataset and adds no signal beyond Recall@5. Recall@K, MRR, and NDCG@K are the meaningful retrieval metrics here.

### 6.2 Generation & Answer Quality Metrics

These metrics are computed for comparison across combinations. Only Hallucination Rate carries a fixed target (Section 6.4); Correctness and Citation Accuracy are comparison-only.

**Metric 4: Correctness (Automated)**
- Definition: Answer matches ground truth (token-overlap heuristic against key words in the ground truth answer)
- Calculation: Fraction of ground-truth answer's significant words that appear in the generated answer, thresholded

**Metric 5: Citation Accuracy**
- Definition: Cited sources actually match the ground-truth document (no citations to an irrelevant document)
- Calculation: Automated check that every cited `document_index` matches the question's ground-truth document

**Metric 6: Hallucination Rate** *(hard requirement, see 6.4)*
- Definition: Fraction of answers containing information NOT grounded in the retrieved context
- Calculation: Fraction of the generated answer's significant words that do not appear in the retrieved context, thresholded

### 6.3 Unanswerable Question Detection

**Metric 7: Unanswerable Question Detection** *(hard requirement, see 6.4)*
- Definition: Correctly identifies questions the corpus cannot answer
- Calculation: True negative rate on the no-answer question set

### 6.4 Hard Requirements

Given the single-day scope and the small, mixed-topic corpus actually available (Section 2), most metrics above are used purely to compare chunking strategies and embedding models against each other — there is no external ground truth for what a "good enough" Recall@5 or MRR looks like on this corpus. The project instead holds two behaviors to a fixed bar, since these are safety/trust properties rather than retrieval-quality tuning knobs:

- **Hallucination Rate**: target 0% — the system must never state information that isn't grounded in retrieved context.
- **Unanswerable Question Detection**: target ≥ 85% — the system must correctly recognize when a question can't be answered from the corpus, most of the time.

All other metrics (Recall@K, MRR, NDCG@K, Correctness, Citation Accuracy) are reported per combination in the comparison table (Section 6.5) and used to pick a "best" combination, without a fixed pass/fail bar.

### 6.5 Comparison Table Structure

Compare across:
- **Rows**: Chunking Strategy (Semantic, Sentence-based)
- **Columns**: Embedding Model (MiniLM, MPNet)
- **Cells**: Metric values (Recall@5, MRR, NDCG@5, Correctness, Citation Accuracy, Hallucination Rate, No-Answer Detection)

**Example Template**:

| Strategy | Model | Recall@5 | MRR | NDCG@5 | Correctness | Hallucination % |
|----------|-------|----------|-----|--------|-------------|-----------------|
| Semantic | MiniLM | 0.82 | 0.76 | 0.71 | 0.78 | 0.05 |
| Semantic | MPNet | 0.85 | 0.79 | 0.74 | 0.81 | 0.03 |
| Sentence | MiniLM | 0.75 | 0.68 | 0.64 | 0.71 | 0.08 |
| Sentence | MPNet | 0.79 | 0.72 | 0.68 | 0.75 | 0.06 |

---

## 7. Testing & Validation

### 7.1 Evaluation Sets

**Test 1: Single-Passage QA (40 questions)**
- Questions answerable from a single document
- Correctness is reported for comparison across combinations; no fixed pass/fail threshold (Section 6.4)

**Test 2: Multi-Passage QA (40 questions)**
- Questions requiring synthesis of multiple sections/sentences of a document (Section 2.3)
- Correctness is reported for comparison across combinations; no fixed pass/fail threshold (Section 6.4)

**Test 3: No-Answer QA (40 questions)**
- Questions where the corpus has no answer
- Expected: System should say "I don't have this information in my corpus."
- **Hard requirement**: ≥ 85% correct rejection rate (Section 6.4)

### 7.2 End-to-End Validation

**Pipeline Verification Checklist**:
1. [ ] Documents load successfully (20 records)
2. [ ] Semantic chunking produces reasonable chunks (inspect 5-10 manually)
3. [ ] Sentence chunking produces reasonable chunks (inspect 5-10 manually)
4. [ ] Embeddings generate without error (384/768 dims)
5. [ ] Vector stores initialize with all chunks
6. [ ] Queries retrieve relevant results (manual spot check on 5 queries)
7. [ ] LLM generates answers with citations
8. [ ] All metrics compute successfully
9. [ ] Comparison table populates with real numbers
10. [ ] No crashes or timeout errors

### 7.3 Baseline Expectations

- **Hallucination Rate**: 0% (hard requirement)
- **No-Answer Detection**: ≥ 85% correct rejection (hard requirement)
- **Single-Passage / Multi-Passage Correctness, Citation Accuracy**: tracked and compared across combinations, no fixed baseline

---

## 8. Deliverables

### 8.1 Working Pipeline

**Code Structure**:
```
rag-pipeline/
├── src/
│   ├── __init__.py
│   ├── ingestion.py          # Load docs, implement chunking
│   ├── embedding.py          # Generate embeddings
│   ├── vectorstore.py        # Abstract VectorStore interface + Chroma impl
│   ├── retrieval.py          # Query vector store
│   ├── generation.py         # Call LLM, format response
│   └── pipeline.py           # Orchestrate stages
├── data/                      # Generated artifacts (gitignored, rebuilt on demand)
│   ├── chunks_*.json
│   ├── embeddings_*.npz
│   └── chroma_db/              # Persisted vector store
├── dataset/                    # Original CSV files
├── eval/
│   ├── evaluate.py            # Compute metrics
│   └── comparison_report.py   # Generate comparison table
├── spec/
│   └── SPEC.md                # This specification document
├── README.md                   # Project overview & usage
└── pyproject.toml              # Project metadata (uv-based dependency management)
```

**Core Modules**:
- `ingestion.py`: `load_documents()`, `chunk_semantic()`, `chunk_sentence()`
- `embedding.py`: `embed_chunks_minilm()`, `embed_chunks_mpnet()`
- `vectorstore.py`: `VectorStore` abstract class, `ChromaVectorStore` implementation
- `retrieval.py`: `Retriever.retrieve(query, k=5)`
- `generation.py`: `Generator.generate(query, retrieved_chunks)`
- `pipeline.py`: `RAGPipeline` orchestrator class

### 8.2 Comparison Table

**Format**: Markdown table in `RESULTS.md`
- Rows: Chunking strategies (Semantic, Sentence-based)
- Columns: Embedding models (MiniLM, MPNet)
- Cells: Metrics (Recall@5, MRR, NDCG@5, Correctness, Citation Accuracy, Hallucination Rate, No-Answer Detection)
- Must include real numbers from evaluation
- Must show which combination performs best and why

### 8.3 Larger Corpus Write-up

**Format**: One paragraph (100-200 words)
**Location**: `RESULTS.md`
**Content**: Address what you would change if this dataset were 10x or 100x larger

**Guiding Questions**:
- How would chunking strategy change?
- What about embedding strategy?
- Would the vector store choice change? Why?
- How would you handle distributed processing?
- What about indexing and retrieval speed?

### 8.4 Evaluation Report

**Format**: Detailed report with results on all three question sets
**Location**: `RESULTS.md`
**Contents**:
- Table of metrics across all strategy/model combinations
- Per-question-type analysis (which questions failed and why)
- Examples of correct answers with citations
- Examples of failure modes (if any)
- Recommendations for improvement

---

## 9. Success Criteria

### 9.1 Functional Success Criteria
- [ ] **FR-1**: System ingests all documents present in `dataset/documents.csv` without errors
- [ ] **FR-2**: Both chunking strategies produce reasonable chunks
- [ ] **FR-3**: Embeddings generate for both models successfully
- [ ] **FR-4**: Vector store stores all embedded chunks with metadata
- [ ] **FR-5**: Retrieval returns top-K results with similarity scores
- [ ] **FR-6**: LLM generates answers using only retrieved context
- [ ] **FR-7**: All answers include source citations
- [ ] **FR-8**: System explicitly states when answers aren't in corpus
- [ ] **FR-9**: Comparison metrics computed for both strategies and both models

### 9.2 Quality Success Criteria

Hard requirements (must pass):
- [ ] **Hallucination rate**: 0%
- [ ] **No-answer questions**: ≥ 85% correctly identified as unanswerable

Tracked for comparison (no fixed target — see Section 6.4):
- Single-passage correctness
- Multi-passage correctness
- Citation accuracy
- Recall@5, MRR, NDCG@5

### 9.3 Code Quality Success Criteria
- [ ] Code is modular with clear separation of concerns (4 stages)
- [ ] Vector store layer uses abstraction interface for swappability
- [ ] All functions have docstrings
- [ ] Code follows PEP 8 style guidelines
- [ ] All dependencies listed in `pyproject.toml`

### 9.4 Deliverables Success Criteria
- [ ] Working pipeline runs end-to-end without errors
- [ ] Comparison table populated with real numbers from evaluation
- [ ] Write-up completed on larger corpus considerations
- [ ] Evaluation report with results on all 3 question sets

---

## 10. Dependencies & Environment

### 10.1 Python Version
- Python >= 3.12 (as per pyproject.toml)

### 10.2 Dependency Management

All dependencies are managed via **uv** and declared in `pyproject.toml`. Use `uv sync` to install all dependencies.

**Key Dependencies**:

| Package | Purpose |
|---------|---------|
| `sentence-transformers` | Embedding models (MiniLM, MPNet) |
| `chromadb` | Vector database |
| `anthropic` | Claude API client |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `nltk` | Sentence tokenization |
| `scikit-learn` | Metrics (MRR, NDCG calculation) |

### 10.3 Environment Variables

The app reads these directly from the process environment (no `python-dotenv`
or in-app `.env` loading), so supply them via `uv run --env-file .env <file>`:

```bash
ANTHROPIC_AUTH_TOKEN=sk-...        # Required for LLM generation
GENERATION_MODEL=claude-haiku-4-5  # Optional, this is the default
```

---

## 11. Project Timeline

**Timeline**: Single day implementation
**Approach**: Lean, focused execution targeting core deliverables

### Morning (4-5 hours)
- Ingestion & Embedding pipeline
  - Load and chunk documents (both strategies)
  - Generate embeddings with both models
  - Persist chunks and embeddings
- Vector store setup
  - Initialize Chroma collections
  - Load embeddings and metadata
  - Implement basic retrieval

### Afternoon (4-5 hours)
- Generation & Citation
  - Integrate Claude API
  - Implement citation extraction
  - End-to-end pipeline testing with sample queries
- Evaluation
  - Run metrics on all 3 question sets
  - Generate comparison table
  - Document results and larger corpus considerations

### Key Decisions for Speed
- Use pre-trained models (no fine-tuning)
- Simplified metrics implementation
- Focus on correctness over optimization
- Automated evaluation script for all metrics at once
- Single consolidated evaluation report

---

## 12. Appendix: Configuration Examples

### 12.1 Chunking Parameters

```python
# Semantic Splitting Config
SEMANTIC_CHUNK_CONFIG = {
    "target_size": 800,        # tokens
    "min_size": 600,           # tokens
    "max_size": 1000,          # tokens
    "overlap_ratio": 0.1,      # 100 token overlap
}

# Sentence-based Config
SENTENCE_CHUNK_CONFIG = {
    "target_size": 400,        # tokens
    "min_size": 300,           # tokens
    "max_size": 500,           # tokens
    "overlap": 1,              # last sentence of previous chunk
}
```

### 12.2 Embedding Parameters

```python
EMBEDDING_CONFIG = {
    "models": [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
    ],
    "batch_size": 32,
    "normalize": True,  # L2 normalization
    "device": "cuda",   # or "cpu"
}
```

### 12.3 Retrieval Parameters

```python
RETRIEVAL_CONFIG = {
    "top_k": 5,
    "similarity_threshold": 0.15,
    "candidate_pool_size": 20,
    "max_chunks_per_document": 2,
    "include_metadata": True,
}
```

### 12.4 Generation Parameters

```python
GENERATION_CONFIG = {
    "model": "claude-haiku-4-5",
    "temperature": 0.3,
    "max_tokens": 500,
    "timeout": 30,  # seconds
}
```

---

## 13. References

- **Dataset**: Kaggle "Single Topic RAG Evaluation Dataset" (upstream source; see Section 2.1 for what's actually in this repo)
- **Embedding Models**: Sentence-Transformers (https://www.sbert.net/)
- **Vector DB**: Chroma (https://docs.trychroma.com/)
- **LLM**: Anthropic Claude (https://docs.anthropic.com/)
- **Metrics**: Standard IR metrics (Recall, MRR, NDCG)

---

**Document Version**: 2.0
**Last Updated**: 2026-07-06
**Author**: RAG Learning Project
**Status**: Active - Aligned with actual dataset and implementation
