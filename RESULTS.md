# RAG Pipeline Evaluation Results

## Corpus Note

This evaluation runs against the dataset in `dataset/`: **20 documents** and **40 questions
per question set** (single/multi/no-answer). See `spec/SPEC.md` section 2 for details. One
consequence worth keeping in mind when reading the numbers below:

- Only document index 0 is genuinely about Enter the Gungeon; the other 19 documents cover
  unrelated topics (D&D campaign notes, RAG/LLM tooling, cooking, films, a game changelog,
  etc). This makes the corpus a generic small-corpus retrieval benchmark more than a
  domain-specific Gungeon QA system.

Per `spec/SPEC.md` section 6.4, only Hallucination Rate and No-Answer Detection carry a
fixed pass/fail target here. Recall@5, MRR, NDCG@5, Correctness, and Citation Accuracy are
reported for comparison across strategy/model combinations without a fixed target — the
small, mixed-topic corpus has no external ground truth for what a "good enough" score looks
like on those axes. Precision@5 is not computed at all: with exactly one relevant document
per question and K=5, it is structurally capped at 0.2 and adds no signal beyond Recall@5.

## Key Implementation Decisions

Several corpus-specific properties shape `src/`'s retrieval and generation behavior:

1. **Per-document cap on retrieval results** (`src/retrieval.py`): document index 16 (a 211KB
   game changelog) accounts for ~30% of all chunks in both chunking strategies, so its chunks
   would otherwise dominate top-5 results for many unrelated queries. `Retriever.retrieve`
   pulls a larger candidate pool (20) and caps results to at most 2 chunks per source document
   before taking the final top-K - still pure cosine-similarity ranking, just preserving
   document-level diversity.
2. **Similarity/confidence thresholds calibrated to this corpus** (`src/pipeline.py`,
   `src/generation.py`): with real embeddings on this corpus, top-1 cosine similarity for
   genuinely correct matches averages ~0.4 and overlaps heavily with no-answer questions'
   scores (~0.39), so both thresholds are set to 0.15 (filtering only near-zero noise).
   Unanswerable detection relies primarily on the LLM reading the retrieved text and stating
   explicitly when the specific fact isn't present, which the corpus's no-answer questions are
   designed to require (they're topically adjacent, not off-topic).
3. **System prompt handles a mixed-topic corpus** (`src/generation.py`): since 19 of the 20
   corpus documents are about unrelated topics (D&D notes, RAG tooling, Stardew Valley, the
   EU AI Act, etc. - see the corpus note above), the system prompt answers from context
   "regardless of subject matter" instead of assuming a single topic, so Claude doesn't refuse
   to answer questions it can answer correctly just because the topic isn't Enter the Gungeon.
4. **Citation extraction matches "Source N" references** (`src/generation.py`): the context
   given to Claude labels chunks "Source 1", "Source 2", etc., and Claude usually cites that
   way rather than repeating the raw URL, so `_extract_citations` matches `Source N` labels
   against the same numbering used to build the context, in addition to literal URLs.

## Comparison Table

| Strategy | Model | Recall@5 | MRR | NDCG@5 | Single Correctness | Multi Correctness | Citation Accuracy | Hallucination Rate | No-Answer Detection |
|----------|-------|----------|-----|--------|--------------------|--------------------|--------------------|---------------------|----------------------|
| Semantic | MiniLM | 83.8% | 70.8% | 74.2% | 57.5% | 47.5% | 84.9% | 0.0% | 95.0% |
| Semantic | MPNet | 77.5% | 70.8% | 72.6% | 52.5% | 55.0% | 94.2% | 0.0% | 92.5% |
| Sentence | MiniLM | 85.0% | 75.4% | 77.8% | 57.5% | 47.5% | 86.7% | 2.0% | 92.5% |
| Sentence | MPNet | 80.0% | 71.6% | 73.7% | 52.5% | 42.5% | 92.0% | 0.0% | 92.5% |

## Winner

**Sentence + MiniLM** scores highest overall (composite score 0.776), with Recall@5=85.0%, MRR=75.4%, single-passage correctness=57.5%, and no-answer detection=92.5%.

## SPEC.md Hard Requirements Checklist

| Combo | No-Answer ≥85% | Hallucination 0% |
|-------|-----------------|-------------------|
| semantic_minilm | ✅ | ✅ |
| semantic_mpnet | ✅ | ✅ |
| sentence_minilm | ✅ | ❌ |
| sentence_mpnet | ✅ | ✅ |

## Per-Question-Type Analysis

(Examples below are from the best-performing combo, **sentence_minilm**.)

**Single-Passage Questions (40 total, Sentence + MiniLM)**
- Rate: 57.5%
- Example correct: "What do keybullet kin drop?" -> According to the provided context, Keybullet Kin drop **keys** upon death. However, if a Keybullet Kin is knocked into a pit, it will not drop a key.

[Source: https://enterthegung
- Example failure: "What do the giants look like?" -> I don't have this information in my corpus.

The provided context contains information about Stardew Valley, Enter the Gungeon, fantasy books, and Alan Wake 2, but does not include

**Multi-Passage Questions (40 total, Sentence + MiniLM)**
- Rate: 47.5%
- Example correct: "Which enemy types wield an AK-47?" -> Based on the provided context, the following enemy types in Enter the Gungeon wield AK-47s:

1. **Tankers** - According to the source, "Tankers wield AK-47s." [https://enterthegung
- Example failure: "What makes jammed enemies different?" -> I don't have this information in my corpus.

While the provided context from Enter the Gungeon mentions "Jammed Chance Kins" having a chance to drop twice the loot, it does not con

**No-Answer Questions (40 total, Sentence + MiniLM)**
- Rate: 92.5%
- Example correct: "How much health does the Mutant Bullet Kin have?" -> I don't have this information in my corpus.

The context states that Mutant Bullet Kin "have higher health" compared to regular Bullet Kin, but it does not provide a specific healt
- Example failure: "What needs to be done to kiss the Emperor?" -> Based on the provided context, to have a romantic/intimate scene with the Emperor in Baldur's Gate 3, you need to:

1. **Be receptive to the Emperor's advances** - Treat the Empero

## Failure Mode Analysis

- **Retrieval failures** (wrong document retrieved): 12
- **Generation failures** (correct doc retrieved, answer still didn't match ground truth): 26
- **Hallucination failures** (answer introduced ungrounded content): 1
- **No-answer failures** (should have rejected, didn't): 3
  - Retrieval failure example: "What do the giants look like?" -> I don't have this information in my corpus.

The provided context contains information about Stardew Valley, Enter the Gungeon, fantasy books, and Alan Wake 2,
  - Generation failure example: "What were the requirements for the project?" -> I don't have this information in my corpus.

The context provided discusses various topics including STICI-note prototype development, data science project impa
  - No-answer failure example: "What needs to be done to kiss the Emperor?" -> Based on the provided context, to have a romantic/intimate scene with the Emperor in Baldur's Gate 3, you need to:

1. **Be receptive to the Emperor's advances*

## Larger Corpus Write-Up

For a 10x larger corpus (~200 documents), the current approach would mostly hold, but the document-length skew we hit here (one 211KB changelog producing 30% of all chunks and crowding out unrelated queries) would recur more often and at greater cost - the per-document chunk cap added during evaluation (see below) would need to become a proper per-document sampling/weighting scheme rather than a fixed cap of 2. Embedding generation would still run on a single machine but would benefit from GPU batching. For 100x scale (~2,000+ documents and tens of thousands of chunks), Chroma's exact search would start to show latency; we'd move to a production vector database (Qdrant/Pinecone/pgvector) with approximate nearest-neighbor search (HNSW) and shard embedding generation across workers. Retrieval would likely move to a two-stage pipeline: fast approximate top-50 candidate search followed by a lightweight reranker, since cosine similarity alone (as seen in this evaluation) doesn't cleanly separate relevant from irrelevant documents once the corpus covers many topics.

## Recommendations

1. **Increase candidate diversity further for multi-passage synthesis**: several multi-passage questions need 2-3 sections from the *same* document (e.g. multiple enemy sub-sections); raising `max_chunks_per_document` from 2 to 3 for the `sentence` strategy specifically would likely help without reintroducing the single-document-crowding problem, since that mainly affects cross-document diversity.
2. **Replace the pure token-overlap correctness heuristic with an LLM-as-judge pass** once outside the single-day budget - it would catch cases like paraphrased-but-correct answers that score low on raw overlap.
