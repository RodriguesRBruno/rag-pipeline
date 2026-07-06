"""Evaluation metrics for the RAG pipeline.

Computes retrieval quality (Recall@K, MRR, NDCG@K) and answer quality
(Correctness, Citation Accuracy, Hallucination Rate, Unanswerable Detection)
across all 4 (chunking strategy x embedding model) combinations, using the
ground truth in dataset/single_passage_answer_questions.csv,
dataset/multi_passage_answer_questions.csv, and dataset/no_answer_questions.csv.

Ground truth in this corpus is a single `document_index` per question (even
for the "multi-passage" set - see spec/SPEC.md section 2.3), so retrieval
relevance is evaluated against that one document.

Precision@K is intentionally not computed: with exactly one relevant document
per question and K=5, it's structurally capped at 0.2 for any system that
retrieves the correct document at all, so it adds no signal beyond Recall@K
on this dataset (see spec/SPEC.md section 6.1).
"""

from __future__ import annotations

import json
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import label_ranking_average_precision_score, ndcg_score

from src.generation import GroundedResponse
from src.pipeline import RAGPipeline
from src.vectorstore import RetrievedChunk

logger = logging.getLogger(__name__)

STRATEGIES = ["semantic", "sentence"]
MODELS = ["minilm", "mpnet"]

QUESTION_SET_FILES = {
    "single": "single_passage_answer_questions.csv",
    "multi": "multi_passage_answer_questions.csv",
    "no_answer": "no_answer_questions.csv",
}


def _ensure_nltk_data() -> None:
    for resource in ["corpora/stopwords", "tokenizers/punkt_tab"]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[-1], quiet=True)


_ensure_nltk_data()

# NLTK's stopword list, plus generic LLM discourse markers (from our fixed
# system prompt's phrasing, e.g. "according to the provided context") that
# would otherwise look like "novel"/mismatched content in the token-overlap
# heuristics below even though they never carry a factual claim.
_EXTRA_STOPWORDS = {
    "according", "provided", "context", "however", "additionally",
    "furthermore", "moreover", "important", "note", "source", "sources",
    "information", "answer", "specifically", "essentially", "overall",
    "summary", "exception", "detail", "details", "corpus", "based",
    "mentioned", "mentions", "states", "stated", "also", "one",
}
_STOPWORDS = set(stopwords.words("english")) | _EXTRA_STOPWORDS
_STEMMER = SnowballStemmer("english")


def _tokenize(text: str) -> set:
    """Lowercased, stopword-filtered, stemmed word set for overlap heuristics."""
    words = word_tokenize(text.lower())
    return {
        _STEMMER.stem(w)
        for w in words
        if w.isalpha() and len(w) > 2 and w not in _STOPWORDS
    }


@dataclass
class RetrievalMetrics:
    """Retrieval-quality metrics for a single question."""

    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


@dataclass
class GenerationMetrics:
    """Answer-quality metrics for a single question. Fields are None when not
    applicable to the question's type (e.g. correctness for no-answer questions)."""

    correct: Optional[bool] = None
    citation_correct: Optional[bool] = None
    hallucinated: Optional[bool] = None
    correctly_rejected: Optional[bool] = None


@dataclass
class QuestionResult:
    """Full evaluation record for a single question against one combo."""

    question: str
    question_type: str
    ground_truth_document_index: Optional[int]
    ground_truth_answer: Optional[str]
    retrieved_document_indices: List[int]
    retrieval: RetrievalMetrics
    generated_answer: str
    is_answerable: bool
    confidence: float
    cited_document_indices: List[int]
    generation: GenerationMetrics


def _dedupe_preserve_order(values: List[int]) -> List[int]:
    seen: set = set()
    deduped = []
    for v in values:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


def compute_retrieval_metrics(
    retrieved_document_indices: List[int],
    ground_truth_index: int,
    k: int = 5,
) -> RetrievalMetrics:
    """Compute Recall@K, MRR, and NDCG@K for a single question.

    This corpus has exactly one relevant document per question. Recall@K is
    plain set/count arithmetic over the top-K - there's no scikit-learn
    helper for truncated-K retrieval recall over an arbitrary candidate list
    (`top_k_accuracy_score` assumes scores over the full label space, which
    doesn't fit a variable-length candidate list), so it stays a direct
    computation.

    MRR and NDCG@K are delegated to scikit-learn's ranking metrics
    (`label_ranking_average_precision_score`, `ndcg_score`), which is exactly
    equivalent to reciprocal rank / DCG-over-IDCG here since there's only one
    relevant item: precision at that item's rank is 1/rank, and the ideal
    ranking places it first (IDCG@K = 1/log2(2) = 1). A genuine miss (ground
    truth absent from the top-K) is short-circuited to 0 for both rather than
    fed into sklearn - the ideal placement (IDCG) is fixed by the corpus'
    single relevant document regardless of what got retrieved, but sklearn's
    per-call IDCG is computed from what's the given candidate window, so
    padding in the missing document at a floor score would incorrectly award
    partial credit for "ranked last" instead of "not retrieved at all".

    The retriever may return more than one chunk from the same document (see
    `max_chunks_per_document` in src.retrieval); metrics here are document-level,
    so repeated documents are deduped (keeping first/best-ranked occurrence)
    before ranking - otherwise a doc appearing twice could count as two hits.
    """
    unique_docs = _dedupe_preserve_order(retrieved_document_indices)
    top_k = unique_docs[:k]
    hits = [1 if doc == ground_truth_index else 0 for doc in top_k]

    recall = 1.0 if any(hits) else 0.0

    if not any(hits) or len(top_k) < 2:
        # Miss, or too few candidates for sklearn's ndcg_score (requires >1);
        # a single-candidate hit is trivially a perfect ranking (rank 1).
        reciprocal_rank = 1.0 if hits and hits[0] else 0.0
        ndcg = reciprocal_rank
    else:
        y_true = np.array([hits])
        y_score = np.array([[len(top_k) - i for i in range(len(top_k))]])
        reciprocal_rank = float(label_ranking_average_precision_score(y_true, y_score))
        ndcg = float(ndcg_score(y_true, y_score, k=k))

    return RetrievalMetrics(
        recall_at_k=recall,
        reciprocal_rank=reciprocal_rank,
        ndcg_at_k=ndcg,
    )


def compute_generation_metrics(
    response: GroundedResponse,
    question_type: str,
    ground_truth_index: Optional[int] = None,
    ground_truth_answer: Optional[str] = None,
    context_text: str = "",
    correctness_overlap_threshold: float = 0.5,
    hallucination_overlap_threshold: float = 0.5,
) -> GenerationMetrics:
    """Compute answer-quality metrics for a single question's response.

    Heuristics (no ground-truth-answer semantic model, per spec):
      - correct: >= `correctness_overlap_threshold` of ground-truth answer's
        significant words appear in the generated answer.
      - citation_correct: every cited document matches the ground-truth
        document (no citations to an irrelevant document).
      - hallucinated: less than `hallucination_overlap_threshold` of the
        answer's significant words are grounded in the retrieved context.
      - correctly_rejected: (no-answer questions only) response was marked
        unanswerable.
    """
    if question_type == "no_answer":
        return GenerationMetrics(correctly_rejected=not response.is_answerable)

    if not response.is_answerable:
        # An answerable question that the system refused to answer counts as
        # incorrect; citation/hallucination don't apply since no answer was given.
        return GenerationMetrics(correct=False)

    gt_tokens = _tokenize(ground_truth_answer or "")
    answer_tokens = _tokenize(response.answer)
    overlap = (len(gt_tokens & answer_tokens) / len(gt_tokens)) if gt_tokens else 0.0
    correct = overlap >= correctness_overlap_threshold

    cited_indices = {s.document_index for s in response.sources}
    citation_correct = bool(cited_indices) and cited_indices.issubset({ground_truth_index})

    context_tokens = _tokenize(context_text)
    novel = answer_tokens - context_tokens
    novel_ratio = (len(novel) / len(answer_tokens)) if answer_tokens else 0.0
    hallucinated = novel_ratio >= hallucination_overlap_threshold

    return GenerationMetrics(
        correct=correct,
        citation_correct=citation_correct,
        hallucinated=hallucinated,
    )


def _mean(values: List[Optional[bool]]) -> Optional[float]:
    filtered = [float(v) for v in values if v is not None]
    return statistics.mean(filtered) if filtered else None


class Evaluator:
    """Evaluate the RAG pipeline's retrieval and generation quality."""

    def __init__(
        self,
        data_dir: str = "data",
        dataset_dir: str = "dataset",
        top_k: int = 5,
        max_workers: int = 8,
    ):
        """Initialize the evaluator.

        Args:
            data_dir: Directory holding chunks/embeddings/vector-store artifacts.
            dataset_dir: Directory holding the 3 ground-truth question CSVs.
            top_k: K used for retrieval metrics and pipeline retrieval.
            max_workers: Thread-pool size for concurrent (network-bound) generation calls.
        """
        self.data_dir = data_dir
        self.dataset_dir = Path(dataset_dir)
        self.top_k = top_k
        self.max_workers = max_workers
        self.question_sets: Dict[str, pd.DataFrame] = {
            name: pd.read_csv(self.dataset_dir / filename)
            for name, filename in QUESTION_SET_FILES.items()
        }

    def _evaluate_question(
        self,
        pipeline: RAGPipeline,
        question_type: str,
        row: pd.Series,
    ) -> QuestionResult:
        question = row["question"]
        ground_truth_index = int(row["document_index"]) if not pd.isna(row.get("document_index")) else None
        ground_truth_answer = row.get("answer")
        if ground_truth_answer is not None and pd.isna(ground_truth_answer):
            ground_truth_answer = None

        chunks: List[RetrievedChunk] = pipeline.retriever.retrieve(
            question, collection_name=pipeline.collection_name, k=self.top_k
        )
        response = pipeline.generator.generate(question, chunks)

        retrieved_document_indices = [c.document_index for c in chunks]
        retrieval = (
            compute_retrieval_metrics(retrieved_document_indices, ground_truth_index, k=self.top_k)
            if ground_truth_index is not None
            else RetrievalMetrics(0.0, 0.0, 0.0)
        )

        context_text = "\n".join(c.text for c in chunks)
        generation = compute_generation_metrics(
            response,
            question_type=question_type,
            ground_truth_index=ground_truth_index,
            ground_truth_answer=ground_truth_answer,
            context_text=context_text,
        )

        return QuestionResult(
            question=question,
            question_type=question_type,
            ground_truth_document_index=ground_truth_index,
            ground_truth_answer=ground_truth_answer,
            retrieved_document_indices=retrieved_document_indices,
            retrieval=retrieval,
            generated_answer=response.answer,
            is_answerable=response.is_answerable,
            confidence=response.confidence,
            cited_document_indices=[s.document_index for s in response.sources],
            generation=generation,
        )

    def evaluate_question_set(
        self, pipeline: RAGPipeline, question_type: str
    ) -> List[QuestionResult]:
        """Evaluate one question set (single/multi/no_answer) for one pipeline combo."""
        df = self.question_sets[question_type]
        rows = [row for _, row in df.iterrows()]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(
                executor.map(lambda row: self._evaluate_question(pipeline, question_type, row), rows)
            )
        return results

    @staticmethod
    def summarize(results: List[QuestionResult]) -> Dict:
        """Aggregate per-question metrics into mean summary statistics."""
        retrieval_summary = {
            "recall_at_k": _mean([r.retrieval.recall_at_k for r in results]),
            "mrr": _mean([r.retrieval.reciprocal_rank for r in results]),
            "ndcg_at_k": _mean([r.retrieval.ndcg_at_k for r in results]),
        }
        generation_summary = {
            "correctness": _mean([r.generation.correct for r in results]),
            "citation_accuracy": _mean([r.generation.citation_correct for r in results]),
            "hallucination_rate": _mean([r.generation.hallucinated for r in results]),
            "unanswerable_detection_rate": _mean([r.generation.correctly_rejected for r in results]),
        }
        return {
            "n_questions": len(results),
            "retrieval": retrieval_summary,
            "generation": generation_summary,
        }

    def evaluate_combo(self, strategy: str, model: str) -> Dict:
        """Run all 3 question sets against one (strategy, model) pipeline combo."""
        logger.info("Evaluating combo strategy=%s model=%s", strategy, model)
        pipeline = RAGPipeline(strategy=strategy, model=model, top_k=self.top_k, data_dir=self.data_dir)

        by_type: Dict[str, List[QuestionResult]] = {}
        for question_type in QUESTION_SET_FILES:
            by_type[question_type] = self.evaluate_question_set(pipeline, question_type)

        return {
            "combo": f"{strategy}_{model}",
            "strategy": strategy,
            "model": model,
            "by_question_type": {
                question_type: {
                    "summary": self.summarize(results),
                    "questions": [asdict(r) for r in results],
                }
                for question_type, results in by_type.items()
            },
        }

    def evaluate_all(self) -> Dict[str, Dict]:
        """Run evaluation across all 4 (strategy, model) combinations."""
        return {
            f"{strategy}_{model}": self.evaluate_combo(strategy, model)
            for strategy in STRATEGIES
            for model in MODELS
        }


def save_results(results: Dict, path: str = "eval/results/evaluation_results.json") -> None:
    """Save evaluation results to JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Saved evaluation results -> %s", path)


def load_results(path: str = "eval/results/evaluation_results.json") -> Dict:
    """Load previously saved evaluation results from JSON."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(output_path: str = "eval/results/evaluation_results.json") -> Dict:
    """Run the full evaluation sweep and persist results to disk."""
    evaluator = Evaluator()
    results = evaluator.evaluate_all()
    save_results(results, output_path)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_evaluation()
