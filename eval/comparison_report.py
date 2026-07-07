"""Generates the comparison report (RESULTS.md) from eval.evaluate's output:
a markdown table across all 4 (strategy, model) combinations, per-question-type
analysis with concrete examples, failure-mode analysis, and the larger-corpus
write-up required by spec/SPEC.md section 8.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from eval.evaluate import load_results

logger = logging.getLogger(__name__)

STRATEGY_LABELS = {"semantic": "Semantic", "sentence": "Sentence"}
MODEL_LABELS = {"minilm": "MiniLM", "mpnet": "MPNet"}

# SPEC.md section 6.4 hard requirements. Every other metric (correctness,
# citation accuracy, recall, MRR, NDCG) is comparison-only, with no fixed target.
TARGETS = {
    "no_answer_detection": 0.85,
    "hallucination_rate_max": 0.0,
}


def _fmt(value: Optional[float], pct: bool = True) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%" if pct else f"{value:.3f}"


def _combo_row_metrics(combo_result: Dict) -> Dict:
    """Flatten one combo's nested results into the metrics shown in the table."""
    by_type = combo_result["by_question_type"]
    single = by_type["single"]["summary"]
    multi = by_type["multi"]["summary"]
    no_answer = by_type["no_answer"]["summary"]

    # Retrieval metrics averaged over single+multi (the only sets with a
    # meaningful "correct document" ground truth - see spec/SPEC.md 2.5).
    n_single = single["n_questions"]
    n_multi = multi["n_questions"]
    n_total = n_single + n_multi

    def weighted(field):
        s = single["retrieval"][field] or 0.0
        m = multi["retrieval"][field] or 0.0
        return (s * n_single + m * n_multi) / n_total if n_total else 0.0

    def weighted_gen(field):
        vals, weights = [], []
        for summary, n in [(single, n_single), (multi, n_multi)]:
            v = summary["generation"][field]
            if v is not None:
                vals.append(v * n)
                weights.append(n)
        return sum(vals) / sum(weights) if weights else None

    return {
        "recall_at_5": weighted("recall_at_k"),
        "mrr": weighted("mrr"),
        "ndcg_at_5": weighted("ndcg_at_k"),
        "single_correctness": single["generation"]["correctness"],
        "multi_correctness": multi["generation"]["correctness"],
        "citation_accuracy": weighted_gen("citation_accuracy"),
        "hallucination_rate": weighted_gen("hallucination_rate"),
        "no_answer_detection": no_answer["generation"]["unanswerable_detection_rate"],
        "recall_single": single["retrieval"]["recall_at_k"],
        "recall_multi": multi["retrieval"]["recall_at_k"],
    }


class ComparisonReporter:
    """Generate the comparison report across all strategy/model combinations."""

    def __init__(self, results: Dict):
        self.results = results
        self.combo_metrics = {combo: _combo_row_metrics(r) for combo, r in results.items()}

    def generate_comparison_table(self) -> str:
        header = (
            "| Strategy | Model | Recall@5 | MRR | NDCG@5 | "
            "Single Correctness | Multi Correctness | Citation Accuracy | "
            "Hallucination Rate | No-Answer Detection |\n"
            "|----------|-------|----------|-----|--------|"
            "--------------------|--------------------|--------------------|"
            "---------------------|----------------------|\n"
        )
        rows = []
        for combo, r in self.results.items():
            m = self.combo_metrics[combo]
            rows.append(
                f"| {STRATEGY_LABELS[r['strategy']]} | {MODEL_LABELS[r['model']]} | "
                f"{_fmt(m['recall_at_5'])} | "
                f"{_fmt(m['mrr'])} | {_fmt(m['ndcg_at_5'])} | "
                f"{_fmt(m['single_correctness'])} | {_fmt(m['multi_correctness'])} | "
                f"{_fmt(m['citation_accuracy'])} | {_fmt(m['hallucination_rate'])} | "
                f"{_fmt(m['no_answer_detection'])} |"
            )
        return header + "\n".join(rows)

    def identify_winner(self) -> str:
        def score(combo):
            m = self.combo_metrics[combo]
            parts = [
                m["recall_at_5"], m["mrr"], m["ndcg_at_5"],
                m["single_correctness"] or 0, m["multi_correctness"] or 0,
                m["citation_accuracy"] or 0, 1 - (m["hallucination_rate"] or 0),
                m["no_answer_detection"] or 0,
            ]
            return sum(parts) / len(parts)

        best = max(self.combo_metrics, key=score)
        r = self.results[best]
        m = self.combo_metrics[best]
        return (
            f"**{STRATEGY_LABELS[r['strategy']]} + {MODEL_LABELS[r['model']]}** scores highest "
            f"overall (composite score {score(best):.3f}), with Recall@5={_fmt(m['recall_at_5'])}, "
            f"MRR={_fmt(m['mrr'])}, single-passage correctness={_fmt(m['single_correctness'])}, "
            f"and no-answer detection={_fmt(m['no_answer_detection'])}."
        )

    def _example_for(self, combo: str, question_type: str, correct: bool) -> Optional[Dict]:
        questions = self.results[combo]["by_question_type"][question_type]["questions"]
        for q in questions:
            gen = q["generation"]
            if question_type == "no_answer":
                is_match = gen["correctly_rejected"] is (True if correct else False)
            else:
                is_match = gen["correct"] is (True if correct else False)
            if is_match:
                return q
        return None

    def analyze_per_question_type(self, primary_combo: str) -> str:
        sections = []
        r = self.results[primary_combo]
        label = f"{STRATEGY_LABELS[r['strategy']]} + {MODEL_LABELS[r['model']]}"

        for question_type, title, n in [
            ("single", "Single-Passage Questions", r["by_question_type"]["single"]["summary"]["n_questions"]),
            ("multi", "Multi-Passage Questions", r["by_question_type"]["multi"]["summary"]["n_questions"]),
            ("no_answer", "No-Answer Questions", r["by_question_type"]["no_answer"]["summary"]["n_questions"]),
        ]:
            summary = r["by_question_type"][question_type]["summary"]
            metric_key = "unanswerable_detection_rate" if question_type == "no_answer" else "correctness"
            rate = summary["generation"][metric_key]

            good = self._example_for(primary_combo, question_type, correct=True)
            bad = self._example_for(primary_combo, question_type, correct=False)

            lines = [f"**{title} ({n} total, {label})**", f"- Rate: {_fmt(rate)}"]
            if good:
                lines.append(f'- Example correct: "{good["question"]}" -> {good["generated_answer"][:180].strip()}')
            if bad:
                lines.append(f'- Example failure: "{bad["question"]}" -> {bad["generated_answer"][:180].strip()}')
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def identify_failure_modes(self, primary_combo: str) -> str:
        r = self.results[primary_combo]
        retrieval_failures, generation_failures, hallucination_failures, no_answer_failures = [], [], [], []

        for question_type in ("single", "multi"):
            for q in r["by_question_type"][question_type]["questions"]:
                gen = q["generation"]
                if gen["correct"] is False and q["ground_truth_document_index"] not in q["retrieved_document_indices"]:
                    retrieval_failures.append(q)
                elif gen["correct"] is False:
                    generation_failures.append(q)
                if gen["hallucinated"]:
                    hallucination_failures.append(q)

        for q in r["by_question_type"]["no_answer"]["questions"]:
            if not q["generation"]["correctly_rejected"]:
                no_answer_failures.append(q)

        lines = [
            f"- **Retrieval failures** (wrong document retrieved): {len(retrieval_failures)}",
            f"- **Generation failures** (correct doc retrieved, answer still didn't match ground truth): {len(generation_failures)}",
            f"- **Hallucination failures** (answer introduced ungrounded content): {len(hallucination_failures)}",
            f"- **No-answer failures** (should have rejected, didn't): {len(no_answer_failures)}",
        ]
        for label, bucket in [
            ("Retrieval failure example", retrieval_failures),
            ("Generation failure example", generation_failures),
            ("No-answer failure example", no_answer_failures),
        ]:
            if bucket:
                q = bucket[0]
                lines.append(f'  - {label}: "{q["question"]}" -> {q["generated_answer"][:160].strip()}')
        return "\n".join(lines)

    @staticmethod
    def write_larger_corpus_section() -> str:
        return (
            "For a 10x larger corpus (~200 documents), the current approach would mostly hold, but "
            "the document-length skew we hit here (one 211KB changelog producing 30% of all chunks and "
            "crowding out unrelated queries) would recur more often and at greater cost - the per-document "
            "chunk cap added during evaluation (see below) would need to become a proper per-document "
            "sampling/weighting scheme rather than a fixed cap of 2. Embedding generation would still run "
            "on a single machine but would benefit from GPU batching. For 100x scale (~2,000+ documents and "
            "tens of thousands of chunks), Chroma's exact search would start to show latency; we'd move to a "
            "production vector database (Qdrant/Pinecone/pgvector) with approximate nearest-neighbor search "
            "(HNSW) and shard embedding generation across workers. Retrieval would likely move to a two-stage "
            "pipeline: fast approximate top-50 candidate search followed by a lightweight reranker, since "
            "cosine similarity alone (as seen in this evaluation) doesn't cleanly separate relevant from "
            "irrelevant documents once the corpus covers many topics."
        )

    def recommendations(self, primary_combo: str) -> str:
        return (
            "1. **Increase candidate diversity further for multi-passage synthesis**: several multi-passage "
            "questions need 2-3 sections from the *same* document (e.g. multiple enemy sub-sections); "
            "raising `max_chunks_per_document` from 2 to 3 for the `sentence` strategy specifically would "
            "likely help without reintroducing the single-document-crowding problem, since that mainly "
            "affects cross-document diversity.\n"
            "2. **Replace the pure token-overlap correctness heuristic with an LLM-as-judge pass** once "
            "outside the single-day budget - it would catch cases like paraphrased-but-correct answers that "
            "score low on raw overlap.\n"
            "3. **Re-balance the corpus** if extending this project: the dataset's mix of unrelated "
            "topics (D&D notes, RAG tooling, cooking, films) alongside a single real Gungeon document makes "
            "this a generic small-corpus retrieval benchmark rather than a domain-specific Gungeon QA system; "
            "sourcing more Gungeon-specific documents would make the single/multi-passage results more "
            "representative of a Gungeon-focused QA system."
        )


def generate_full_report(results: Dict, output_path: str = "RESULTS.md") -> None:
    """Generate the complete markdown report and write it to output_path."""
    reporter = ComparisonReporter(results)

    def score(combo):
        m = reporter.combo_metrics[combo]
        parts = [
            m["recall_at_5"], m["mrr"], m["ndcg_at_5"],
            m["single_correctness"] or 0, m["multi_correctness"] or 0,
            m["citation_accuracy"] or 0, 1 - (m["hallucination_rate"] or 0),
            m["no_answer_detection"] or 0,
        ]
        return sum(parts) / len(parts)

    primary_combo = max(reporter.combo_metrics, key=score)

    checklist_rows = []
    for combo, m in reporter.combo_metrics.items():
        checklist_rows.append(
            f"| {combo} | "
            f"{'✅' if (m['no_answer_detection'] or 0) >= TARGETS['no_answer_detection'] else '❌'} | "
            f"{'✅' if (m['hallucination_rate'] if m['hallucination_rate'] is not None else 1) <= TARGETS['hallucination_rate_max'] else '❌'} |"
        )

    report = f"""# RAG Pipeline Evaluation Results

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

{reporter.generate_comparison_table()}

## Winner

{reporter.identify_winner()}

## SPEC.md Hard Requirements Checklist

| Combo | No-Answer ≥85% | Hallucination 0% |
|-------|-----------------|-------------------|
{chr(10).join(checklist_rows)}

## Per-Question-Type Analysis

(Examples below are from the best-performing combo, **{primary_combo}**.)

{reporter.analyze_per_question_type(primary_combo)}

## Failure Mode Analysis

{reporter.identify_failure_modes(primary_combo)}

## Larger Corpus Write-Up

{reporter.write_larger_corpus_section()}

## Recommendations

{reporter.recommendations(primary_combo)}
"""

    Path(output_path).write_text(report, encoding="utf-8")
    logger.info("Wrote comparison report -> %s", output_path)


def run_comparison_report(
    results_path: str = "eval/results/evaluation_results.json",
    output_path: str = "RESULTS.md",
) -> None:
    results = load_results(results_path)
    generate_full_report(results, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_comparison_report()
