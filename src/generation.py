"""Stage 4: Answer generation and citation extraction via the Claude API.

Formats retrieved chunks as grounded context, asks Claude to answer using
only that context, and extracts/validates citations against the retrieved
sources to guard against hallucination.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from anthropic import Anthropic

from src.vectorstore import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("GENERATION_MODEL", "claude-sonnet-5")

UNANSWERABLE_PHRASE = "I don't have this information in my corpus."

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about Enter the Gungeon, "
    "a video game. Use ONLY the provided context to answer the question. "
    "Include citations to source documents by referencing their source URL. "
    f'If the answer is not in the context, respond exactly: "{UNANSWERABLE_PHRASE}"'
)

_URL_RE = re.compile(r"https?://[^\s)>\]\"']+")


@dataclass
class Source:
    """Citation source."""

    url: str
    document_index: int
    chunk_id: str


@dataclass
class GroundedResponse:
    """Generated response with citations."""

    question: str
    answer: str
    sources: List[Source] = field(default_factory=list)
    is_answerable: bool = True
    confidence: float = 0.0


class Generator:
    """Generate answers grounded in retrieved context using Claude."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
        max_tokens: int = 500,
        timeout: int = 30,
        low_confidence_threshold: float = 0.5,
    ):
        """Initialize the Claude client.

        Requires the ANTHROPIC_AUTH_TOKEN environment variable to be set.
        """
        self.client = Anthropic(auth_token=os.getenv("ANTHROPIC_AUTH_TOKEN"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.low_confidence_threshold = low_confidence_threshold

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks as numbered, source-attributed context."""
        blocks = [
            f"Source {i}: {chunk.source_url}\n{chunk.text}"
            for i, chunk in enumerate(chunks, start=1)
        ]
        return "<context>\n" + "\n\n".join(blocks) + "\n</context>"

    def _compute_confidence(self, chunks: List[RetrievedChunk]) -> float:
        """Average similarity of the top-3 retrieved chunks."""
        if not chunks:
            return 0.0
        top = sorted((c.similarity_score for c in chunks), reverse=True)[:3]
        return float(sum(top) / len(top))

    def _extract_citations(
        self,
        response: str,
        chunks: List[RetrievedChunk],
    ) -> List[Source]:
        """Extract citations by matching URLs mentioned in the response
        against retrieved chunks; falls back to citing all retrieved chunks
        if the model didn't include explicit URLs."""
        mentioned_urls = set(_URL_RE.findall(response))

        sources = [
            Source(url=c.source_url, document_index=c.document_index, chunk_id=c.chunk_id)
            for c in chunks
            if c.source_url in mentioned_urls
        ]

        if not sources:
            sources = [
                Source(url=c.source_url, document_index=c.document_index, chunk_id=c.chunk_id)
                for c in chunks
            ]

        # De-duplicate while preserving order.
        seen = set()
        deduped = []
        for source in sources:
            if source.chunk_id not in seen:
                seen.add(source.chunk_id)
                deduped.append(source)
        return deduped

    def _is_answerable(self, response: str, chunks: List[RetrievedChunk]) -> bool:
        """Determine answerability from the model's response and retrieval confidence."""
        if UNANSWERABLE_PHRASE.lower() in response.lower():
            return False
        if not chunks:
            return False
        if self._compute_confidence(chunks) < self.low_confidence_threshold:
            return False
        return True

    def generate(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        ground_truth: Optional[str] = None,
    ) -> GroundedResponse:
        """Generate an answer grounded in retrieved_chunks.

        Args:
            query: User question.
            retrieved_chunks: Top-K retrieved chunks from the retrieval stage.
            ground_truth: Unused for generation; accepted for evaluation callers.

        Returns:
            GroundedResponse with answer, sources, answerability, and confidence.
        """
        confidence = self._compute_confidence(retrieved_chunks)

        if not retrieved_chunks:
            return GroundedResponse(
                question=query,
                answer=UNANSWERABLE_PHRASE,
                sources=[],
                is_answerable=False,
                confidence=0.0,
            )

        context = self._format_context(retrieved_chunks)
        user_message = f"{context}\n\nQuestion: {query}\nAnswer:"

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            timeout=self.timeout,
        )
        answer_text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()

        is_answerable = self._is_answerable(answer_text, retrieved_chunks)
        sources = self._extract_citations(answer_text, retrieved_chunks) if is_answerable else []

        return GroundedResponse(
            question=query,
            answer=answer_text,
            sources=sources,
            is_answerable=is_answerable,
            confidence=confidence,
        )
