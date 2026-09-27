from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import re


@dataclass(frozen=True)
class Citation:
    """A source citation associated with a retrieved evidence item."""

    citation_id: str
    source: str
    chunk_id: str | None
    score: float | None
    excerpt: str


class CitationManager:
    """Builds stable source citations for the UI and answer metadata."""

    def build_citations(
        self,
        retrieved_results: Iterable[dict[str, Any]],
        max_citations: int = 6,
    ) -> list[Citation]:
        if max_citations <= 0:
            raise ValueError("max_citations must be greater than 0")

        citations: list[Citation] = []

        for index, result in enumerate(
            list(retrieved_results)[:max_citations], start=1
        ):
            if not isinstance(result, dict):
                continue

            metadata = result.get("metadata") or {}
            source = (
                metadata.get("source")
                or metadata.get("filename")
                or metadata.get("title")
                or result.get("source")
                or "Unknown source"
            )

            chunk_id = result.get("chunk_id")
            score = self._score(result)
            text = str(result.get("text", "")).strip()

            citations.append(
                Citation(
                    citation_id=f"S{index}",
                    source=str(source),
                    chunk_id=str(chunk_id) if chunk_id is not None else None,
                    score=score,
                    excerpt=self._excerpt(text),
                )
            )

        return citations

    def attach(
        self,
        answer: str,
        citations: Iterable[Citation],
    ) -> str:
        """
        Append a compact source list.

        The answer itself is not modified with unsupported inline claims.
        This keeps source attribution deterministic and lets the frontend
        render richer source cards separately.
        """
        answer = answer.strip()
        citations = list(citations)

        if not answer:
            raise ValueError("answer cannot be empty")

        if not citations:
            return answer

        lines = ["", "Sources:"]
        for citation in citations:
            lines.append(f"[{citation.citation_id}] {citation.source}")

        return answer + "\n".join(lines)

    @staticmethod
    def _score(result: dict[str, Any]) -> float | None:
        for key in ("reranker_score", "rrf_score", "score", "semantic_score"):
            value = result.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    @staticmethod
    def _excerpt(text: str, limit: int = 240) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "…"
