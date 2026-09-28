from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from src.generation.grounding_checker import GroundingChecker


@dataclass
class EvaluationRecord:
    query: str
    retrieval_required: bool
    retrieval_count: int
    citation_count: int
    groundedness: float
    grounded: bool
    retrieval_recall: float | None
    time_to_first_token_ms: float | None
    total_latency_ms: float | None


class Evaluator:
    """
    Evaluation and telemetry layer for the Theme 4 demo.

    Current metrics:
      - retrieval recall when gold chunk IDs are supplied
      - groundedness
      - citation coverage/count
      - total request latency

    Token-level TTFT is intentionally unavailable until the API exposes
    real token streaming. It must not be inferred from request timestamps.
    """

    def __init__(self, grounding_checker: GroundingChecker | None = None):
        self.grounding_checker = grounding_checker or GroundingChecker()
        self.records: list[EvaluationRecord] = []

    def evaluate_turn(
        self,
        *,
        query: str,
        answer: str,
        retrieved_results: Iterable[dict[str, Any]],
        citations: Iterable[Any] | None = None,
        gold_chunk_ids: Iterable[str] | None = None,
        retrieval_required: bool = True,
        started_at: float | None = None,
        first_token_at: float | None = None,
        completed_at: float | None = None,
    ) -> EvaluationRecord:
        retrieved = list(retrieved_results)
        citation_list = list(citations or [])

        grounding = self.grounding_checker.check(answer, retrieved)
        recall = self._retrieval_recall(retrieved, gold_chunk_ids)

        # Real token-level TTFT is not available while /chat returns
        # one complete JSON response. Do not report a synthetic TTFT.
        ttft = None

        total = None
        if started_at is not None and completed_at is not None:
            total = max(0.0, (completed_at - started_at) * 1000.0)

        record = EvaluationRecord(
            query=query,
            retrieval_required=retrieval_required,
            retrieval_count=len(retrieved),
            citation_count=len(citation_list),
            groundedness=grounding.score,
            grounded=grounding.grounded,
            retrieval_recall=recall,
            time_to_first_token_ms=ttft,
            total_latency_ms=total,
        )
        self.records.append(record)
        return record

    def summary(self) -> dict[str, Any]:
        if not self.records:
            return {
                "turns": 0,
                "avg_groundedness": 0.0,
                "grounded_turn_rate": 0.0,
                "avg_retrieval_recall": None,
                "avg_ttft_ms": None,
                "avg_total_latency_ms": None,
                "avg_retrieval_count": 0.0,
                "avg_citation_count": 0.0,
            }

        def avg(values: list[float]) -> float | None:
            return round(sum(values) / len(values), 3) if values else None

        recalls = [
            r.retrieval_recall
            for r in self.records
            if r.retrieval_recall is not None
        ]
        ttfts = [
            r.time_to_first_token_ms
            for r in self.records
            if r.time_to_first_token_ms is not None
        ]
        totals = [
            r.total_latency_ms
            for r in self.records
            if r.total_latency_ms is not None
        ]

        return {
            "turns": len(self.records),
            "avg_groundedness": round(
                sum(r.groundedness for r in self.records) / len(self.records), 3
            ),
            "grounded_turn_rate": round(
                sum(r.grounded for r in self.records) / len(self.records), 3
            ),
            "avg_retrieval_recall": avg(recalls),
            "avg_ttft_ms": avg(ttfts),
            "avg_total_latency_ms": avg(totals),
            "avg_retrieval_count": round(
                sum(r.retrieval_count for r in self.records) / len(self.records), 3
            ),
            "avg_citation_count": round(
                sum(r.citation_count for r in self.records) / len(self.records), 3
            ),
        }

    @staticmethod
    def _retrieval_recall(
        retrieved: list[dict[str, Any]],
        gold_chunk_ids: Iterable[str] | None,
    ) -> float | None:
        if gold_chunk_ids is None:
            return None

        gold = {str(item) for item in gold_chunk_ids}
        if not gold:
            return None

        found = {
            str(item.get("chunk_id"))
            for item in retrieved
            if item.get("chunk_id") is not None
        }
        return round(len(found & gold) / len(gold), 4)

    def export(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.records]
