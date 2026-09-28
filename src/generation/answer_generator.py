from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class GenerationResult:
    answer: str
    latency_ms: float


class AnswerGenerator:
    """
    Generation interface for the RAG pipeline.

    The current implementation is a deterministic placeholder that returns
    retrieved evidence. The interface is intentionally kept stable so an LLM
    can be connected later without changing the orchestration/API contract.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.model = kwargs.get("model")
        self.client = kwargs.get("client")

    def generate(
        self,
        query: str,
        retrieved_results: list[dict[str, Any]] | None = None,
        session_context: str = "",
    ) -> GenerationResult:
        start = time.perf_counter()
        results = retrieved_results or []

        if not results:
            answer = "I don't have enough retrieved evidence to answer that reliably."
        else:
            answer = str(results[0].get("text", "")).strip()

        return GenerationResult(
            answer=answer,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
