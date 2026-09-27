from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any

@dataclass
class GenerationResult:
    answer: str
    latency_ms: float

class AnswerGenerator:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, query: str, retrieved_results: list[dict[str, Any]] | None = None, session_context: str = "") -> GenerationResult:
        start = time.perf_counter()
        results = retrieved_results or []

        if not results:
            answer = "I don't have enough retrieved evidence to answer that reliably."
        else:
            answer = results[0].get("text", "").strip()

        return GenerationResult(
            answer=answer,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
