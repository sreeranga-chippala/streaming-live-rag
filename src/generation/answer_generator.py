from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


@dataclass
class GenerationResult:
    answer: str
    latency_ms: float


class AnswerGenerator:
    """
    Corpus-grounded answer synthesis.

    Responsibilities:
        1. Receive the user's query.
        2. Receive retrieved corpus evidence.
        3. Synthesize a concise answer using the LLM.
        4. Prevent unsupported facts from being introduced.
        5. Use session context when refining an existing answer.

    Retrieval remains outside this class.
    Citation construction and grounding verification are also
    handled by their dedicated components.
    """

    def __init__(
        self,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or os.getenv(
            "GEMINI_MODEL",
            "gemini-3.8-flash",
        )

        self.client = client

        if self.client is None:
            api_key = os.getenv("GEMINI_API_KEY")

            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not configured."
                )

            self.client = genai.Client(
                api_key=api_key
            )

    def generate(
        self,
        query: str,
        retrieved_results: list[dict[str, Any]] | None = None,
        session_context: str = "",
    ) -> GenerationResult:
        start = time.perf_counter()

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string"
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty"
            )

        results = retrieved_results or []

        if not results:
            return GenerationResult(
                answer=(
                    "I don't have enough retrieved evidence "
                    "to answer that reliably."
                ),
                latency_ms=(
                    time.perf_counter() - start
                ) * 1000,
            )

        evidence_blocks = []

        for index, result in enumerate(results, start=1):
            if not isinstance(result, dict):
                continue

            text = str(
                result.get("text", "")
            ).strip()

            if not text:
                continue

            source = str(
                result.get("source", "unknown")
            )

            chunk_id = str(
                result.get("chunk_id", "unknown")
            )

            evidence_blocks.append(
                f"[Evidence {index}]\n"
                f"Source: {source}\n"
                f"Chunk: {chunk_id}\n"
                f"Content:\n{text}"
            )

        if not evidence_blocks:
            return GenerationResult(
                answer=(
                    "I don't have enough retrieved evidence "
                    "to answer that reliably."
                ),
                latency_ms=(
                    time.perf_counter() - start
                ) * 1000,
            )

        evidence = "\n\n".join(
            evidence_blocks
        )

        context_section = ""

        if session_context.strip():
            context_section = f"""
Previous session evidence:
{session_context.strip()}
"""

        prompt = f"""
You are the answer-generation component of a Streaming Live RAG system.

Answer the user's question using ONLY the retrieved corpus evidence
provided below.

STRICT RULES:

1. Do not use outside knowledge.
2. Do not invent facts, policies, numbers, dates, names, or sources.
3. Do not mention information that cannot be supported by the evidence.
4. If the evidence does not contain enough information to answer part
   of the question, explicitly say that the available evidence is
   insufficient for that part.
5. Synthesize the evidence instead of copying large source passages.
6. Answer every supported sub-question in the user's request.
7. Prefer concise bullet points when the question contains multiple
   sub-questions.
8. Preserve important numbers, conditions, exceptions, and dates exactly
   when they are supported by the evidence.
9. Do not fabricate citation IDs. Citation mapping is handled separately.
10. Do not refer to the evidence as "the retrieved documents" in the
    final answer unless necessary.
11. If previous session evidence is supplied, use it only when relevant
    to refining the current answer.
12. Do not restart unrelated parts of the answer merely because a new
    constraint was introduced.

User question:
{query}

{context_section}

Current retrieved corpus evidence:
{evidence}

Produce only the final answer.
"""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are a strict corpus-grounded RAG "
                        "answer generator."
                    ),
                ),
            )

            answer = (
                getattr(response, "text", None)
                or ""
            ).strip()

        except Exception as exc:
            raise RuntimeError(
                f"Answer generation failed: {exc}"
            ) from exc

        if not answer:
            answer = (
                "I don't have enough retrieved evidence "
                "to answer that reliably."
            )

        return GenerationResult(
            answer=answer,
            latency_ms=(
                time.perf_counter() - start
            ) * 1000,
        )
