
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.evaluation.evaluator import Evaluator
from src.generation.answer_generator import AnswerGenerator
from src.generation.citation_manager import CitationManager
from src.generation.grounding_checker import GroundingChecker
from src.orchestration.pipeline import StreamingRAGPipeline
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.multi_query_retriever import MultiQueryRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore
from src.streaming.transcript_stream import TranscriptChunk


class ChatRequest(BaseModel):
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    text: str
    is_final: bool = True
    sequence_id: int | None = None


class SourceResponse(BaseModel):
    citation_id: str
    source: str
    chunk_id: str | None
    score: float | None
    excerpt: str


class ChatResponse(BaseModel):
    session_id: str
    action: str
    reason: str
    query: str | None
    answer: str | None
    citations: list[SourceResponse]
    groundedness: float | None
    grounded: bool | None
    retrieval_count: int
    latency_ms: float


class RAGApplication:
    """Application service joining the streaming pipeline to generation and evaluation."""

    def __init__(
        self,
        pipeline: StreamingRAGPipeline | None = None,
        generator: AnswerGenerator | None = None,
    ) -> None:
        self.generator = generator or AnswerGenerator()
        self.citations = CitationManager()
        self.grounding = GroundingChecker()
        self.evaluator = Evaluator(self.grounding)

        self.pipeline = pipeline or StreamingRAGPipeline(
            retriever_factory=self._build_retriever
        )

    def _build_retriever(self) -> MultiQueryRetriever:
        embedding_model = EmbeddingModel()

        vector_store = VectorStore(
            index_dir="data/processed/vector_store",
            dimension=embedding_model.get_dimension(),
        )

        retriever = Retriever(
            embedding_model=embedding_model,
            vector_store=vector_store,
            default_top_k=5,
        )

        hybrid_search = HybridSearch(
            retriever=retriever,
            vector_store=vector_store,
            semantic_top_k=5,
            keyword_top_k=5,
        )

        fusion = ReciprocalRankFusion(k=60)

        reranker = Reranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            batch_size=16,
        )

        return MultiQueryRetriever(
            hybrid_search=hybrid_search,
            fusion=fusion,
            reranker=reranker,
            default_subquery_top_k=5,
            default_final_top_k=5,
            semantic_weight=1.0,
            keyword_weight=1.0,
        )

    def process(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()

        chunk = TranscriptChunk(
            session_id=request.session_id,
            text=request.text,
            timestamp=time.time(),
            is_final=request.is_final,
            sequence_id=request.sequence_id,
        )

        try:
            result = self.pipeline.process_chunk(chunk)
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

        action = result.retrieval_decision.action.value
        retrieved = list(result.retrieved_results)

        if not retrieved:
            elapsed = (
                time.perf_counter() - started
            ) * 1000.0

            return ChatResponse(
                session_id=request.session_id,
                action=action,
                reason=result.retrieval_decision.reason,
                query=result.retrieval_decision.query,
                answer=None,
                citations=[],
                groundedness=None,
                grounded=None,
                retrieval_count=0,
                latency_ms=round(elapsed, 2),
            )

        query = (
            result.retrieval_decision.query
            or request.text
        )

        session_context = self.pipeline.session_manager.get_context(
            result.session.session_id
        )

        generated = self.generator.generate(
            query=query,
            retrieved_results=retrieved,
            session_context=session_context,
        )

        citations = self.citations.build_citations(
            answer=generated.answer,
            retrieved_results=retrieved,
        )
        grounding = self.grounding.check(
            generated.answer,
            retrieved,
        )

        result.session.current_answer = generated.answer

        elapsed = (
            time.perf_counter() - started
        ) * 1000.0

        response = ChatResponse(
            session_id=request.session_id,
            action=action,
            reason=result.retrieval_decision.reason,
            query=query,
            answer=generated.answer,
            citations=[
                SourceResponse(
                    citation_id=item.citation_id,
                    source=item.source,
                    chunk_id=item.chunk_id,
                    score=item.score,
                    excerpt=item.excerpt,
                )
                for item in citations
            ],
            groundedness=grounding.score,
            grounded=grounding.grounded,
            retrieval_count=len(retrieved),
            latency_ms=round(elapsed, 2),
        )

        self.evaluator.evaluate_turn(
            query=query,
            answer=generated.answer,
            retrieved_results=retrieved,
            citations=citations,
            retrieval_required=True,
            started_at=started,
            first_token_at=(
                started
                + generated.latency_ms / 1000.0
            ),
            completed_at=time.perf_counter(),
        )

        return response


app = FastAPI(
    title="PRISM Streaming Live RAG",
    version="1.0.0",
    description="Samsung PRISM Theme 4 demo API.",
)

service = RAGApplication()


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "prism-live-rag",
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        return service.process(request)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return service.evaluator.summary()