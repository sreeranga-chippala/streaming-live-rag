
from dataclasses import dataclass
from typing import Any, Callable

from src.intelligence.intent_detector import (
    IntentDetector,
    IntentResult,
)
from src.intelligence.query_decomposer import (
    DecompositionResult,
    QueryDecomposer,
)
from src.intelligence.query_refiner import (
    QueryRefiner,
    RefinedQuery,
)
from src.intelligence.session_manager import (
    SessionManager,
    SessionState,
)
from src.retrieval.multi_query_retriever import MultiQueryRetriever
from src.streaming.chunk_handler import (
    ChunkHandler,
    ChunkProcessingResult,
)
from src.streaming.stream_controller import (
    RetrievalAction,
    RetrievalDecision,
    StreamController,
)
from src.streaming.transcript_stream import TranscriptChunk


@dataclass(frozen=True)
class PipelineResult:
    """
    Result produced after processing one transcript chunk.
    """

    session_id: str
    chunk_result: ChunkProcessingResult
    retrieval_decision: RetrievalDecision
    intent: IntentResult | None
    decomposition: DecompositionResult | None
    refined_queries: tuple[RefinedQuery, ...]
    retrieved_results: tuple[dict[str, Any], ...]
    session: SessionState


class StreamingRAGPipeline:
    """
    Orchestrates the streaming, intelligence, and optional retrieval stages.

    Retrieval is initialized lazily only when the stream controller
    decides that retrieval is required.
    """

    def __init__(
        self,
        session_manager: SessionManager | None = None,
        stream_controller: StreamController | None = None,
        intent_detector: IntentDetector | None = None,
        query_decomposer: QueryDecomposer | None = None,
        query_refiner: QueryRefiner | None = None,
        multi_query_retriever: MultiQueryRetriever | None = None,
        retriever_factory: Callable[[], MultiQueryRetriever] | None = None,
    ) -> None:
        self.session_manager = (
            session_manager or SessionManager()
        )
        self.stream_controller = (
            stream_controller or StreamController()
        )
        self.intent_detector = (
            intent_detector or IntentDetector()
        )
        self.query_decomposer = (
            query_decomposer or QueryDecomposer()
        )
        self.query_refiner = (
            query_refiner or QueryRefiner()
        )
        self.multi_query_retriever = multi_query_retriever
        self.retriever_factory = retriever_factory

        self._chunk_handlers: dict[str, ChunkHandler] = {}

    def process_chunk(
        self,
        chunk: TranscriptChunk,
    ) -> PipelineResult:
        """
        Process one incoming transcript chunk through the
        streaming, intelligence, and optional retrieval stages.
        """

        session = self.session_manager.get_or_create(
            chunk.session_id
        )

        handler = self._get_chunk_handler(chunk.session_id)

        chunk_result = handler.process(chunk)

        self.session_manager.add_transcript(
            session_id=chunk.session_id,
            text=chunk.text,
            timestamp=chunk.timestamp,
        )

        retrieval_decision = self.stream_controller.decide(
            chunk=chunk,
            accumulated_text=chunk_result.accumulated_text,
        )

        if retrieval_decision.action != RetrievalAction.RETRIEVE:
            return PipelineResult(
                session_id=chunk.session_id,
                chunk_result=chunk_result,
                retrieval_decision=retrieval_decision,
                intent=None,
                decomposition=None,
                refined_queries=(),
                retrieved_results=(),
                session=session,
            )

        if self.multi_query_retriever is None:
            if self.retriever_factory is None:
                raise RuntimeError("Retriever is not configured")

            self.multi_query_retriever = self.retriever_factory()

        query = retrieval_decision.query

        if query is None:
            raise RuntimeError(
                "RETRIEVE decision did not contain a query"
            )

        intent = self.intent_detector.detect(query)

        self.session_manager.add_query(
            session_id=chunk.session_id,
            query=query,
            timestamp=chunk.timestamp,
        )

        self.session_manager.add_intent(
            session_id=chunk.session_id,
            intent=intent.intent.value,
            timestamp=chunk.timestamp,
        )

        decomposition = self.query_decomposer.decompose(query)

        refined_queries = tuple(
            self.query_refiner.refine(
                sub_query.query,
                session_context=self.session_manager.get_context(
                    chunk.session_id
                ),
            )
            for sub_query in decomposition.sub_queries
        )

        retrieved_results: tuple[dict[str, Any], ...] = ()

        results = self.multi_query_retriever.retrieve(
            [
                refined_query.refined_query
                for refined_query in refined_queries
            ]
        )

        retrieved_results = tuple(results)

        context_parts = [
            result["text"]
            for result in results
            if isinstance(result.get("text"), str)
            and result["text"].strip()
        ]

        if context_parts:
            self.session_manager.add_retrieved_context(
                session_id=chunk.session_id,
                context="\n\n".join(context_parts),
                timestamp=chunk.timestamp,
            )

        return PipelineResult(
            session_id=chunk.session_id,
            chunk_result=chunk_result,
            retrieval_decision=retrieval_decision,
            intent=intent,
            decomposition=decomposition,
            refined_queries=refined_queries,
            retrieved_results=retrieved_results,
            session=session,
        )

    def _get_chunk_handler(
        self,
        session_id: str,
    ) -> ChunkHandler:
        """
        Return the persistent chunk handler for a session.
        """

        if session_id not in self._chunk_handlers:
            self._chunk_handlers[session_id] = ChunkHandler(
                session_id
            )

        return self._chunk_handlers[session_id]

    def close_session(
        self,
        session_id: str,
    ) -> SessionState:
        """
        Close the session and release its streaming state.
        """

        self._chunk_handlers.pop(session_id, None)
        self.stream_controller.clear_session(session_id)

        return self.session_manager.close_session(
            session_id
        )
