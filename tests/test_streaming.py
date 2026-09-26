from src.streaming.stream_controller import (
    RetrievalAction,
    StreamController,
)
from src.streaming.transcript_stream import TranscriptChunk


def make_chunk(
    text: str,
    timestamp: float,
    is_final: bool = False,
) -> TranscriptChunk:
    return TranscriptChunk(
        session_id="test-session",
        text=text,
        timestamp=timestamp,
        is_final=is_final,
    )


def test_short_fragment_waits():
    controller = StreamController()

    decision = controller.decide(
        make_chunk("I need to", 0.0),
        "I need to",
    )

    assert decision.action == RetrievalAction.WAIT
    assert decision.query is None


def test_early_retrieval_from_sufficient_context():
    controller = StreamController()

    text = "I need to plan a customer workshop in Pune"

    decision = controller.decide(
        make_chunk(text, 0.8),
        text,
    )

    assert decision.action == RetrievalAction.RETRIEVE
    assert decision.query == text
    assert decision.reason == "sufficient_context_for_early_retrieval"


def test_final_information_request_retrieves():
    controller = StreamController()

    text = "What is the cancellation policy for this venue?"

    decision = controller.decide(
        make_chunk(text, 2.1, is_final=True),
        text,
    )

    assert decision.action == RetrievalAction.RETRIEVE
    assert decision.reason == "final_information_request"


def test_presentation_request_suppresses_retrieval():
    controller = StreamController()

    text = "Please repeat your last answer in bullet points."

    decision = controller.decide(
        make_chunk(text, 2.0, is_final=True),
        text,
    )

    assert decision.action == RetrievalAction.NO_RETRIEVAL
    assert decision.query is None


def test_retrieve_requires_query():
    controller = StreamController()

    text = "What is the refund policy?"

    decision = controller.decide(
        make_chunk(text, 1.0),
        text,
    )

    assert decision.action == RetrievalAction.RETRIEVE
    assert decision.query


def test_decision_timestamp_matches_chunk():
    controller = StreamController()

    chunk = make_chunk(
        "I need to plan a workshop in Pune",
        0.8,
    )

    decision = controller.decide(
        chunk,
        chunk.text,
    )

    assert decision.timestamp == 0.8

def test_duplicate_retrieval_is_suppressed():
    controller = StreamController()

    first_text = "I need to plan a customer workshop in Pune"

    first_decision = controller.decide(
        make_chunk(first_text, 0.8),
        first_text,
    )

    assert first_decision.action == RetrievalAction.RETRIEVE

    second_text = "I need to plan a customer workshop in Pune."

    second_decision = controller.decide(
        make_chunk(second_text, 2.1, is_final=True),
        second_text,
    )

    assert second_decision.action == RetrievalAction.NO_RETRIEVAL
    assert second_decision.reason == "query_already_retrieved"
    assert second_decision.query is None

def test_changed_query_triggers_new_retrieval():
    controller = StreamController()

    first_text = "I need to plan a customer workshop in Pune"

    first_decision = controller.decide(
        make_chunk(first_text, 0.8),
        first_text,
    )

    assert first_decision.action == RetrievalAction.RETRIEVE

    changed_text = (
        "I need to plan a customer workshop in Pune for 50 people"
    )

    changed_decision = controller.decide(
        make_chunk(changed_text, 1.6),
        changed_text,
    )

    assert changed_decision.action == RetrievalAction.RETRIEVE
    assert changed_decision.query == changed_text