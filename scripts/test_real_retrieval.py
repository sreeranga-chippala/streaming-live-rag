from pathlib import Path
import sys

from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever


# ============================================================
# CONFIGURATION
# ============================================================

VECTOR_STORE_DIR = Path(
    "data/processed/vector_store"
)

MODEL_NAME = "all-MiniLM-L6-v2"

# Smoke-test threshold for this fixed embedding setup.
RELEVANCE_THRESHOLD = 0.20


# ============================================================
# TEST CASES
# ============================================================

POSITIVE_TESTS = [
    {
        "query": "What does the streaming pipeline contain?",
        "expected_terms": [
            "transcript streaming",
            "chunk handling",
            "retrieval decision logic",
            "intent detection",
            "query decomposition",
            "query refinement",
            "session management",
            "orchestration",
        ],
    },
    {
        "query": "What retrieval methods does this project use?",
        "expected_terms": [
            "semantic search",
            "keyword search",
            "reciprocal rank fusion",
            "reranking",
        ],
    },
    {
        "query": "What metrics does the evaluation layer measure?",
        "expected_terms": [
            "retrieval recall",
            "groundedness",
            "citation support",
            "time to first token",
            "latency",
            "retrieval statistics",
        ],
    },
]


NEGATIVE_TESTS = [
    "How do I enable Bluetooth on a Samsung phone?",
    "What is the weather in Bengaluru today?",
]


# ============================================================
# HELPERS
# ============================================================

def fail(message: str) -> None:
    print(f"[FAIL] {message}")


def passed(message: str) -> None:
    print(f"[PASS] {message}")


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    print("=" * 70)
    print("REAL DOCUMENT RETRIEVAL SMOKE TEST")
    print("=" * 70)

    failures = 0

    # --------------------------------------------------------
    # LOAD EMBEDDING MODEL
    # --------------------------------------------------------

    embedding_model = EmbeddingModel(
        model_name=MODEL_NAME,
        batch_size=32,
    )

    # --------------------------------------------------------
    # LOAD VECTOR STORE
    # --------------------------------------------------------

    vector_store = VectorStore(
        index_dir=VECTOR_STORE_DIR,
        dimension=embedding_model.get_dimension(),
    )

    faiss_count = vector_store.count()
    metadata_count = vector_store.metadata_count()
    keyword_count = vector_store.keyword_count()

    print()
    print(f"FAISS vectors : {faiss_count}")
    print(f"SQLite records: {metadata_count}")
    print(f"FTS5 rows     : {keyword_count}")

    # --------------------------------------------------------
    # INDEX HEALTH CHECKS
    # --------------------------------------------------------

    if faiss_count <= 0:
        fail("FAISS index contains no vectors.")
        failures += 1
    else:
        passed(f"FAISS contains {faiss_count} vector(s).")

    if metadata_count <= 0:
        fail("SQLite metadata store contains no records.")
        failures += 1
    else:
        passed(
            f"SQLite metadata contains "
            f"{metadata_count} record(s)."
        )

    if keyword_count <= 0:
        fail("FTS5 keyword index contains no rows.")
        failures += 1
    else:
        passed(
            f"FTS5 keyword index contains "
            f"{keyword_count} row(s)."
        )

    if faiss_count != metadata_count:
        fail(
            "FAISS and SQLite counts do not match: "
            f"{faiss_count} != {metadata_count}"
        )
        failures += 1
    else:
        passed("FAISS and SQLite counts are synchronized.")

    # --------------------------------------------------------
    # CREATE RETRIEVER
    # --------------------------------------------------------

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        default_top_k=5,
    )

    # --------------------------------------------------------
    # POSITIVE RETRIEVAL TESTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("POSITIVE RETRIEVAL TESTS")
    print("=" * 70)

    for test in POSITIVE_TESTS:

        query = test["query"]
        expected_terms = test["expected_terms"]

        print()
        print(f"QUERY: {query}")

        results = retriever.retrieve(
            query=query,
            top_k=5,
        )

        if not results:
            fail("No retrieval results returned.")
            failures += 1
            continue

        top_result = results[0]

        score = float(top_result["score"])
        text = top_result["text"].lower()

        print(f"Top score: {score:.4f}")
        print(f"Chunk ID  : {top_result['chunk_id']}")
        print(
            f"Source   : "
            f"{top_result['metadata'].get('source')}"
        )

        if score < RELEVANCE_THRESHOLD:
            fail(
                f"Top result score {score:.4f} is below "
                f"the relevance threshold "
                f"{RELEVANCE_THRESHOLD:.2f}."
            )
            failures += 1
            continue

        passed(
            f"Relevant result returned with score "
            f"{score:.4f}."
        )

        matched_terms = [
            term
            for term in expected_terms
            if term.lower() in text
        ]

        if not matched_terms:
            fail(
                "Retrieved text did not contain any "
                "expected evidence terms."
            )
            failures += 1
        else:
            passed(
                "Expected evidence found: "
                + ", ".join(matched_terms)
            )

    # --------------------------------------------------------
    # UNRELATED QUERY REJECTION TESTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("UNRELATED-QUERY REJECTION TESTS")
    print("=" * 70)

    for query in NEGATIVE_TESTS:

        print()
        print(f"QUERY: {query}")

        results = retriever.retrieve(
            query=query,
            top_k=5,
        )

        if not results:
            passed("No results returned for unrelated query.")
            continue

        top_score = float(results[0]["score"])

        print(f"Top score: {top_score:.4f}")

        if top_score >= RELEVANCE_THRESHOLD:
            fail(
                "Unrelated query produced a result above "
                "the relevance threshold: "
                f"{top_score:.4f} >= "
                f"{RELEVANCE_THRESHOLD:.2f}"
            )
            failures += 1
        else:
            passed(
                "Unrelated query remained below the "
                f"relevance threshold: {top_score:.4f}"
            )

    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    vector_store.close()

    print()
    print("=" * 70)

    if failures == 0:
        print("SMOKE TEST RESULT: PASS")
        print("=" * 70)
        return 0

    print(
        f"SMOKE TEST RESULT: FAIL "
        f"({failures} failure(s))"
    )
    print("=" * 70)

    return 1


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    sys.exit(main())