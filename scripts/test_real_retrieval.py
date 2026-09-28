from pathlib import Path

from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever


# ============================================================
# CONFIGURATION
# ============================================================

VECTOR_STORE_DIR = Path(
    "data/processed/vector_store"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("REAL DOCUMENT RETRIEVAL TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD EMBEDDING MODEL
    # --------------------------------------------------------

    embedding_model = EmbeddingModel(
        model_name="all-MiniLM-L6-v2",
        batch_size=32,
    )

    # --------------------------------------------------------
    # LOAD EXISTING VECTOR STORE
    # --------------------------------------------------------

    vector_store = VectorStore(
        index_dir=VECTOR_STORE_DIR,
        dimension=embedding_model.get_dimension(),
    )

    print()
    print(
        f"FAISS vectors : {vector_store.count()}"
    )

    print(
        f"SQLite records: "
        f"{vector_store.metadata_count()}"
    )

    # --------------------------------------------------------
    # CREATE RETRIEVER
    # --------------------------------------------------------

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        default_top_k=5,
    )

    # --------------------------------------------------------
    # QUERIES
    # --------------------------------------------------------

    queries = [
    "How do I enable Bluetooth on a Samsung phone?",
    "How can I turn on WiFi on a Samsung phone?",
    "What are the steps to turn Bluetooth on?",
]

    # --------------------------------------------------------
    # RUN QUERIES
    # --------------------------------------------------------

    for query in queries:

        print()
        print("=" * 70)
        print(
            f"QUERY: {query}"
        )
        print("=" * 70)

        results = retriever.retrieve(
            query=query,
            top_k=5,
        )

        if not results:

            print(
                "No results found."
            )

            continue

        for index, result in enumerate(
            results,
            start=1
        ):

            print()
            print(
                f"Result {index}"
            )

            print(
                f"Score    : "
                f"{result['score']:.4f}"
            )

            print(
                f"Chunk ID : "
                f"{result['chunk_id']}"
            )

            print(
                f"Source   : "
                f"{result['metadata'].get('source')}"
            )

            # Show page/row/record/sheet when available.

            metadata = result["metadata"]

            for field in (
                "page",
                "row",
                "record",
                "sheet",
                "slide",
            ):

                if field in metadata:

                    print(
                        f"{field.capitalize():9}: "
                        f"{metadata[field]}"
                    )

            print(
                f"Text     : "
                f"{result['text']}"
            )

    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    vector_store.close()

    print()
    print("=" * 70)
    print("REAL DOCUMENT RETRIEVAL TEST COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()