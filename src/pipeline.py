"""
pipeline.py

Core LlamaIndex RAG pipeline for the Fiddler Docs Assistant.

Responsibilities:
  - Configure local HuggingFace embeddings (BAAI/bge-small-en-v1.5) and
    Anthropic Claude as the LLM via LlamaIndex Settings.
  - Load scraped .txt documents from data/ with section metadata.
  - Build and persist a VectorStoreIndex, or reload one from disk.
  - Expose a query engine (similarity_top_k=5, tree_summarize).

Usage:
    python src/pipeline.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
INDEX_STORE_DIR = ROOT_DIR / "index_store"

# ---------------------------------------------------------------------------
# LlamaIndex global settings
# ---------------------------------------------------------------------------


def configure_settings() -> None:
    """Set the global LlamaIndex LLM and embedding model.

    Called once at startup.  All subsequent LlamaIndex operations inherit
    these settings without needing to pass them explicitly.
    """
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Copy .env.example → .env and add your key."
        )

    print("Configuring LLM: claude-sonnet-4-5 (Anthropic)")
    Settings.llm = Anthropic(model="claude-sonnet-4-5", api_key=api_key)

    print("Configuring embeddings: BAAI/bge-small-en-v1.5 (local HuggingFace)")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------


def _file_metadata(file_path: str) -> dict[str, str]:
    """Return metadata dict for a document file.

    Derives the Fiddler documentation section from the file's parent
    directory name (e.g. data/observability/alerts-platform.txt →
    section='observability').
    """
    path = Path(file_path)
    return {
        "file_name": path.name,
        "section": path.parent.name,
    }


def load_documents(data_dir: Path = DATA_DIR) -> list:
    """Load all .txt files from data/ and attach section metadata.

    Returns a list of LlamaIndex Document objects.
    """
    print(f"Loading documents from: {data_dir}")
    reader = SimpleDirectoryReader(
        input_dir=str(data_dir),
        recursive=True,
        required_exts=[".txt"],
        file_metadata=_file_metadata,
    )
    docs = reader.load_data()
    print(f"  Loaded {len(docs)} documents")
    return docs


# ---------------------------------------------------------------------------
# Index building and persistence
# ---------------------------------------------------------------------------


def build_index(
    data_dir: Path = DATA_DIR,
    persist_dir: Path = INDEX_STORE_DIR,
) -> VectorStoreIndex:
    """Build a VectorStoreIndex from scraped docs and persist it to disk.

    Embeddings are generated locally using the model configured in
    ``configure_settings()``.  Building takes a few minutes on first run.

    Args:
        data_dir: Directory containing the scraped .txt files.
        persist_dir: Directory where the index will be saved.

    Returns:
        The built VectorStoreIndex.
    """
    docs = load_documents(data_dir)

    print(f"Building VectorStoreIndex ({len(docs)} documents)...")
    print("  This may take a few minutes while embeddings are generated.")
    index = VectorStoreIndex.from_documents(docs, show_progress=True)

    persist_dir.mkdir(parents=True, exist_ok=True)
    print(f"Persisting index to: {persist_dir}")
    index.storage_context.persist(persist_dir=str(persist_dir))
    print("  Index saved.")

    return index


def load_index(persist_dir: Path = INDEX_STORE_DIR) -> VectorStoreIndex:
    """Load a previously persisted VectorStoreIndex from disk.

    Args:
        persist_dir: Directory where the index was saved by ``build_index()``.

    Returns:
        The loaded VectorStoreIndex.
    """
    print(f"Loading index from: {persist_dir}")
    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    index = load_index_from_storage(storage_context)
    print("  Index loaded.")
    return index


# ---------------------------------------------------------------------------
# Query engine
# ---------------------------------------------------------------------------


def get_query_engine(index: VectorStoreIndex):
    """Return a query engine configured for RAG over the Fiddler docs.

    Settings:
        similarity_top_k=5  — retrieve the 5 most relevant chunks
        response_mode="tree_summarize"  — synthesise a grounded answer from
                                          retrieved nodes via recursive
                                          summarisation

    Args:
        index: A built or loaded VectorStoreIndex.

    Returns:
        A LlamaIndex query engine ready to answer natural language questions.
    """
    return index.as_query_engine(
        similarity_top_k=5,
        response_mode="tree_summarize",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    "How does Fiddler detect data drift, and what drift metrics does it support?",
    "How does Fiddler use SHAP values to explain model predictions?",
    "What metrics does Fiddler track for LLM monitoring, such as faithfulness or toxicity?",
    "How do I set up an alert in Fiddler and what notification channels are supported?",
    "What data integrity checks does Fiddler perform on model inputs?",
]


def _index_exists(persist_dir: Path) -> bool:
    """Return True if a persisted index is present in persist_dir."""
    return (persist_dir / "docstore.json").exists()


if __name__ == "__main__":
    configure_settings()

    if _index_exists(INDEX_STORE_DIR):
        print("\nFound existing index — loading from disk.")
        index = load_index()
    else:
        print("\nNo existing index found — building from scraped docs.")
        index = build_index()

    query_engine = get_query_engine(index)

    print(f"\n{'═' * 60}")
    print("Running test queries")
    print(f"{'═' * 60}\n")

    for i, query in enumerate(TEST_QUERIES, start=1):
        print(f"Query {i}/{len(TEST_QUERIES)}: {query}")
        print("─" * 60)

        response = query_engine.query(query)

        print(f"Response:\n{response}\n")

        print("Sources:")
        for node in response.source_nodes:
            score = f"{node.score:.4f}" if node.score is not None else "n/a"
            file_name = node.metadata.get("file_name", "unknown")
            section = node.metadata.get("section", "unknown")
            print(f"  [{score}]  {section}/{file_name}")

        print(f"\n{'═' * 60}\n")
