"""
experiments.py

Runs three experiments on top of the core RAG pipeline to explore how
retrieval quality is affected by (1) chunk size, (2) metadata filtering,
and (3) structural vs. fixed-token chunking.

Experiment 1 — Chunk size comparison
    Tests chunk sizes 256, 512, and 1024 tokens (overlap = 20% of chunk size).
    Each chunk size gets a fresh in-memory index.  The same 5 test queries run
    against each, and results are printed as a summary table.

Experiment 2 — Section-filtered retrieval
    Builds a single index with section metadata attached to every document.
    Runs 2 queries with no filter, then with an ExactMatch filter restricting
    retrieval to a specific section.  Results are printed side by side so the
    contrast is visible.

Experiment 3 — Structural (Markdown) vs. fixed-token chunking
    Compares MarkdownNodeParser (splits on heading boundaries) against the
    512-token SentenceSplitter baseline from Experiment 1.  For each strategy
    the experiment reports node-count/size statistics, per-query word count and
    source count, and an LLM-as-judge relevance score (1–5) so configurations
    can be ranked on actual response quality rather than proxy metrics alone.

Usage:
    python src/experiments.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

# Re-use shared setup from the core pipeline module.
import sys

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import (
    DATA_DIR,
    TEST_QUERIES,
    configure_settings,
    load_documents,
)

RESULTS_DIR = Path(__file__).parent.parent / "experiment_results"

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class EvalScores:
    """Multi-dimensional LLM-as-judge scores for a single response."""

    relevance: float     # Does it directly answer the question asked?
    completeness: float  # Does it cover all key aspects?
    specificity: float   # Concrete details vs. vague generalities?
    analysis: str = ""   # Judge's chain-of-thought reasoning

    @property
    def composite(self) -> float:
        """Equal-weight average across all three scored dimensions."""
        return (self.relevance + self.completeness + self.specificity) / 3.0


@dataclass
class QueryResult:
    """Holds the outcome of a single RAG query."""

    query: str
    response_text: str
    source_files: list[str]
    eval_scores: EvalScores | None = None

    @property
    def word_count(self) -> int:
        """Number of words in the response."""
        return len(self.response_text.split())

    @property
    def num_sources(self) -> int:
        """Number of source nodes retrieved."""
        return len(self.source_files)


@dataclass
class ChunkExperimentResult:
    """All query results for a single chunk size."""

    chunk_size: int
    chunk_overlap: int
    results: list[QueryResult] = field(default_factory=list)


@dataclass
class FilteredQueryPair:
    """Unfiltered and section-filtered responses for a single query."""

    query: str
    section: str
    unfiltered_response: str
    unfiltered_sources: list[str]
    filtered_response: str
    filtered_sources: list[str]
    unfiltered_eval_scores: EvalScores | None = None
    filtered_eval_scores: EvalScores | None = None


# ---------------------------------------------------------------------------
# Experiment 1 — Chunk size comparison
# ---------------------------------------------------------------------------


def _build_in_memory_index(
    chunk_size: int,
    chunk_overlap: int,
) -> VectorStoreIndex:
    """Build a fresh VectorStoreIndex in memory with the given chunking params.

    No persistence — results are discarded after the experiment.
    """
    docs = load_documents(DATA_DIR)
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    index = VectorStoreIndex.from_documents(
        docs,
        transformations=[splitter],
        show_progress=True,
    )
    return index


def run_chunk_size_experiment(
    chunk_sizes: list[int] | None = None,
) -> list[ChunkExperimentResult]:
    """Experiment 1: compare retrieval across three chunk sizes.

    For each chunk size, builds an in-memory index and runs all TEST_QUERIES.
    Records response word count, number of sources, and source file names.

    Args:
        chunk_sizes: Token counts to test.  Defaults to [256, 512, 1024].

    Returns:
        One ChunkExperimentResult per chunk size.
    """
    if chunk_sizes is None:
        chunk_sizes = [256, 512, 1024]

    all_results: list[ChunkExperimentResult] = []

    for chunk_size in chunk_sizes:
        chunk_overlap = int(chunk_size * 0.2)
        print(f"\n{'─' * 60}")
        print(f"Chunk size: {chunk_size}  |  Overlap: {chunk_overlap}")
        print(f"{'─' * 60}")

        index = _build_in_memory_index(chunk_size, chunk_overlap)
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode="tree_summarize",
        )

        experiment = ChunkExperimentResult(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for query in TEST_QUERIES:
            print(f"  Querying: {query[:70]}...")
            response = query_engine.query(query)
            response_text = str(response)
            source_files = [
                node.metadata.get("file_name", "unknown")
                for node in response.source_nodes
            ]
            print(f"    Scoring with LLM-as-judge (3 dimensions)...")
            score = _score_response(query, response_text)
            if score:
                print(f"      rel={score.relevance:.0f} comp={score.completeness:.0f} spec={score.specificity:.0f}  → {score.analysis}")
            experiment.results.append(
                QueryResult(
                    query=query,
                    response_text=response_text,
                    source_files=source_files,
                    eval_scores=score,
                )
            )

        all_results.append(experiment)

    return all_results


def print_chunk_experiment_summary(results: list[ChunkExperimentResult]) -> None:
    """Print a formatted comparison table for the chunk size experiment."""
    col_w = 20
    q_w = 42

    header_sizes = "  ".join(
        f"{'CS=' + str(r.chunk_size):>{col_w}}" for r in results
    )
    print(f"\n{'═' * 80}")
    print("EXPERIMENT 1 — Chunk Size Comparison")
    print(f"{'═' * 80}")
    print(f"{'Query':<{q_w}}  {header_sizes}")
    print(f"{'':─<{q_w}}  {'':─<{col_w * len(results) + 2 * (len(results) - 1)}}")

    # Zip per-query results across chunk sizes.
    for row in zip(*[r.results for r in results]):
        query_label = row[0].query[:q_w - 2] + ".." if len(row[0].query) > q_w else row[0].query
        cells = "  ".join(
            f"{'%dw/comp=%.1f' % (qr.word_count, qr.eval_scores.composite) if qr.eval_scores else '%dw/n/a' % qr.word_count:>{col_w}}"
            for qr in row
        )
        print(f"{query_label:<{q_w}}  {cells}")

    print(f"\n{'─' * 80}")
    print("Column key:  <words>w/comp=<composite score 1–5>")
    print(f"{'─' * 80}\n")

    # Averages with per-dimension breakdown
    print("Averages per chunk size:")
    for r in results:
        avg_words = sum(qr.word_count for qr in r.results) / len(r.results)
        avg_src = sum(qr.num_sources for qr in r.results) / len(r.results)
        scored = [qr.eval_scores for qr in r.results if qr.eval_scores is not None]
        if scored:
            avg_rel  = sum(s.relevance     for s in scored) / len(scored)
            avg_comp = sum(s.completeness  for s in scored) / len(scored)
            avg_spec = sum(s.specificity   for s in scored) / len(scored)
            avg_composite = sum(s.composite for s in scored) / len(scored)
            score_str = (f"rel={avg_rel:.2f}  comp={avg_comp:.2f}  "
                         f"spec={avg_spec:.2f}  composite={avg_composite:.2f}")
        else:
            score_str = "n/a"
        print(
            f"  chunk_size={r.chunk_size:<5}  "
            f"avg words={avg_words:>6.1f}  avg sources={avg_src:.1f}  {score_str}"
        )

    print(f"\n{'═' * 80}")
    print(
        "INTERPRETATION: Smaller chunks (256) tend to retrieve more topically\n"
        "precise passages while larger chunks (1024) capture more surrounding\n"
        "context per node.  In practice on this corpus, 256 and 1024 perform\n"
        "comparably on most queries, while 512 is the most sensitive to retrieval\n"
        "misses — a single bad retrieval (e.g. data integrity query pulling\n"
        "irrelevant chunks) collapses its composite score more than the others.\n"
        "The best chunk size depends on which queries matter most; see\n"
        "experiment_results/exp1_chunk_sizes.json for per-query breakdowns."
    )
    print(f"{'═' * 80}\n")


# ---------------------------------------------------------------------------
# Experiment 2 — Section-filtered retrieval
# ---------------------------------------------------------------------------

FILTER_QUERIES = [
    (
        "What drift metrics does Fiddler track for production models?",
        "observability",
    ),
    (
        "How do I get started with LLM monitoring in Fiddler?",
        "getting-started",
    ),
]


def run_filter_experiment() -> list[FilteredQueryPair]:
    """Experiment 2: compare unfiltered vs. section-filtered retrieval.

    Builds one shared index, then for each query runs it twice:
      - once with no metadata filter (full corpus)
      - once filtered to the section most relevant to the query

    Prints both responses side by side so the contrast is visible.

    Returns:
        One FilteredQueryPair per query in FILTER_QUERIES.
    """
    print(f"\n{'═' * 80}")
    print("EXPERIMENT 2 — Section-Filtered Retrieval")
    print(f"{'═' * 80}\n")

    print("Building shared index for filter experiment...")
    docs = load_documents(DATA_DIR)
    index = VectorStoreIndex.from_documents(
        docs,
        transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=102)],
        show_progress=True,
    )

    unfiltered_engine = index.as_query_engine(
        similarity_top_k=5,
        response_mode="tree_summarize",
    )

    pairs: list[FilteredQueryPair] = []

    for query, section in FILTER_QUERIES:
        print(f"\nQuery   : {query}")
        print(f"Section filter: '{section}'")
        print("─" * 80)

        # Unfiltered query
        print("[ NO FILTER ]")
        unfiltered_response = unfiltered_engine.query(query)
        unfiltered_sources = [
            f"{n.metadata.get('section','?')}/{n.metadata.get('file_name','?')}"
            for n in unfiltered_response.source_nodes
        ]
        unfiltered_text = str(unfiltered_response)
        print(f"Response ({len(unfiltered_text.split())} words):")
        print(unfiltered_text[:600] + ("..." if len(unfiltered_text) > 600 else ""))
        print("Sources:")
        for s in unfiltered_sources:
            print(f"  {s}")
        print(f"  Scoring with LLM-as-judge (3 dimensions)...")
        unfiltered_score = _score_response(query, unfiltered_text)
        if unfiltered_score:
            print(f"  rel={unfiltered_score.relevance:.0f} comp={unfiltered_score.completeness:.0f} spec={unfiltered_score.specificity:.0f}  → {unfiltered_score.analysis}")

        # Section-filtered query
        print(f"\n[ FILTERED TO section='{section}' ]")
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="section",
                    value=section,
                    operator=FilterOperator.EQ,
                )
            ]
        )
        filtered_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode="tree_summarize",
            filters=filters,
        )
        filtered_response = filtered_engine.query(query)
        filtered_sources = [
            f"{n.metadata.get('section','?')}/{n.metadata.get('file_name','?')}"
            for n in filtered_response.source_nodes
        ]
        filtered_text = str(filtered_response)
        print(f"Response ({len(filtered_text.split())} words):")
        print(filtered_text[:600] + ("..." if len(filtered_text) > 600 else ""))
        print("Sources:")
        for s in filtered_sources:
            print(f"  {s}")
        print(f"  Scoring with LLM-as-judge (3 dimensions)...")
        filtered_score = _score_response(query, filtered_text)
        if filtered_score:
            print(f"  rel={filtered_score.relevance:.0f} comp={filtered_score.completeness:.0f} spec={filtered_score.specificity:.0f}  → {filtered_score.analysis}")

        print("─" * 80)

        pairs.append(
            FilteredQueryPair(
                query=query,
                section=section,
                unfiltered_response=unfiltered_text,
                unfiltered_sources=unfiltered_sources,
                filtered_response=filtered_text,
                filtered_sources=filtered_sources,
                unfiltered_eval_scores=unfiltered_score,
                filtered_eval_scores=filtered_score,
            )
        )

    print(f"\n{'═' * 80}")
    print(
        "INTERPRETATION: Without a filter, retrieval casts a wide net across all\n"
        "sections, which is useful for broad questions but can introduce noise from\n"
        "unrelated parts of the docs (e.g. API reference pages appearing for a\n"
        "conceptual question).  Section-filtered retrieval constrains the search to\n"
        "the most topically relevant part of the corpus, producing more focused\n"
        "answers — at the cost of missing cross-section context when the answer\n"
        "genuinely spans multiple areas."
    )
    print(f"{'═' * 80}\n")

    return pairs


# ---------------------------------------------------------------------------
# Experiment 3 — Structural (Markdown) chunking vs. fixed-token baseline
# ---------------------------------------------------------------------------


@dataclass
class StructuralExperimentResult:
    """Results for one chunking strategy in the structural experiment."""

    strategy: str
    num_nodes: int
    avg_node_chars: float
    min_node_chars: int
    max_node_chars: int
    results: list[QueryResult] = field(default_factory=list)

    @property
    def avg_relevance(self) -> float | None:
        """Mean composite LLM-as-judge score across all queries, or None."""
        scored = [r.eval_scores.composite for r in self.results if r.eval_scores is not None]
        return sum(scored) / len(scored) if scored else None


def _score_response(query: str, response_text: str) -> EvalScores | None:
    """Score a RAG response on three dimensions using LLM-as-judge with CoT.

    Asks the configured LLM to reason about the response before scoring,
    producing relevance, completeness, and specificity scores on a fully
    defined 1–5 rubric.  The result is parsed from JSON output.

    The three improvements over a single-dimension 1–5 prompt:
      1. All five rubric levels are explicitly defined (no ambiguous mid-points).
      2. The model writes a brief analysis before committing to numbers,
         reducing anchoring bias and surface-level pattern-matching.
      3. Three separate dimensions expose failure modes that a single
         "relevance" score would collapse (e.g. relevant but vague = high
         relevance, low specificity).

    Returns:
        EvalScores on success, None on parse or API failure.
    """
    prompt = (
        "You are evaluating a RAG system's response. Score the RESPONSE against "
        "the QUESTION on three dimensions.\n\n"
        "First, write a brief analysis of what the response covers well and what "
        "it misses or gets wrong. Then assign integer scores.\n\n"
        "RUBRIC (apply independently to each dimension):\n"
        "  1 = Off-topic, factually wrong, or fails to address this dimension\n"
        "  2 = Mentions the right topic but misses the core substance\n"
        "  3 = Addresses the dimension but is vague or missing important details\n"
        "  4 = Good with relevant specifics, but not fully comprehensive\n"
        "  5 = Thorough, specific, and complete — every key aspect covered with "
        "concrete details\n\n"
        "DIMENSIONS:\n"
        "  relevance    — Does the response directly answer the question asked?\n"
        "  completeness — Does it cover all key aspects of the question?\n"
        "  specificity  — Does it give concrete details (names, steps, examples) "
        "rather than vague generalities?\n\n"
        f"QUESTION: {query}\n\n"
        f"RESPONSE: {response_text}\n\n"
        "Respond in this EXACT JSON format — no markdown fences, no extra text:\n"
        '{"analysis": "<one or two sentences>", '
        '"relevance": <1-5>, "completeness": <1-5>, "specificity": <1-5>}'
    )
    try:
        raw = str(Settings.llm.complete(prompt)).strip()
        # Strip markdown code fences if the model wraps the JSON anyway.
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        return EvalScores(
            relevance=float(data["relevance"]),
            completeness=float(data["completeness"]),
            specificity=float(data["specificity"]),
            analysis=str(data.get("analysis", "")),
        )
    except Exception as exc:
        print(f"    Warning: LLM-as-judge scoring failed ({exc})")
    return None


def _build_structural_index(strategy: str) -> tuple[VectorStoreIndex, dict]:
    """Build an in-memory index for the given strategy and collect node stats.

    Args:
        strategy: ``"markdown"`` uses MarkdownNodeParser (splits on headings)
                  with a SentenceSplitter fallback cap of 1024 tokens.
                  ``"sentence_512"`` uses the plain 512-token baseline.

    Returns:
        (index, stats_dict) where stats_dict has keys: num_nodes, avg_chars,
        min_chars, max_chars.
    """
    docs = load_documents(DATA_DIR)

    if strategy == "markdown":
        # MarkdownNodeParser first, then cap oversized sections at 1024 tokens.
        transformations = [
            MarkdownNodeParser(),
            SentenceSplitter(chunk_size=1024, chunk_overlap=200),
        ]
    else:
        transformations = [SentenceSplitter(chunk_size=512, chunk_overlap=102)]

    index = VectorStoreIndex.from_documents(
        docs,
        transformations=transformations,
        show_progress=True,
    )

    node_lengths = [
        len(node.get_content())
        for node in index.storage_context.docstore.docs.values()
    ]
    stats = {
        "num_nodes": len(node_lengths),
        "avg_chars": sum(node_lengths) / len(node_lengths) if node_lengths else 0,
        "min_chars": min(node_lengths) if node_lengths else 0,
        "max_chars": max(node_lengths) if node_lengths else 0,
    }
    return index, stats


def run_structural_chunking_experiment() -> list[StructuralExperimentResult]:
    """Experiment 3: MarkdownNodeParser vs. 512-token SentenceSplitter.

    Builds one index per strategy, runs all TEST_QUERIES against each,
    and scores every response with a 1–5 LLM-as-judge relevance rating so
    configurations can be ranked on quality rather than proxy metrics alone.

    Returns:
        One StructuralExperimentResult per strategy.
    """
    strategies = ["sentence_512", "markdown"]
    all_results: list[StructuralExperimentResult] = []

    for strategy in strategies:
        label = "Markdown (structural)" if strategy == "markdown" else "SentenceSplitter 512 (baseline)"
        print(f"\n{'─' * 60}")
        print(f"Strategy: {label}")
        print(f"{'─' * 60}")

        index, stats = _build_structural_index(strategy)
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode="tree_summarize",
        )

        experiment = StructuralExperimentResult(
            strategy=label,
            num_nodes=stats["num_nodes"],
            avg_node_chars=stats["avg_chars"],
            min_node_chars=stats["min_chars"],
            max_node_chars=stats["max_chars"],
        )

        for query in TEST_QUERIES:
            print(f"  Querying: {query[:70]}...")
            response = query_engine.query(query)
            response_text = str(response)
            source_files = [
                node.metadata.get("file_name", "unknown")
                for node in response.source_nodes
            ]
            print(f"    Scoring with LLM-as-judge (3 dimensions)...")
            score = _score_response(query, response_text)
            if score:
                print(f"      rel={score.relevance:.0f} comp={score.completeness:.0f} spec={score.specificity:.0f}  → {score.analysis}")
            experiment.results.append(
                QueryResult(
                    query=query,
                    response_text=response_text,
                    source_files=source_files,
                    eval_scores=score,
                )
            )

        all_results.append(experiment)

    return all_results


def print_structural_experiment_summary(results: list[StructuralExperimentResult]) -> None:
    """Print a formatted comparison table for the structural chunking experiment."""
    print(f"\n{'═' * 80}")
    print("EXPERIMENT 3 — Structural (Markdown) vs. Fixed-Token Chunking")
    print(f"{'═' * 80}\n")

    # Node statistics
    print(f"{'Strategy':<35}  {'Nodes':>6}  {'Avg chars':>10}  {'Min':>6}  {'Max':>7}")
    print(f"{'':─<35}  {'':─>6}  {'':─>10}  {'':─>6}  {'':─>7}")
    for r in results:
        print(
            f"{r.strategy:<35}  {r.num_nodes:>6}  "
            f"{r.avg_node_chars:>10.0f}  {r.min_node_chars:>6}  {r.max_node_chars:>7}"
        )

    print(f"\n{'─' * 80}")
    print(f"{'Query':<42}  ", end="")
    for r in results:
        print(f"  {r.strategy[:18]:>18}", end="")
    print()
    print(f"{'':─<42}  {'':─<{20 * len(results)}}")

    for row in zip(*[r.results for r in results]):
        query_label = row[0].query[:40] + ".." if len(row[0].query) > 42 else row[0].query
        print(f"{query_label:<42}", end="")
        for qr in row:
            comp_str = f"{qr.eval_scores.composite:.1f}" if qr.eval_scores else "n/a"
            cell = f"{qr.word_count}w  comp={comp_str}"
            print(f"  {cell:>18}", end="")
        print()

    print(f"\n{'─' * 80}")
    print("Column key:  <word count>w  comp=<composite score 1–5>")
    print(f"{'─' * 80}\n")

    print("Averages per strategy:")
    for r in results:
        avg_words = sum(qr.word_count for qr in r.results) / len(r.results)
        scored = [qr.eval_scores for qr in r.results if qr.eval_scores is not None]
        if scored:
            avg_rel  = sum(s.relevance     for s in scored) / len(scored)
            avg_comp = sum(s.completeness  for s in scored) / len(scored)
            avg_spec = sum(s.specificity   for s in scored) / len(scored)
            avg_composite = sum(s.composite for s in scored) / len(scored)
            score_str = (f"rel={avg_rel:.2f}  comp={avg_comp:.2f}  "
                         f"spec={avg_spec:.2f}  composite={avg_composite:.2f}")
        else:
            score_str = "n/a"
        print(f"  {r.strategy:<35}  avg words={avg_words:>6.1f}  {score_str}")

    best = max(results, key=lambda r: r.avg_relevance or 0.0)
    print(f"\n  → Highest avg composite: {best.strategy}")

    print(f"\n{'═' * 80}")
    print(
        "INTERPRETATION: MarkdownNodeParser preserves the natural section\n"
        "boundaries of Fiddler's documentation (each ## heading becomes its\n"
        "own node), which keeps conceptually related content together. This\n"
        "typically improves relevance for focused questions. However, section\n"
        "length varies widely, so a SentenceSplitter fallback cap (1024 tokens)\n"
        "is applied to prevent oversized nodes from diluting similarity scores.\n"
        "Fixed-token splitting is simpler and more uniform but can cut mid-\n"
        "explanation; it remains a strong baseline for short factual queries."
    )
    print(f"{'═' * 80}\n")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_experiment_results(
    chunk_results: list[ChunkExperimentResult],
    filter_results: list[FilteredQueryPair],
    structural_results: list[StructuralExperimentResult],
) -> None:
    """Write all experiment results to JSON files under experiment_results/.

    Each experiment gets its own file and is overwritten on every run.
    The run timestamp is embedded so git diffs show when results changed.

    Files written:
        experiment_results/exp1_chunk_sizes.json
        experiment_results/exp2_section_filter.json
        experiment_results/exp3_structural.json
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    files = {
        "exp1_chunk_sizes.json": {
            "experiment": "Chunk Size Comparison",
            "run_timestamp": timestamp,
            "results": [asdict(r) for r in chunk_results],
        },
        "exp2_section_filter.json": {
            "experiment": "Section-Filtered Retrieval",
            "run_timestamp": timestamp,
            "results": [asdict(p) for p in filter_results],
        },
        "exp3_structural.json": {
            "experiment": "Structural vs. Fixed-Token Chunking",
            "run_timestamp": timestamp,
            "results": [asdict(r) for r in structural_results],
        },
    }

    for filename, payload in files.items():
        out_path = RESULTS_DIR / filename
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  Saved → {out_path.relative_to(RESULTS_DIR.parent)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    configure_settings()

    print("\n" + "═" * 80)
    print("FIDDLER RAG — RETRIEVAL EXPERIMENTS")
    print("=" * 80)

    # --- Experiment 1 ---
    print("\nStarting Experiment 1: Chunk Size Comparison")
    print("(Builds 3 separate in-memory indexes — takes a few minutes)\n")
    chunk_results = run_chunk_size_experiment()
    print_chunk_experiment_summary(chunk_results)

    # --- Experiment 2 ---
    print("\nStarting Experiment 2: Section-Filtered Retrieval")
    filter_results = run_filter_experiment()

    # --- Experiment 3 ---
    print("\nStarting Experiment 3: Structural (Markdown) vs. Fixed-Token Chunking")
    print("(Builds 2 indexes and scores responses with LLM-as-judge — takes a few minutes)\n")
    structural_results = run_structural_chunking_experiment()
    print_structural_experiment_summary(structural_results)

    # --- Save results ---
    print(f"\n{'─' * 60}")
    print("Saving results to experiment_results/")
    save_experiment_results(chunk_results, filter_results, structural_results)

    print("All experiments complete.")
