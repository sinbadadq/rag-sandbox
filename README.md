# Fiddler Docs Assistant

A retrieval-augmented generation (RAG) pipeline over [Fiddler AI's public documentation](https://docs.fiddler.ai), built as part of interview preparation for a data scientist role at Fiddler AI.

The system scrapes Fiddler's docs, indexes them with [LlamaIndex](https://www.llamaindex.ai/) using local [HuggingFace embeddings](https://huggingface.co/BAAI/bge-small-en-v1.5) (no external embedding API cost), and answers natural language questions about Fiddler's ML observability platform using [Anthropic Claude](https://www.anthropic.com/) as the LLM.

A suite of three experiments compares chunking strategies, retrieval filters, and LLM-as-judge evaluation — results are persisted in `experiment_results/`. See [METHODOLOGY.md](METHODOLOGY.md) for a detailed write-up of the experiment design, evaluation approach, and concept explanations.

## Project Structure

```
rag-sandbox/
  data/                   # scraped Fiddler docs, organized by section (323 files)
  src/
    scraper.py            # scrapes docs.fiddler.ai → data/
    pipeline.py           # LlamaIndex settings, document loading, index build/load
    experiments.py        # three RAG experiments with LLM-as-judge evaluation
  app/
    app.py                # Streamlit UI for interactive Q&A
  experiment_results/     # saved JSON output from each experiment
    exp1_chunk_sizes.json
    exp2_section_filter.json
    exp3_structural.json
  index_store/            # persisted vector index (generated, not committed)
  Makefile                # conda environment helpers
  requirements.txt
  .env.example
  README.md
```

## Getting Started

### 1. Create and activate the conda environment

```bash
make create-env
conda activate rag-sandbox
```

### 2. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and fill in your Anthropic API key:

```
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

### 3. *(Optional)* Re-scrape Fiddler's documentation

The scraped corpus is already committed in `data/`. Re-scraping is only needed if you want to pull fresher content from docs.fiddler.ai.

```bash
python src/scraper.py
```

### 4. Build the vector index

The `index_store/` directory is not committed (embeddings are auto-generated). Build it once before querying:

```bash
python src/pipeline.py
```

The index is persisted to `index_store/` and reused on subsequent runs. Building takes a few minutes while local embeddings are generated.

### 5. Ask questions via the Streamlit app

```bash
streamlit run app/app.py
```

Open the URL printed in the terminal.

### 6. *(Optional)* Re-run the experiments

```bash
python src/experiments.py
```

Results are written to `experiment_results/` as JSON files.

## Experiments

Three experiments were run to evaluate different RAG configurations using an **LLM-as-judge** scorer that rates responses on three dimensions (1–5 scale each):

| Dimension | What it measures |
|---|---|
| **Relevance** | Does the answer address the question? |
| **Completeness** | Does it cover all key aspects? |
| **Specificity** | Does it cite Fiddler-specific details? |

A **composite score** averages all three.  The judge uses chain-of-thought reasoning (an `analysis` field) before scoring to reduce positional bias and score degeneracies.

### Experiment 1 — Chunk Size

Compares `SentenceSplitter` chunk sizes of **256**, **512**, and **1024 tokens** across five representative queries.

Key finding: 256 and 1024 perform comparably; 512 is most sensitive to retrieval misses — a single poorly-matched chunk collapses its composite score more than the other sizes.  See `experiment_results/exp1_chunk_sizes.json` for per-query breakdowns.

### Experiment 2 — Metadata Filtering

Tests whether restricting retrieval to a specific documentation **section** (e.g. `observability`) improves answer quality for section-specific queries, compared to searching the full corpus.

Key finding: Filtering consistently improves specificity and completeness for targeted questions by preventing cross-section noise, but can hurt recall for queries whose answers span multiple sections.

### Experiment 3 — Structural Chunking

Compares **`MarkdownNodeParser`** (splits on heading boundaries, with a `SentenceSplitter` fallback for oversized sections) against **fixed-token splitting** at 512 tokens.

Key finding: Structural chunking improves specificity on content-dense sections where a single heading covers a coherent concept.  Fixed-token splitting is more robust for long prose sections where heading boundaries are sparse.

## Tech Stack

| Component | Choice |
|---|---|
| Framework | [LlamaIndex](https://www.llamaindex.ai/) |
| Embeddings | `BAAI/bge-small-en-v1.5` (local, via HuggingFace) |
| LLM | Anthropic Claude (`claude-sonnet-4-5`) |
| UI | Streamlit |
| Scraping | `requests` + `beautifulsoup4` |
