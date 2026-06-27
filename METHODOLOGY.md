# Methodology: Building and Evaluating a RAG Pipeline

This document walks through the design decisions, experiment progression, and evaluation approach taken in this project. It is written to be readable by both technical and non-technical audiences — concepts are explained inline as they are introduced.

---

## Background: What is RAG?

**Retrieval-Augmented Generation (RAG)** is a pattern for making large language models (LLMs) useful over a specific knowledge base. Instead of relying on the LLM's training data (which may be stale, generic, or absent), a RAG system:

1. **Indexes** the knowledge base by converting documents into numerical vectors called *embeddings* — a mathematical representation of meaning that allows semantically similar text to be stored near each other.
2. **Retrieves** the most relevant chunks of text for a given query using vector similarity search.
3. **Generates** an answer by sending both the query and retrieved chunks to an LLM, which synthesizes a grounded response.

The quality of a RAG system depends heavily on how the source documents are split (chunked) before indexing, how retrieval is scoped, and how the final response is evaluated.

---

## The Corpus

The knowledge base is Fiddler AI's public documentation, scraped from [docs.fiddler.ai](https://docs.fiddler.ai) using `src/scraper.py`. The corpus totals **323 documents** (~3.1 MB of text), organized into sections: `api`, `observability`, `getting-started`, `developers`, `integrations`, `reference`, `protect-and-guardrails`, `evaluate-and-test`, `changelog`, and `concepts`.

Documents are stored as Markdown-formatted `.txt` files in `data/`, preserving heading structure from the original site — which becomes important in Experiment 3.

---

## Experiment Design

### Why run experiments at all?

The core pipeline (`src/pipeline.py`) establishes sensible defaults: a 512-token chunk size with 20% overlap, `similarity_top_k=5` retrieval, and a `tree_summarize` response mode. But "sensible defaults" is not evidence of optimality. Several design choices meaningfully affect answer quality:

- **Chunk size** controls the granularity of retrieved passages. Too small, and retrieved chunks may lack context. Too large, and a single chunk may cover multiple topics, diluting the signal for any one of them.
- **Retrieval scope** (filtered vs. unfiltered) determines whether search is global or constrained to a known relevant section.
- **Chunking strategy** — splitting on a fixed token budget versus respecting the document's own heading structure — may better preserve coherent units of meaning for documentation.

`src/experiments.py` runs three experiments to test these variables systematically.

---

## The Experiments

### Experiment 1 — Chunk Size Comparison

**Configurations tested:** `SentenceSplitter` with chunk sizes of 256, 512, and 1024 tokens (overlap fixed at 20%).

**Test queries (5 total):**
1. How does Fiddler detect data drift, and what drift metrics does it support?
2. How does Fiddler use SHAP values to explain model predictions?
3. What metrics does Fiddler track for LLM monitoring, such as faithfulness or toxicity?
4. How do I set up an alert in Fiddler and what notification channels are supported?
5. What data integrity checks does Fiddler perform on model inputs?

These queries were chosen to span different documentation sections and require different levels of specificity — from high-level conceptual answers to procedural, implementation-level details.

**Results summary (average composite score, 1–5 scale):**

| Chunk Size | Avg Composite |
|---|---|
| 256 tokens | **4.73** |
| 512 tokens | 4.13 |
| 1024 tokens | 4.67 |

**Key finding:** 512 tokens performed worst — not because it is inherently worse, but because it failed completely on a single query (data integrity checks, composite score: **2.0**). The retriever surfaced irrelevant chunks (`agentic-document-extraction.txt`, `trust-service.txt`, `mcp-server.txt`) rather than `data-integrity-platform.txt`. At 512 tokens, this section happened to be split across chunk boundaries in a way that degraded its similarity score relative to the query. The 256 and 1024 configurations both retrieved the right source and scored 5.0 on that query.

This illustrates an important property of RAG systems: **a single retrieval failure has a disproportionate impact on the average score**. 512's aggregate score looks poor, but the underlying issue is not the chunk size — it is the stochastic nature of chunk boundary alignment with specific queries.

---

### Experiment 2 — Section-Filtered Retrieval

**Motivation:** Fiddler's documentation attaches a `section` metadata field to every document (derived from the directory structure in `data/`). When a question is clearly scoped to a specific section — e.g., a question about monitoring metrics is most relevant to the `observability` section — restricting the vector search to that section may reduce noise.

**What metadata filtering does:** Instead of ranking all 1,491 chunks globally, the query engine only considers chunks whose `section` field matches the specified filter. This narrows the candidate pool from the full corpus to a single documentation section.

**Results:**

| Query | Unfiltered composite | Filtered composite | Winner |
|---|---|---|---|
| Drift metrics (section: `observability`) | 4.0 | **5.0** | Filtered |
| LLM monitoring onboarding (section: `getting-started`) | **4.33** | 4.0 | Unfiltered |

**Key finding:** Filtering helped when the answer lived primarily in one section (drift metrics are well-covered in `observability`). It hurt when the answer benefited from cross-section context — the LLM monitoring question was well-answered using a broader sweep that included `developers/` and `getting-started/` sources together. The unfiltered composite was 4.33 vs. filtered's 4.0, with specificity dropping from 4 to 3 under filtering.

**Conclusion:** Section filtering is a useful tool for targeted, section-scoped queries, but should not be applied blindly. The better design is to apply it selectively when the query's intent can be mapped to a specific section with confidence.

---

### Experiment 3 — Structural vs. Fixed-Token Chunking

**Motivation:** The corpus is Markdown-formatted documentation. Every document has a natural heading hierarchy — `#`, `##`, `###` — that delineates semantic boundaries. A structural chunker respects these boundaries; a fixed-token chunker may split mid-section.

**Configurations:**
- **Baseline:** `SentenceSplitter` at 512 tokens (same as Experiment 1's 512 configuration)
- **Structural:** `MarkdownNodeParser`, which splits at every heading boundary, with a `SentenceSplitter` fallback for sections that exceed 1024 tokens

**Index statistics:**

| Strategy | Nodes | Avg chars/node | Min | Max |
|---|---|---|---|---|
| SentenceSplitter 512 | 1,491 | 1,900 | 3 | 7,278 |
| MarkdownNodeParser | **3,552** | 721 | 6 | 14,675 |

The structural chunker creates more than twice as many nodes, each much smaller on average. This is because the docs have many short sections. The 14,675-character maximum is a single oversized section that the `SentenceSplitter` fallback handles.

**Results summary (average composite score):**

| Strategy | Avg Composite |
|---|---|
| SentenceSplitter 512 | 4.20 |
| MarkdownNodeParser | **4.33** |

**Per-query breakdown:**

| Query | SentenceSplitter 512 | MarkdownNodeParser |
|---|---|---|
| Data drift | 4.33 | 4.33 |
| SHAP values | 4.67 | 4.67 |
| LLM monitoring metrics | **5.0** | 4.67 |
| Alert setup | **5.0** | 3.67 |
| Data integrity checks | 2.0 | **4.33** |

**Key findings:**
- Structural chunking **fixed the data integrity failure** from Experiment 1. By keeping the `data-integrity-platform.txt` content within its own heading boundaries, the relevant chunk stayed intact and scored 4.33 vs. 2.0.
- Structural chunking **hurt the alerts query**. The notification channels section was split into a separate Markdown heading node, and the retriever surfaced only the alert configuration nodes — meaning it answered "how to set up an alert" but lacked the notification channels content. Fixed-token splitting kept both pieces in the same chunk and retrieved a complete answer.
- Neither strategy dominates across all queries. The tradeoff is: structural chunking preserves conceptual coherence but may separate procedural steps that appear under different headings; fixed-token chunking treats all boundaries as equally significant and occasionally merges or splits at inopportune places.

---

## Evaluation: LLM-as-Judge

### Why a human-readable quality score?

The experiments above compare different retrieval strategies. To compare them fairly, we need a quality signal. The most common options are:

- **Ground truth matching:** Compare the system's answer against a reference answer. Requires a labeled dataset.
- **Human evaluation:** A human rates each response. Accurate but slow and expensive.
- **LLM-as-judge:** Use a capable LLM to rate each response against the original question. Fast, automatable, and surprisingly consistent with human judgment when prompted carefully.

This project uses the **LLM-as-judge** approach, implemented in `_score_response()` in `src/experiments.py`. For each (query, response) pair, the judge LLM receives a structured prompt asking it to evaluate the response on three dimensions.

---

### The Scoring Dimensions

Each response receives three scores on a 1–5 integer scale:

#### Relevance
Does the response actually answer the question that was asked? A response that answers a different question (even a related one) or hedges without providing substantive content scores low.

| Score | Meaning |
|---|---|
| 5 | Directly and precisely answers the question |
| 4 | Mostly answers the question with minor gaps |
| 3 | Partially addresses the question |
| 2 | Tangentially related but doesn't answer |
| 1 | Does not address the question |

#### Completeness
Does the response cover all the key aspects of the question? A concise, correct partial answer may score high on relevance but low on completeness.

| Score | Meaning |
|---|---|
| 5 | Covers all key aspects thoroughly |
| 4 | Covers most aspects with minor gaps |
| 3 | Covers the main point but misses important aspects |
| 2 | Covers only superficial aspects |
| 1 | Covers almost nothing of substance |

#### Specificity
Does the response cite concrete, Fiddler-specific details — names of metrics, APIs, configuration parameters, SDK methods — rather than staying generic? This is particularly important in a documentation assistant context, where "use the API to configure alerts" is far less useful than naming the `AlertRule` class and its parameters.

| Score | Meaning |
|---|---|
| 5 | Highly specific, names Fiddler-specific details throughout |
| 4 | Mostly specific with some generic phrasing |
| 3 | Mix of specific and generic content |
| 2 | Mostly generic, minimal Fiddler-specific detail |
| 1 | Entirely generic, could apply to any platform |

#### Composite Score
The composite score is the arithmetic mean of the three dimensions:

```
composite = (relevance + completeness + specificity) / 3
```

This gives equal weight to all three dimensions. A response that perfectly answers the question (`relevance=5`) but is vague and generic (`completeness=3, specificity=2`) would score `(5+3+2)/3 = 3.33`, reflecting that its usefulness is limited.

---

### Iterating on the Judge: Breaking Score Degeneracies

#### The first version: ceiling effect

The initial evaluation prompt asked the judge to score responses on a 1–5 scale with only three reference points defined (1 = irrelevant, 3 = partial, 5 = fully answers). In practice, the judge assigned a 5 to nearly every response — because documentation retrieval tends to produce at least plausible-looking answers, and an underspecified rubric provided no principled basis for using 4 or lower.

This is a known failure mode called a **ceiling effect**: scores cluster at the top of the scale, making it impossible to distinguish good responses from great ones.

**Empirical evidence of the problem:** With the original rubric, the vast majority of query scores were 5.0 across all chunk sizes, making the experiment results meaningless — every configuration looked equivalent.

#### Three targeted fixes

**1. Full 5-point rubric (all levels defined)**

Rather than anchoring only three levels (1, 3, 5), the rubric was extended to define all five levels for each dimension. This gave the judge a concrete reference point when deciding between a 4 and a 5, or a 2 and a 3 — preventing it from defaulting to the ceiling.

**2. Chain-of-thought reasoning (analysis field)**

The prompt was modified to require an `analysis` field in the judge's JSON output *before* the scores. This forces the judge to articulate its reasoning before committing to a number — a technique borrowed from chain-of-thought prompting that is known to improve LLM consistency and reduce anchoring bias.

Example judge output:
```json
{
  "analysis": "The response directly addresses the question by naming JSD and PSI as the supported drift metrics and explaining the baseline comparison mechanism. However, it does not mention implementation details like binning methods, threshold configuration, or the vector embedding drift detection approach, which limits completeness.",
  "relevance": 5,
  "completeness": 4,
  "specificity": 4
}
```

**3. Multi-dimensional scoring**

Separating the evaluation into three distinct dimensions forced the judge to consider different qualities independently. A response can be highly relevant (answers the question) but low specificity (uses generic language). Collapsing this into a single "relevance" score made these tradeoffs invisible.

#### Effect of the improvements

After implementing all three changes, the score distribution showed meaningful spread:
- The data integrity query under 512-token chunking received a **2.0 composite** (relevance=3, completeness=2, specificity=1), correctly identifying it as a near-total retrieval failure.
- Alerts queries that answered only half the question (how to set up alerts but not notification channels) received **3.67 composites** under structural chunking, distinguishing them from complete answers scoring 5.0.
- The top-scoring configurations genuinely scored higher (4.67–5.0) while poor responses registered clearly below the midpoint.

---

## Concepts Glossary

### SHAP Values
**SHAP** (SHapley Additive exPlanations) is a method for explaining machine learning model predictions by quantifying how much each input feature contributed to a specific prediction. It is grounded in cooperative game theory: the SHAP value for a feature is computed by averaging its marginal contribution across all possible orderings of features.

Concretely: if a model predicts a customer has a 70% churn probability, SHAP values decompose that 70% into per-feature contributions. "Account age contributed −15 percentage points (kept the prediction down), tenure contributed +20 percentage points, recent support tickets contributed +10 percentage points," and so on. The contributions sum to the difference between the model's prediction and the expected prediction across all data.

Fiddler implements two variants: standard SHAP and `FIDDLER_SHAP`, an optimized version that maintains the same theoretical guarantees with better production performance.

### Jensen-Shannon Divergence (JSD)
A statistical measure of how different two probability distributions are. JSD = 0 means the distributions are identical; JSD = 1 means they share no probability mass. Fiddler uses JSD as its primary data drift metric: if a feature's distribution in production diverges significantly from its distribution at training time, JSD rises, triggering a drift alert.

JSD is symmetric (the distance from A to B equals B to A) and bounded between 0 and 1, which makes it more interpretable than the related KL divergence, which is unbounded and asymmetric.

### Population Stability Index (PSI)
PSI measures how much a variable's distribution has shifted between two datasets, typically a baseline (training) and a current (production) dataset. It is computed by binning the variable, comparing the fraction of observations in each bin across the two datasets, and summing a weighted log-ratio.

A common rule of thumb: PSI < 0.1 indicates no significant shift; PSI 0.1–0.25 indicates moderate shift warranting investigation; PSI > 0.25 indicates major shift requiring action. Fiddler supports PSI alongside JSD so users can choose the metric most familiar to their team.

### Embeddings
A vector embedding is a list of floating-point numbers (e.g., 384 numbers for `BAAI/bge-small-en-v1.5`) that encodes the semantic meaning of a piece of text. Similar meaning → similar vectors. The entire RAG retrieval system depends on this: at query time, the query is embedded, and the index returns the document chunks whose vectors are closest (measured by cosine similarity).

Choosing a local embedding model (running on CPU/GPU without an API call) rather than a hosted embedding API was a deliberate design choice: it eliminates per-query embedding costs for the ~323 documents in the corpus.

### Similarity Score (Retrieval Score)
When the vector index returns chunks, each chunk receives a **similarity score** between 0 and 1, reflecting how semantically close it is to the query embedding. In the Streamlit app, this score appears in the Sources expander as `[0.847]`, for example. Higher is better. A retrieval failure (like the 512-token data integrity case) often manifests as the system returning high-similarity scores for chunks that are topically adjacent but not directly relevant — the similarity metric captures surface-level semantic proximity, not logical relevance to the specific question.

### Tree Summarize (Response Mode)
After retrieving the top-k chunks, LlamaIndex assembles a final response. With `response_mode="tree_summarize"`, it recursively summarizes groups of retrieved chunks, then synthesizes a final answer from the intermediate summaries. This is better than simply concatenating chunks and asking the LLM to answer, because it handles cases where the combined context exceeds the LLM's context window gracefully, and it produces responses that integrate information across multiple sources rather than just quoting one.

### LLM-as-Judge
Using one LLM to evaluate the outputs of another (or the same) LLM. The judge is given a rubric, a question, and a response, and asked to score the response according to the rubric. Research has shown that capable LLMs (GPT-4, Claude Opus/Sonnet) correlate well with human expert ratings when the rubric is well-defined — making this a scalable alternative to human evaluation for development-time experimentation. The principal risk is that the judge may share the generator's blind spots, particularly when both are from the same model family.

In this project, Anthropic Claude (`claude-sonnet-4-5`) acts as both generator and judge. The prompt design mitigations — full rubric, chain-of-thought, multi-dimensional scoring — help reduce self-consistency bias.
