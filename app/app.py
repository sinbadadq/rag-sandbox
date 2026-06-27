"""
app.py — Streamlit interface for the Fiddler Docs RAG Assistant.

Requires a built index in index_store/.  If none exists, run:
    python src/pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pipeline import INDEX_STORE_DIR, configure_settings, get_query_engine, load_index

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fiddler Docs Assistant",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 Fiddler Docs Assistant")
st.caption(
    "Ask anything about Fiddler AI's ML observability platform. "
    "Answers are grounded in the official Fiddler documentation."
)

# ── Index guard ──────────────────────────────────────────────────────────────

if not (INDEX_STORE_DIR / "docstore.json").exists():
    st.error(
        "**No index found.** "
        "Build it first by running:\n\n"
        "```\npython src/pipeline.py\n```"
    )
    st.stop()

# ── Load engine (cached) ─────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading index…")
def _get_engine():
    configure_settings()
    index = load_index(INDEX_STORE_DIR)
    return get_query_engine(index)


engine = _get_engine()

# ── Query UI ──────────────────────────────────────────────────────────────────

with st.form("query_form"):
    query = st.text_input(
        "Your question",
        placeholder="How does Fiddler detect data drift?",
    )
    submitted = st.form_submit_button("Ask", type="primary")

if submitted and query.strip():
    with st.spinner("Querying the docs…"):
        response = engine.query(query)

    st.markdown("### Answer")
    st.write(str(response))

    if response.source_nodes:
        with st.expander("Sources", expanded=False):
            for node in response.source_nodes:
                score_str = (
                    f"{node.score:.3f}" if node.score is not None else "n/a"
                )
                section = node.metadata.get("section", "—")
                file_name = node.metadata.get("file_name", "—")
                snippet = node.get_content()[:200].replace("\n", " ")
                st.markdown(
                    f"**[{score_str}]** `{section}/{file_name}`  \n"
                    f"_{snippet}…_"
                )
elif submitted:
    st.warning("Please enter a question.")
