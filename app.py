"""Streamlit chat UI.  Run:  streamlit run app.py"""
from __future__ import annotations

from dataclasses import asdict

import streamlit as st

import config
import ingest
from rag import RAGEngine, Source, cited_sources

st.set_page_config(page_title="Knowledge Base", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading models…")
def get_engine() -> RAGEngine:
    return RAGEngine()


try:
    engine = get_engine()
except Exception as e:  # most often: missing API key or missing provider package
    st.error(
        f"Couldn't start the assistant: {e}\n\n"
        "Check your `.env` (API key, `LLM_PROVIDER`) and that dependencies are installed."
    )
    st.stop()


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
st.session_state.setdefault("messages", [])  # {"role", "content", "sources", "searched"}
st.session_state.setdefault("uploader_key", 0)


def render_sources(sources: list[dict], cited: list[int], show_all: bool) -> None:
    """Show cited sources; optionally the other retrieved passages too."""
    shown = [s for s in sources if s["index"] in cited or show_all]
    if not shown:
        return
    label = f"Sources ({len(cited)} cited)" if cited else "Retrieved passages (none cited)"
    with st.expander(label, expanded=False):
        for s in shown:
            mark = "" if s["index"] in cited else " — retrieved, not cited"
            st.markdown(
                f"**[{s['index']}] {s['filename']}**, page {s['page']} "
                f"· relevance {s['score']:.2f}{mark}"
            )
            st.caption(s["text"])


# --------------------------------------------------------------------------- #
# Sidebar: documents + settings
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Documents")

    uploads = st.file_uploader(
        "Add PDFs",
        type="pdf",
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploads and st.button("Index uploaded files", type="primary", use_container_width=True):
        config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
        saved = []
        for up in uploads:
            dest = config.DOCS_DIR / up.name
            dest.write_bytes(up.getbuffer())
            saved.append(dest)

        bar = st.progress(0.0, text="Indexing…")
        problems = []

        def on_progress(i: int, n: int, r: ingest.IngestResult) -> None:
            bar.progress(i / n, text=f"{r.filename}: {r.status}")
            if r.status in ("error", "empty"):
                problems.append(f"{r.filename}: {r.message}")

        ingest.ingest_paths(saved, progress=on_progress)
        bar.empty()
        for p in problems:
            st.warning(p)
        if not problems:
            st.session_state.uploader_key += 1
            st.rerun()

    docs = ingest.list_documents()
    if not docs:
        st.info("No documents yet. Upload PDFs above, or run `python ingest.py`.")
    for d in docs:
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{d['source']}**  \n{d['pages']} pages · {d['chunks']} chunks")
        if c2.button("✕", key=f"del_{d['source']}", help=f"Remove {d['source']}"):
            ingest.delete_document(d["source"])
            st.rerun()

    st.divider()
    st.header("Settings")
    top_k = st.slider("Passages to retrieve", 2, 12, config.TOP_K)
    show_all = st.checkbox("Also show passages the answer didn't cite", value=False)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"LLM: {config.LLM_PROVIDER}/{config.LLM_MODEL}  \nEmbeddings: {config.EMBEDDING_MODEL}")


# --------------------------------------------------------------------------- #
# Main: chat
# --------------------------------------------------------------------------- #
st.title("Ask your documents")
st.caption("Answers come only from the indexed PDFs, with numbered citations you can open below each reply.")

if engine.chunk_count() == 0:
    st.info("The knowledge base is empty. Add PDFs in the sidebar to get started.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("searched"):
                st.caption(f"Searched for: {msg['searched']}")
            render_sources(msg.get("sources", []), msg.get("cited", []), show_all)

if prompt := st.chat_input("Ask a question about your documents…"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            retrieval = engine.retrieve(prompt, history, top_k=top_k)

        searched = (
            retrieval.standalone_question
            if retrieval.standalone_question.strip() != prompt.strip()
            else ""
        )
        if searched:
            st.caption(f"Searched for: {searched}")

        answer = st.write_stream(engine.stream(retrieval))
        cited = [s.index for s in cited_sources(answer, retrieval.sources)]
        sources = [asdict(s) for s in retrieval.sources]
        render_sources(sources, cited, show_all)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "cited": cited,
            "searched": searched,
        }
    )
