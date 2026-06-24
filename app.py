"""
ScholarGPT — Streamlit UI.

Layout:
  Sidebar  : PDF upload + document list + controls
  Main area: Multi-turn chat with expandable source citations
"""
import logging

import streamlit as st

from src.config import UPLOAD_DIR
from src.llm import get_active_backend, stream_ollama_answer
from src.rag_pipeline import RAGPipeline
from src.retriever import retrieve

logging.basicConfig(level=logging.INFO)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ScholarGPT",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ──────────────────────────────────────────────────────────
# RAGPipeline is created once per browser session and cached in session_state.
# This means the VectorStore (and loaded embedding model) survive reruns.
if "pipeline" not in st.session_state:
    with st.spinner("Initialising RAG pipeline…"):
        st.session_state.pipeline = RAGPipeline()

if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []

pipeline: RAGPipeline = st.session_state.pipeline

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 ScholarGPT")
    st.caption("Upload research PDFs · Ask questions · Get cited answers")

    backend = get_active_backend()
    if backend == "ollama":
        st.info("LLM: Ollama (local)", icon="🦙")
    else:
        st.warning(
            "Ollama not detected. Start Ollama (`ollama serve`) to enable "
            "generated answers. Retrieval + citations still work.",
            icon="⚠️",
        )

    st.divider()

    # ── Upload ──────────────────────────────────────────────────────────────
    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Select one or more PDF files",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for f in uploaded_files:
            dest = UPLOAD_DIR / f.name
            dest.write_bytes(f.read())
            with st.spinner(f"Ingesting {f.name}…"):
                try:
                    info = pipeline.ingest_pdf(dest)
                    st.success(
                        f"✅ **{info['filename']}**  \n"
                        f"{info['pages']} pages → {info['chunks']} chunks"
                    )
                except Exception as exc:
                    st.error(f"Failed to ingest {f.name}: {exc}")

    st.divider()

    # ── Document list ───────────────────────────────────────────────────────
    docs = pipeline.list_documents()
    st.subheader(f"Documents ({len(docs)})")

    if docs:
        for doc in docs:
            col_name, col_btn = st.columns([5, 1])
            col_name.markdown(f"`{doc}`")
            if col_btn.button("🗑", key=f"del_{doc}", help=f"Remove {doc}"):
                removed = pipeline.delete_document(doc)
                st.toast(f"Removed {removed} chunks for {doc}")
                st.rerun()
    else:
        st.info("No documents uploaded yet.")

    st.divider()

    # ── Chat controls ───────────────────────────────────────────────────────
    if st.button("🗑️  Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.caption(
        f"Model: `{st.session_state.pipeline.store.encoder.get_sentence_embedding_dimension()}`-dim "
        f"embeddings  |  Top-K: retrieve then rerank"
    )

# ── Main area ──────────────────────────────────────────────────────────────
st.title("💬 ScholarGPT — Research Q&A")

if not docs:
    st.info(
        "👈  Upload a PDF in the sidebar to get started.  "
        "Then ask any question about its contents."
    )

# Render chat history
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander(f"📖 Sources ({len(turn['sources'])})"):
                for i, src in enumerate(turn["sources"], 1):
                    st.markdown(
                        f"**[{i}]** `{src['filename']}` — Page **{src['page']}** "
                        f"(relevance: {src['score']})\n\n> {src['excerpt']}"
                    )

# Chat input
if query := st.chat_input("Ask a question about your uploaded documents…"):
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        history = st.session_state.chat_history[:-1]

        if get_active_backend() == "ollama" and pipeline.store.count() > 0:
            chunks = retrieve(pipeline.store, query)
            st.caption("Retrieving context and calling Ollama — first token may take 30-90s on CPU...")
            full_answer = st.write_stream(
                stream_ollama_answer(query, chunks, chat_history=history)
            )
            sources = [
                {
                    "filename": c["filename"],
                    "page": c["page"],
                    "excerpt": c["text"][:300] + ("..." if len(c["text"]) > 300 else ""),
                    "score": round(c.get("rerank_score", c.get("score", 0.0)), 3),
                }
                for c in chunks
            ]
        else:
            with st.spinner("Searching documents and generating answer…"):
                result = pipeline.answer(query, chat_history=history)
            full_answer = result["answer"]
            sources = result["sources"]
            st.markdown(full_answer)

        if sources:
            with st.expander(f"📖 Sources ({len(sources)})"):
                for i, src in enumerate(sources, 1):
                    st.markdown(
                        f"**[{i}]** `{src['filename']}` — Page **{src['page']}** "
                        f"(relevance: {src['score']})\n\n> {src['excerpt']}"
                    )
        elif pipeline.store.count() > 0:
            st.caption("No relevant passages found for this query.")

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": full_answer,
            "sources": sources,
        }
    )
