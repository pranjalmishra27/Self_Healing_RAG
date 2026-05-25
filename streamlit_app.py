"""
Streamlit UI — Self-Healing RAG Pipeline Demo
Run: streamlit run streamlit_app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.set_page_config(
    page_title="Self-Healing RAG",
    page_icon="🔁",
    layout="wide",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .decision-APPROVE       { color: #22c55e; font-weight: bold; }
    .decision-RETRIEVE_AGAIN{ color: #f59e0b; font-weight: bold; }
    .decision-REWRITE_QUERY { color: #3b82f6; font-weight: bold; }
    .decision-FAIL          { color: #ef4444; font-weight: bold; }
    .source-card { background:#1e293b; border-radius:8px; padding:10px 14px;
                   margin-bottom:6px; font-size:0.85em; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔁 Self-Healing RAG Pipeline")
st.caption("An intelligent RAG system that verifies its own answers and retries when grounding is weak.")

# ── Sidebar — Ingestion ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Document Ingestion")
    ingest_path = st.text_input(
        "Directory path",
        value="./data/sample_docs",
        help="Local path to a folder of .txt, .pdf, or .md files",
    )
    if st.button("Ingest Documents", type="primary"):
        from src.ingestion.ingestor import DocumentIngestor
        with st.spinner("Ingesting documents…"):
            try:
                ing = DocumentIngestor()
                result = ing.ingest_directory(ingest_path)
                st.success(
                    f"✅ Ingested {result['documents_ingested']} documents "
                    f"→ {result['chunks_created']} chunks"
                )
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.divider()
    st.header("⚙️ Pipeline Settings")
    max_retries = st.slider("Max retries", 0, 5, 3)
    top_k = st.slider("Top-K retrieval", 1, 10, 5)

# ── Main — Q&A ─────────────────────────────────────────────────────────────────
question = st.text_area(
    "Your question",
    placeholder="e.g. What is the warranty policy for the ProDevice X1?",
    height=80,
)

run_btn = st.button("Ask 🚀", type="primary", disabled=not question.strip())

if run_btn and question.strip():
    from src.graph.workflow import RAGPipeline

    pipeline = RAGPipeline(max_retries=max_retries)

    with st.spinner("Running self-healing RAG pipeline…"):
        try:
            state = pipeline.run(question)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

    # ── Layout: 3 columns ──────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([3, 2, 2])

    # ── Answer panel ──────────────────────────────────────────────────────────
    with col1:
        decision = state.get("critic_decision", "?")
        dec_class = {
            "APPROVE": "decision-APPROVE",
            "RETRIEVE_AGAIN": "decision-RETRIEVE_AGAIN",
            "REWRITE_QUERY": "decision-REWRITE_QUERY",
        }.get(decision, "decision-FAIL")

        st.subheader("💬 Final Answer")
        st.info(state.get("final_response", "—"))

        st.markdown(
            f"**Critic decision:** <span class='{dec_class}'>{decision}</span><br>"
            f"**Critic reason:** {state.get('critic_reason', '—')}<br>"
            f"**Retries used:** {state.get('retry_count', 0)} / {max_retries}",
            unsafe_allow_html=True,
        )

        rq = state.get("rewritten_question", "")
        if rq and rq != question:
            st.caption(f"🔄 Rewritten query: _{rq}_")

    # ── Retrieved context panel ────────────────────────────────────────────────
    with col2:
        st.subheader("📄 Retrieved Sources")
        sources = state.get("source_metadata", [])
        if sources:
            for s in sources:
                page_str = f" · page {s['page']}" if s.get("page") else ""
                score_str = f" · score {s['score']:.3f}" if s.get("score") is not None else ""
                st.markdown(
                    f"<div class='source-card'>"
                    f"<strong>{s['filename']}</strong>{page_str}{score_str}<br>"
                    f"<small>{s['chunk_id']}</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No sources retrieved.")

        # Show chunk previews
        docs = state.get("retrieved_documents", [])
        if docs:
            with st.expander("Show chunk text"):
                for i, doc in enumerate(docs, 1):
                    st.text_area(
                        f"Chunk {i}: {doc.metadata.get('filename', '?')}",
                        value=doc.page_content[:400],
                        height=100,
                        disabled=True,
                        key=f"chunk_{i}",
                    )

    # ── Retry trace panel ──────────────────────────────────────────────────────
    with col3:
        st.subheader("🔁 Retry Trace")
        history = state.get("retry_history", [])
        if history:
            for r in history:
                dec = r["critic_decision"]
                color = {"APPROVE": "🟢", "RETRIEVE_AGAIN": "🟡", "REWRITE_QUERY": "🔵"}.get(dec, "🔴")
                with st.expander(f"{color} Attempt {r['retry_number']}: {dec}"):
                    st.write(f"**Query:** {r['rewritten_question']}")
                    st.write(f"**Critic:** {r['critic_reason']}")
                    st.write(f"**Answer preview:** {r['answer_preview']}")
        else:
            st.caption("No retries occurred — first attempt was approved.")
