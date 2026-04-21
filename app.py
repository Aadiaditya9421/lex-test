"""
app.py — Streamlit UI for LexAssist AI.

STREAMLIT CLOUD FIX:
  - Knowledge base is built inside @st.cache_resource (runs once per dyno).
  - ChromaDB and HuggingFace model cache to /tmp (survives within a session).
  - Secrets must be set in TOML format:
        GROQ_API_KEY = "gsk_..."
"""

import os
import streamlit as st

# ── Streamlit Cloud: redirect caches to /tmp ──────────────────────────────────
# Streamlit Cloud's home dir is read-only; /tmp is writable and persists for
# the lifetime of the running container. We must set these BEFORE importing
# anything that might load config.py or transformers.
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/tmp/st_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_cache")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/chroma_store")

import uuid
from graph.graph import build_graph

st.set_page_config(
    page_title="LexAssist AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Streamlit Cloud caching was moved to top of file ──────────────────────────


# ── Cache expensive resources ─────────────────────────────────────────────────

@st.cache_resource(show_spinner="Building knowledge base… (first run, ~60 s)")
def init_knowledge_base():
    """
    Build or load ChromaDB vector store — runs ONCE per container lifecycle.
    On Streamlit Cloud the filesystem is ephemeral, so this will re-run on
    every cold start, but @st.cache_resource prevents it re-running on every
    page interaction.
    """
    from knowledge_base.ingest import build_vectorstore
    build_vectorstore()


@st.cache_resource(show_spinner="Loading AI graph…")
def load_graph():
    """Load and compile the LangGraph — called once, cached for session."""
    return build_graph()


# ── Session state ─────────────────────────────────────────────────────────────

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


def reset_session():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.pending_query = None


# ── Initialise on every page load (cached, so only runs once) ─────────────────
init_knowledge_base()
graph = load_graph()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚖️ LexAssist AI")
    st.markdown(
        "**Hallucination-free Indian legal information.**\n\n"
        "Citation-backed answers from IPC, CrPC, Constitution, RTI, "
        "Consumer Protection Act, Labour Law, IT Act, and more."
    )
    st.divider()
    st.text(f"Session: {st.session_state.thread_id[:8]}...")
    st.button("🔄 New Conversation", on_click=reset_session, type="primary")
    st.divider()

    st.markdown("### Example Questions")

    def set_query(q):
        st.session_state.pending_query = q

    examples = [
        "What is the IPC punishment for murder?",
        "Explain the right to bail under CrPC.",
        "What are my Fundamental Rights under Article 21?",
        "How do I file an RTI application?",
        "What is the gratuity calculation formula?",
        "What is the current date?",
    ]
    for ex in examples:
        st.button(ex, on_click=set_query, args=(ex,), use_container_width=True)

    st.divider()
    st.caption("⚠️ Legal information only — not legal advice. Consult a qualified advocate for your situation.")


# ── Main area ─────────────────────────────────────────────────────────────────

st.header("⚖️ LexAssist AI — Indian Legal Assistant", divider="grey")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "metrics" in msg:
            m = msg["metrics"]
            cols = st.columns(3)
            cols[0].metric("Route", m.get("route", "—"))
            cols[1].metric("Faithfulness", f"{m.get('confidence', 0.0):.2f}")
            cols[2].metric("Sources", str(m.get("source_count", 0)))
            if m.get("sources"):
                with st.expander("📄 View Sources"):
                    for s in m["sources"]:
                        st.markdown(f"- **{s.get('title', 'Document')}** — `{s.get('source', '')}`")


# ── Handle input ──────────────────────────────────────────────────────────────

user_input = st.chat_input("Ask a legal question...")
pending = st.session_state.pending_query

if pending:
    user_input = pending
    st.session_state.pending_query = None

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Analysing legal context..."):
            try:
                initial_state = {"query": user_input}
                config_dict = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = graph.invoke(initial_state, config_dict)

                answer = result.get("answer", "I encountered an error retrieving the response.")
                route = result.get("route", "N/A")
                confidence = result.get("confidence", 0.0)
                sources = result.get("sources", [])

                st.markdown(answer)

                cols = st.columns(3)
                cols[0].metric("Route", route)
                cols[1].metric("Faithfulness", f"{confidence:.2f}")
                cols[2].metric("Sources", str(len(sources)))

                if sources:
                    with st.expander("📄 View Sources"):
                        for s in sources:
                            st.markdown(f"- **{s.get('title', 'Document')}** — `{s.get('source', '')}`")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metrics": {
                        "route": route,
                        "confidence": confidence,
                        "source_count": len(sources),
                        "sources": sources,
                    },
                })

            except Exception as e:
                err_msg = f"⚠️ Error: {str(e)}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
