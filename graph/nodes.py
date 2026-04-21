"""
graph/nodes.py — All 10 LangGraph node functions for LexAssist AI.

BUG FIXES:
 1. Removed OpenAI import — Groq is the only provider.
 2. eval_node: fixed early-return logic so should_fallback propagates correctly
    for non-RAG routes (was returning stale state.get values).
 3. tool_node: wrapped all branches in try/except so tools never raise.
 4. answer_node: chitchat branch returns proper structure with empty sources.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from config import config
from graph.state import AgentState

log = logging.getLogger("nodes")

_vectorstore = None
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        from langchain_groq import ChatGroq
        _llm = ChatGroq(
            model=config.llm_model,
            temperature=config.temperature,
            groq_api_key=config.groq_api_key,
        )
    return _llm


def _get_vs():
    global _vectorstore
    if _vectorstore is None:
        from knowledge_base.ingest import get_vectorstore
        _vectorstore = get_vectorstore()
    return _vectorstore


def memory_node(state: AgentState) -> dict:
    """Summarize history if it gets too long, otherwise just pass."""
    history = state.get("chat_history", [])
    if len(history) > config.history_summary_threshold:
        return {"chat_history": history[-10:]}
    return {}


def router_node(state: AgentState) -> dict:
    """Classify the user's intent into rag, tool, or chitchat."""
    system = """You are a routing agent for Indian Legal Queries. Analyse the user input:
- If it asks for the current date/time, Limitation Act periods, or a quick lookup for IPC sections
  302 (murder), 376 (rape), 420 (cheating), or 498A (cruelty) — including when the user mentions
  the crime name instead of the section number → Output: 'tool'
- If it relates to detailed legal analysis, specific IPC/CrPC sections, constitutional rights,
  consumer law, RTI, labour law, or any other Indian legal topic → Output: 'rag'
- If it is a generic greeting or conversational nicety → Output: 'chitchat'
Output ONLY one word: rag, tool, or chitchat."""

    prompt = ChatPromptTemplate.from_messages([("system", system), ("user", "{query}")])
    chain = prompt | _get_llm()
    result = chain.invoke({"query": state["query"]})

    route = result.content.strip().lower()
    if route not in ["rag", "tool", "chitchat"]:
        route = "rag"
    return {"route": route}


def rewrite_node(state: AgentState) -> dict:
    """HyDE rewriting: Generate a hypothetical legal passage to improve retrieval."""
    system = (
        "Write a short, authoritative sounding legal paragraph from an Indian court document "
        "that would answer the following query. Do not explain, just write the paragraph."
    )
    prompt = ChatPromptTemplate.from_messages([("system", system), ("user", "{query}")])
    chain = prompt | _get_llm()
    result = chain.invoke({"query": state["query"]})
    return {"rewritten_query": result.content.strip()}


def retrieval_node(state: AgentState) -> dict:
    """Retrieve top contextual documents via MMR."""
    store = _get_vs()
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": config.retrieval_k,
            "fetch_k": config.mmr_fetch_k,
            "lambda_mult": config.mmr_lambda,
        },
    )
    query_to_use = state.get("rewritten_query") or state["query"]
    docs = retriever.invoke(query_to_use)
    return {"documents": docs}


def grader_node(state: AgentState) -> dict:
    """Filter irrelevant retrieved chunks, keeping strictly relevant ones."""
    system = """You are a strict Indian Legal Assessor. Given a query and a retrieved legal chunk:
If the chunk contains concepts, acts, or sections relevant to answering the query → output 'yes'.
If the chunk is irrelevant → output 'no'.
Output ONLY 'yes' or 'no'."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("user", "Query: {query}\n\nChunk: {chunk}")]
    )
    chain = prompt | _get_llm()

    relevant_docs = []
    sources = []

    for doc in state.get("documents", []):
        try:
            result = chain.invoke({"query": state["query"], "chunk": doc.page_content})
            grade = result.content.strip().lower()
        except Exception:
            grade = "no"

        if "yes" in grade:
            relevant_docs.append(doc)
            source_meta = {
                "source": doc.metadata.get("source_file", "Unknown"),
                "title": doc.metadata.get("title", "Unknown Section"),
            }
            if source_meta not in sources:
                sources.append(source_meta)

    should_fallback = len(relevant_docs) == 0
    context = "\n\n".join([d.page_content for d in relevant_docs])

    return {
        "relevant_docs": relevant_docs,
        "context": context,
        "sources": sources,
        "should_fallback": should_fallback,
    }


def tool_node(state: AgentState) -> dict:
    """Deterministic mini-tools for direct lookups without LLM extraction."""
    query = state["query"].lower()
    tool_result = None

    try:
        if "current date" in query or "today" in query or ("time" in query and "limitation" not in query):
            tool_result = f"The current date and time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (IST)."

        elif "limitation" in query and ("period" in query or "act" in query):
            tool_result = (
                "Key Limitation Periods under Limitation Act 1963:\n"
                "• Money due on contract: 3 years\n"
                "• Recovery of immoveable property: 12 years\n"
                "• Execution of a decree: 12 years\n"
                "• Compensation for personal injury: 3 years\n"
                "• Defamation suit: 1 year\n"
                "• Consumer complaint (CP Act 2019): 2 years"
            )

        elif "302" in query or "murder" in query:
            tool_result = (
                "IPC Section 302 — Punishment for Murder:\n"
                "Whoever commits murder shall be punished with death, or imprisonment for life, "
                "and shall also be liable to fine."
            )
        elif "376" in query or "rape" in query:
            tool_result = (
                "IPC Section 376 — Punishment for Rape:\n"
                "Rigorous imprisonment of not less than 10 years, which may extend to life "
                "imprisonment, and shall also be liable to fine. "
                "Aggravated rape (S.376D — gang rape): 20 years to life."
            )
        elif "420" in query or "cheating" in query:
            tool_result = (
                "IPC Section 420 — Cheating and dishonestly inducing delivery of property:\n"
                "Punishable with imprisonment of either description for up to 7 years, "
                "and shall also be liable to fine."
            )
        elif "498a" in query or "498-a" in query or "cruelty" in query:
            tool_result = (
                "IPC Section 498A — Husband or relative subjecting woman to cruelty:\n"
                "Punishable with imprisonment for up to 3 years, and shall also be liable to fine. "
                "Cognizable and non-bailable."
            )

    except Exception as e:
        log.error(f"tool_node exception: {e}")
        tool_result = None

    if tool_result is None:
        # Signal graph to fall back to RAG
        return {"tool_result": None}

    return {
        "tool_result": tool_result,
        "context": tool_result,
        "sources": [{"title": "Quick Lookup Tool", "source": "System Database"}],
    }


def answer_node(state: AgentState) -> dict:
    """Generates the grounded response strictly using the provided context."""
    # Chitchat path
    if state.get("route") == "chitchat":
        return {
            "answer": (
                "Hello! I am LexAssist AI, your Indian Legal Information Assistant. "
                "I can help you understand IPC sections, CrPC procedures, Constitutional rights, "
                "consumer protection, RTI, labour law, and more. How can I assist you today?\n\n"
                "Please note: I provide legal *information* only, not legal *advice*. "
                "Always consult a qualified advocate for your specific situation."
            ),
            "sources": [],
            "confidence": 1.0,
        }

    # Pass tool results directly into the context pipeline to generate detailed answers
    # instead of returning the raw tool string.

    system = """You are an expert Legal Assistant specialising in Indian Law.
Your goal is to provide a comprehensive, well-researched, and easy-to-understand explanation based ONLY on the provided context.
- Be highly descriptive, conversational, yet authoritative. Write like a professional legal researcher explaining the law to a client.
- Provide step-by-step explanations, breaking down complex legal topics and implications clearly.
- MUST EXPLICITLY CITE the exact Acts, Chapter names, Sections, and Articles mentioned in the context.
- Don't just list the rules; explain what they mean in practical terms based on the context.
- If the context does not contain the answer, do not attempt to guess. Respond EXACTLY with:
  'I do not have sufficient information to answer that based on the legal documents available to me.'
- ALWAYS end your response with a newline then: 'Consult a qualified lawyer for advice specific to your situation.'"""

    hist_string = ""
    if state.get("chat_history"):
        last_4 = state["chat_history"][-4:]
        hist_string = "\n".join([f"{msg.type}: {msg.content}" for msg in last_4])

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", "Chat History:\n{history}\n\nContext:\n{context}\n\nUser Query: {query}"),
    ])
    chain = prompt | _get_llm()

    try:
        result = chain.invoke({
            "history": hist_string,
            "context": state.get("context", ""),
            "query": state["query"],
        })
        return {"answer": result.content.strip()}
    except Exception as e:
        log.error(f"answer_node error: {e}")
        return {
            "answer": (
                "I encountered an error generating your response. "
                "Please try again.\n\nConsult a qualified lawyer for advice specific to your situation."
            )
        }


def eval_node(state: AgentState) -> dict:
    """Evaluate hallucination: verify the answer maps to retrieved context."""
    route = state.get("route", "rag")

    # For non-RAG routes or already-flagged fallbacks, propagate unchanged
    if route != "rag" or state.get("should_fallback", False):
        return {
            "confidence": 1.0 if route != "rag" else 0.0,
            "should_fallback": state.get("should_fallback", False),
        }

    system = """Given the context and the generated draft response, rate how faithfully
the response maps to the context.
Output ONLY a float between 0.0 and 1.0.
0.0 = completely hallucinated or irrelevant.
1.0 = perfectly grounded in context.
Return ONLY the float value — no other text."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", "Context:\n{context}\n\nDraft Answer:\n{answer}"),
    ])
    chain = prompt | _get_llm()

    try:
        result = chain.invoke({
            "context": state.get("context", ""),
            "answer": state.get("answer", ""),
        })
        content = result.content.strip()
        import re
        # Look for the first 0.0 to 1.0 or 0/1
        match = re.search(r'(0\.\d+|1\.0|1|0)', content)
        if match:
            score = float(match.group(1))
        else:
            score = 0.0
    except Exception:
        score = 0.0

    should_fallback = score < config.faithfulness_threshold

    return {"confidence": score, "should_fallback": should_fallback}


def fallback_node(state: AgentState) -> dict:
    """Overrides hallucinated or unsupported answers with a standard failure output."""
    return {
        "answer": (
            "I do not have sufficient information to answer that based on the "
            "legal documents available to me.\n\n"
            "Consult a qualified lawyer for advice specific to your situation."
        )
    }


def save_node(state: AgentState) -> dict:
    """Logs the final state into an SQLite tracking table for auditability."""
    try:
        db_path = config.sqlite_path
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS chat_log
               (id TEXT PRIMARY KEY, thread_id TEXT, query TEXT, answer TEXT,
                confidence REAL, sources TEXT, route TEXT, created_at TEXT)"""
        )
        log_id = str(uuid4())
        sources_json = json.dumps(state.get("sources", []))
        c.execute(
            "INSERT INTO chat_log VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                log_id,
                state.get("thread_id", ""),
                state.get("query", ""),
                state.get("answer", ""),
                state.get("confidence", 0.0),
                sources_json,
                state.get("route", ""),
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.error(f"save_node DB error: {e}")
    return {}
