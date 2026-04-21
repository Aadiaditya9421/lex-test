"""tests/test_nodes.py — Unit tests for LangGraph node functions."""

import pytest
import sqlite3
from langchain_core.documents import Document
from config import config
from graph.nodes import router_node, fallback_node, eval_node, grader_node, save_node


def test_router_returns_rag():
    state = {"query": "What is the IPC section for theft?"}
    result = router_node(state)
    assert result["route"] == "rag"


def test_router_returns_tool():
    state = {"query": "What is the current date and time?"}
    result = router_node(state)
    assert result["route"] == "tool"


def test_fallback_returns_correct_message():
    state = {"query": "Tell me about cars."}
    result = fallback_node(state)
    assert "Consult a qualified lawyer" in result["answer"]
    assert "sufficient information" in result["answer"]


def test_eval_low_score_triggers_fallback():
    """If should_fallback is already True, eval_node must preserve it."""
    state = {"route": "rag", "should_fallback": True, "context": "", "answer": ""}
    result = eval_node(state)
    assert result["should_fallback"] is True


def test_grader_filters_irrelevant_doc():
    """An irrelevant chunk (weather data) should be rejected for a legal query."""
    docs = [Document(
        page_content="The weather today is sunny with some clouds.",
        metadata={"source_file": "weather.txt", "title": "Weather Report"}
    )]
    state = {
        "query": "What is the punishment for cheating under IPC?",
        "documents": docs,
    }
    result = grader_node(state)
    assert len(result["relevant_docs"]) == 0
    assert result["should_fallback"] is True


def test_save_node_writes_to_db(tmp_path, monkeypatch):
    """save_node must write a row to SQLite."""
    db_path = str(tmp_path / "test_chat.db")
    monkeypatch.setattr(config, "sqlite_path", db_path)

    state = {
        "thread_id": "test_thread_001",
        "query": "test query about IPC",
        "answer": "test answer",
        "confidence": 0.9,
        "sources": [{"title": "IPC", "source": "01_ipc.txt"}],
        "route": "rag",
    }

    from graph.nodes import save_node
    save_node(state)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT query FROM chat_log WHERE thread_id=?", ("test_thread_001",))
    row = c.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "test query about IPC"
