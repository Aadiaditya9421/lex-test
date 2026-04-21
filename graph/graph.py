"""graph/graph.py — LangGraph StateGraph wiring for LexAssist AI."""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from graph.state import AgentState
from graph.nodes import (
    memory_node, router_node, rewrite_node, retrieval_node,
    grader_node, tool_node, answer_node, eval_node, fallback_node, save_node,
)


def build_graph():
    """Compiles the core LangGraph structure for LexAssist AI."""
    workflow = StateGraph(AgentState)

    # 1. Add all nodes
    workflow.add_node("memory", memory_node)
    workflow.add_node("router", router_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("grader", grader_node)
    workflow.add_node("tool", tool_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("evaluate", eval_node)
    workflow.add_node("fallback", fallback_node)
    workflow.add_node("save", save_node)

    # 2. Entry point
    workflow.set_entry_point("memory")

    # 3. Fixed edges
    workflow.add_edge("memory", "router")
    workflow.add_edge("rewrite", "retrieval")
    workflow.add_edge("retrieval", "grader")
    workflow.add_edge("answer", "evaluate")
    workflow.add_edge("fallback", "save")
    workflow.add_edge("save", END)

    # 4. Conditional edges

    def route_decision(state: AgentState):
        route = state.get("route", "rag")
        if route == "rag":
            return "rewrite"
        elif route == "tool":
            return "tool"
        elif route == "chitchat":
            return "answer"
        return "rewrite"

    workflow.add_conditional_edges(
        "router",
        route_decision,
        {"rewrite": "rewrite", "tool": "tool", "answer": "answer"},
    )

    def grader_decision(state: AgentState):
        return "fallback" if state.get("should_fallback") else "answer"

    workflow.add_conditional_edges(
        "grader",
        grader_decision,
        {"fallback": "fallback", "answer": "answer"},
    )

    def tool_decision(state: AgentState):
        # If tool_result is None, fall back to RAG
        return "answer" if state.get("tool_result") else "rewrite"

    workflow.add_conditional_edges(
        "tool",
        tool_decision,
        {"answer": "answer", "rewrite": "rewrite"},
    )

    def eval_decision(state: AgentState):
        return "fallback" if state.get("should_fallback") else "save"

    workflow.add_conditional_edges(
        "evaluate",
        eval_decision,
        {"fallback": "fallback", "save": "save"},
    )

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


graph = build_graph()
