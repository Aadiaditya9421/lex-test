"""graph/state.py — AgentState TypedDict for LexAssist AI."""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document


class AgentState(TypedDict):
    """
    Represents the state of the LexAssist AI LangGraph.
    Data is passed through nodes during execution.
    """
    # User Input
    query: str
    rewritten_query: str
    thread_id: str

    # Decisions & History
    route: str                                       # "rag", "tool", "chitchat"
    chat_history: Annotated[List[BaseMessage], operator.add]

    # RAG specific context
    documents: List[Document]
    relevant_docs: List[Document]
    context: str
    sources: List[Dict[str, Any]]

    # Execution Flags & Outputs
    tool_result: Optional[str]
    answer: str
    confidence: float
    should_fallback: bool
