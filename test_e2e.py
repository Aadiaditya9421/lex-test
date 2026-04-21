"""
test_e2e.py — End-to-end smoke test for LexAssist AI.

Run: python test_e2e.py
Requires: GROQ_API_KEY in .env and ChromaDB built (python -m knowledge_base.ingest)
"""

from graph.graph import graph
import uuid


def run():
    print("=== LexAssist AI End-to-End Smoke Test ===\n")
    thread_id = str(uuid.uuid4())
    config_dict = {"configurable": {"thread_id": thread_id}}

    queries = [
        {"desc": "IPC Question (RAG Route)", "query": "What is the punishment for cheating under the IPC if someone induces the delivery of property?"},
        {"desc": "Date Question (Tool Route)", "query": "What is the current date today?"},
        {"desc": "Greeting (ChitChat Route)", "query": "Hi there! I need some legal help."},
        {"desc": "Out-of-Scope (Fallback Route)", "query": "Can you explain the US Constitution's second amendment?"},
        {"desc": "Multi-turn Follow-up", "query": "What if someone only attempted to cheat but didn't succeed? How does IPC handle attempts?"},
    ]

    passed = 0
    for q in queries:
        print(f"\nTesting: {q['desc']}")
        print(f"Query: {q['query']}")
        try:
            result = graph.invoke({"query": q["query"]}, config_dict)
            print(f"Route: {result.get('route', 'N/A')}")
            print(f"Confidence: {result.get('confidence', 'N/A')}")
            print(f"Fallback: {result.get('should_fallback', False)}")
            answer = result.get("answer", "")
            print(f"Answer (first 200 chars): {answer[:200]}...")
            passed += 1
        except Exception as e:
            print(f"ERROR: {e}")
        print("-" * 60)

    print(f"\nSmoke test complete: {passed}/{len(queries)} queries processed.")


if __name__ == "__main__":
    run()
