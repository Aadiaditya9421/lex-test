# LexAssist AI ⚖️

> Agentic RAG system for Indian legal information — grounded, citation-backed, hallucination-free.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red.svg)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama3.3--70B-purple.svg)](https://console.groq.com)

---

## What it does

LexAssist answers questions about Indian law (IPC, CrPC, Constitution, RTI, Consumer Protection, Labour Law, IT Act, POCSO, Domestic Violence Act, Limitation Act) using a LangGraph StateGraph pipeline that retrieves relevant legal provisions, scores faithfulness, and refuses to answer when not confident.

**No hallucinated section numbers. Ever.**

---

## Architecture

```
User query
    │
    ▼
memory_node → router_node
                 │
         ┌───────┼───────┐
      rewrite   tool   answer(chitchat)
         │       │
      retrieve  answer
         │       │
       grade   evaluate
         │       │
       answer  ┌─┴──────┐
         │    save   fallback
      evaluate          │
         │             save
     ┌───┴───┐
   save   fallback
              │
            save
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent | LangGraph 0.2+ (StateGraph) |
| LLM | Llama 3.3 70B (Groq free tier) |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) |
| Vector DB | ChromaDB 0.5 |
| Memory | MemorySaver |
| Frontend | Streamlit 1.40 |
| Persistence | SQLite |

---

## Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/<your-username>/lexassist-ai.git
cd lexassist-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (free at https://console.groq.com)

# 5. Build the knowledge base (first time only — takes ~2 minutes)
python -m knowledge_base.ingest

# 6. Run the app
streamlit run app.py
```

---

## Project Structure

```
lexassist-ai/
├── app.py                        # Streamlit UI entry point
├── config.py                     # Centralised configuration
├── requirements.txt
├── .env.example                  # Template — copy to .env
│
├── graph/
│   ├── state.py                  # AgentState TypedDict
│   ├── nodes.py                  # All 10 LangGraph node functions
│   └── graph.py                  # StateGraph wiring + compilation
│
├── knowledge_base/
│   ├── ingest.py                 # Document loading, chunking, embedding
│   └── docs/                     # 12 plain-text legal source files
│       ├── 01_ipc_sections_1_to_120.txt
│       ├── 02_ipc_sections_121_to_300.txt
│       ├── 03_ipc_sections_301_to_511.txt
│       ├── 04_crpc_arrest_bail_trial.txt
│       ├── 05_constitution_fundamental_rights.txt
│       ├── 06_constitution_dpsp_amendments.txt
│       ├── 07_consumer_protection_act_2019.txt
│       ├── 08_rti_act_2005.txt
│       ├── 09_labour_law_pf_gratuity.txt
│       ├── 10_it_act_2000_cyber_offences.txt
│       ├── 11_domestic_violence_pocso.txt
│       └── 12_limitation_act_glossary.txt
│
├── tests/
│   ├── conftest.py               # Shared pytest fixtures
│   ├── test_nodes.py             # Unit tests for node functions
│   └── red_team_results.md       # Documented adversarial test cases
│
├── test_config.py                # Config unit tests
├── test_e2e.py                   # End-to-end smoke test
└── docs/                         # Submission PDF goes here
```

---

## Running Tests

```bash
pytest tests/ test_config.py -v
```

---

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set `app.py` as the entry point
4. Add `GROQ_API_KEY` in the Secrets panel
5. **Important:** Add a startup script or run `python -m knowledge_base.ingest` before the app starts — or commit a pre-built `chroma_store/` to the repo

**Note on ChromaDB at deploy time:** Streamlit Cloud doesn't persist files between deploys. Add `chroma_store/` to your repo (it's ~50MB for 12 docs) OR add a `@st.cache_resource` ingest call in `app.py` to build on first run.

---

## Evaluation Targets

| Metric | Target |
|--------|--------|
| Mean faithfulness score | ≥ 0.80 |
| Router accuracy | ≥ 90% |
| Fallback on OOD queries | 100% |
| Mean response latency | < 8s |

---

## Academic Context

Capstone project for Agentic AI Course.
**Student:** Aditya Singh | **Roll No.:** 23052212 | **Batch:** Agentic AI 2026

---

## Disclaimer

LexAssist provides legal *information* only, not legal *advice*.
Always consult a qualified advocate for your specific situation.
