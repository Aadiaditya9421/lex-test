"""
config.py — Centralised configuration for LexAssist AI.

Every tuneable constant lives here. All other modules import the
module-level `config` singleton — never hardcode values elsewhere.

Usage:
    from config import config

    model = config.llm_model
    config.validate()          # call once at app startup
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).parent.resolve()


@dataclass
class Config:
    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    max_tokens: int = 1024

    # ── Embeddings ────────────────────────────────────────────────────────────
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", str(_ROOT / "chroma_store"))
    )
    collection_name: str = "legal_docs"
    retrieval_k: int = 6
    mmr_fetch_k: int = 12
    mmr_lambda: float = 0.6

    # ── Text chunking ─────────────────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 120

    # ── Knowledge base ────────────────────────────────────────────────────────
    docs_dir: str = str(_ROOT / "knowledge_base" / "docs")

    # ── Evaluation ────────────────────────────────────────────────────────────
    faithfulness_threshold: float = 0.50

    # ── Memory & persistence ─────────────────────────────────────────────────
    sqlite_path: str = str(_ROOT / "chat_history.db")
    history_summary_threshold: int = 20

    # ── API keys ──────────────────────────────────────────────────────────────
    groq_api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )

    # ── App metadata ──────────────────────────────────────────────────────────
    app_name: str = "LexAssist AI"
    app_version: str = "1.0.0"
    app_description: str = "Indian legal information assistant — grounded, cited, hallucination-free."

    def __post_init__(self) -> None:
        self.temperature = float(self.temperature)
        self.faithfulness_threshold = float(self.faithfulness_threshold)
        self.retrieval_k = int(self.retrieval_k)
        self.mmr_fetch_k = int(self.mmr_fetch_k)
        self.chunk_size = int(self.chunk_size)
        self.chunk_overlap = int(self.chunk_overlap)

    def validate(self) -> None:
        """Validate all settings and raise with a clear message on any issue."""
        errors: list[str] = []

        if self.llm_provider == "groq" and not self.groq_api_key:
            errors.append(
                "GROQ_API_KEY is not set. "
                "Copy .env.example -> .env and paste your key."
            )
        elif self.llm_provider == "openai" and not self.openai_api_key:
            errors.append(
                "OPENAI_API_KEY is not set. "
                "Copy .env.example -> .env and paste your key."
            )

        if not (0.0 <= self.faithfulness_threshold <= 1.0):
            errors.append(
                f"faithfulness_threshold must be 0.0-1.0, "
                f"got {self.faithfulness_threshold}"
            )

        if self.retrieval_k > self.mmr_fetch_k:
            errors.append(
                f"retrieval_k ({self.retrieval_k}) must be <= "
                f"mmr_fetch_k ({self.mmr_fetch_k})"
            )

        if self.chunk_overlap >= self.chunk_size:
            errors.append(
                f"chunk_overlap ({self.chunk_overlap}) must be < "
                f"chunk_size ({self.chunk_size})"
            )

        if not Path(self.docs_dir).exists():
            errors.append(
                f"docs_dir not found: {self.docs_dir}. "
                f"Create knowledge_base/docs/ and add your .txt files."
            )

        if errors:
            bullet_list = "\n  * ".join(errors)
            raise EnvironmentError(
                f"LexAssist config validation failed:\n  * {bullet_list}"
            )

    def as_dict(self) -> dict:
        return {
            "llm_model": self.llm_model,
            "llm_provider": self.llm_provider,
            "temperature": self.temperature,
            "embed_model": self.embed_model,
            "chroma_persist_dir": self.chroma_persist_dir,
            "collection_name": self.collection_name,
            "retrieval_k": self.retrieval_k,
            "mmr_fetch_k": self.mmr_fetch_k,
            "mmr_lambda": self.mmr_lambda,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "faithfulness_threshold": self.faithfulness_threshold,
            "sqlite_path": self.sqlite_path,
            "history_summary_threshold": self.history_summary_threshold,
            "app_version": self.app_version,
            "groq_api_key_set": bool(self.groq_api_key),
            "openai_api_key_set": bool(self.openai_api_key),
        }

    def __repr__(self) -> str:
        return (
            f"Config(provider={self.llm_provider!r}, "
            f"model={self.llm_model!r}, "
            f"embed={self.embed_model!r}, "
            f"k={self.retrieval_k}, "
            f"threshold={self.faithfulness_threshold}, "
            f"api_key_set={bool(self.groq_api_key)})"
        )


config = Config()
