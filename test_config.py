"""
tests/test_config.py — Unit tests for config.py.
Run: pytest tests/test_config.py -v
"""

import os
import pytest
from pathlib import Path
from config import Config


# ── Construction & defaults ───────────────────────────────────────────────────

def test_config_instantiates():
    c = Config()
    # BUG FIX: was asserting gpt-4o-mini, now uses groq model
    assert c.llm_model == "llama-3.3-70b-versatile"
    assert c.llm_provider == "groq"
    assert c.temperature == 0.0
    assert c.retrieval_k == 6
    assert c.mmr_fetch_k == 12
    assert c.faithfulness_threshold == 0.50
    assert c.chunk_size == 800
    assert c.chunk_overlap == 120
    assert c.history_summary_threshold == 20

def test_config_paths_are_absolute():
    """__post_init__ resolves paths relative to project root."""
    c = Config()
    assert Path(c.chroma_persist_dir).is_absolute()
    assert Path(c.sqlite_path).is_absolute()
    assert Path(c.docs_dir).is_absolute()

def test_config_repr_does_not_expose_api_key():
    c = Config()
    c.groq_api_key = "gsk-supersecret"
    r = repr(c)
    assert "gsk-supersecret" not in r
    assert "api_key_set=True" in r

def test_as_dict_does_not_expose_api_key():
    c = Config()
    c.groq_api_key = "gsk-supersecret"
    d = c.as_dict()
    assert "groq_api_key" not in d
    assert d["groq_api_key_set"] is True

def test_as_dict_contains_required_keys():
    c = Config()
    d = c.as_dict()
    for key in ["llm_model", "retrieval_k", "faithfulness_threshold",
                "chunk_size", "app_version"]:
        assert key in d

def test_post_init_coerces_types():
    """Fields passed as strings should be coerced to correct types."""
    c = Config.__new__(Config)
    object.__setattr__(c, "temperature", "0.0")
    object.__setattr__(c, "faithfulness_threshold", "0.5")
    object.__setattr__(c, "retrieval_k", "6")
    object.__setattr__(c, "mmr_fetch_k", "12")
    object.__setattr__(c, "chunk_size", "800")
    object.__setattr__(c, "chunk_overlap", "120")
    object.__setattr__(c, "llm_provider", "groq")
    object.__setattr__(c, "llm_model", "llama-3.3-70b-versatile")
    object.__setattr__(c, "embed_model", "sentence-transformers/all-MiniLM-L6-v2")
    object.__setattr__(c, "chroma_persist_dir", "./chroma_store")
    object.__setattr__(c, "collection_name", "legal_docs")
    object.__setattr__(c, "mmr_lambda", 0.6)
    object.__setattr__(c, "docs_dir", "./knowledge_base/docs")
    object.__setattr__(c, "sqlite_path", "./chat_history.db")
    object.__setattr__(c, "history_summary_threshold", 20)
    object.__setattr__(c, "groq_api_key", "")
    object.__setattr__(c, "openai_api_key", "")
    object.__setattr__(c, "app_name", "LexAssist AI")
    object.__setattr__(c, "app_version", "1.0.0")
    object.__setattr__(c, "app_description", "")
    c.__post_init__()
    assert isinstance(c.temperature, float)
    assert isinstance(c.retrieval_k, int)
    assert isinstance(c.faithfulness_threshold, float)


# ── Validation: passing cases ─────────────────────────────────────────────────

def test_validate_passes_with_valid_config(tmp_path):
    c = Config()
    c.groq_api_key = "gsk-test-key"
    c.docs_dir = str(tmp_path)
    c.validate()  # must not raise


# ── Validation: failure cases ─────────────────────────────────────────────────

def test_validate_raises_on_missing_api_key(tmp_path):
    c = Config()
    c.llm_provider = "groq"
    c.groq_api_key = ""
    c.docs_dir = str(tmp_path)
    with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
        c.validate()

def test_validate_raises_on_bad_threshold(tmp_path):
    c = Config()
    c.groq_api_key = "gsk-test"
    c.docs_dir = str(tmp_path)
    c.faithfulness_threshold = 1.5
    with pytest.raises(EnvironmentError, match="faithfulness_threshold"):
        c.validate()

def test_validate_raises_when_k_exceeds_fetch_k(tmp_path):
    c = Config()
    c.groq_api_key = "gsk-test"
    c.docs_dir = str(tmp_path)
    c.retrieval_k = 15
    c.mmr_fetch_k = 6
    with pytest.raises(EnvironmentError, match="retrieval_k"):
        c.validate()

def test_validate_raises_when_overlap_exceeds_chunk_size(tmp_path):
    c = Config()
    c.groq_api_key = "gsk-test"
    c.docs_dir = str(tmp_path)
    c.chunk_overlap = 900
    c.chunk_size = 800
    with pytest.raises(EnvironmentError, match="chunk_overlap"):
        c.validate()

def test_validate_raises_when_docs_dir_missing():
    c = Config()
    c.groq_api_key = "gsk-test"
    c.docs_dir = "/this/path/does/not/exist"
    with pytest.raises(EnvironmentError, match="docs_dir"):
        c.validate()

def test_validate_collects_multiple_errors(tmp_path):
    """All errors should appear together in one raise, not one at a time."""
    c = Config()
    c.llm_provider = "groq"
    c.groq_api_key = ""
    c.faithfulness_threshold = 2.0
    c.retrieval_k = 20
    c.mmr_fetch_k = 5
    c.docs_dir = str(tmp_path)
    with pytest.raises(EnvironmentError) as exc_info:
        c.validate()
    msg = str(exc_info.value)
    assert "GROQ_API_KEY" in msg
    assert "faithfulness_threshold" in msg
    assert "retrieval_k" in msg


# ── Singleton ─────────────────────────────────────────────────────────────────

def test_module_singleton_is_config_instance():
    from config import config
    assert isinstance(config, Config)
