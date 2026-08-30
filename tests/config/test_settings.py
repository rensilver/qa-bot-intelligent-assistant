import importlib

import pytest

from qa_bot_intelligent_assistant.config import settings as settings_module

@pytest.fixture(autouse=True)
def reset_settings_module():
    yield
    importlib.reload(settings_module)

def test_default_settings_when_no_env_vars(monkeypatch):
    for var in (
        "OLLAMA_BASE_URL",
        "LLM_MODEL",
        "EMBEDDING_MODEL",
        "CHROMA_PERSIST_DIR",
        "COLLECTION_NAME",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "TOP_K",
    ):
        monkeypatch.delenv(var, raising=False)

    reloaded = importlib.reload(settings_module)

    assert reloaded.settings.OLLAMA_BASE_URL == "http://localhost:11434"
    assert reloaded.settings.LLM_MODEL == "llama3.2:latest"
    assert reloaded.settings.EMBEDDING_MODEL == "nomic-embed-text"
    assert reloaded.settings.CHROMA_PERSIST_DIR == "./data/chroma_db"
    assert reloaded.settings.COLLECTION_NAME == "documentos"
    assert reloaded.settings.CHUNK_SIZE == 1000
    assert reloaded.settings.CHUNK_OVERLAP == 50
    assert reloaded.settings.TOP_K == 4

def test_settings_read_from_env_vars(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.internal:9999")
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "25")
    monkeypatch.setenv("TOP_K", "8")

    reloaded = importlib.reload(settings_module)

    assert reloaded.settings.OLLAMA_BASE_URL == "http://ollama.internal:9999"
    assert reloaded.settings.LLM_MODEL == "custom-model"
    assert reloaded.settings.CHUNK_SIZE == 500
    assert reloaded.settings.CHUNK_OVERLAP == 25
    assert reloaded.settings.TOP_K == 8
