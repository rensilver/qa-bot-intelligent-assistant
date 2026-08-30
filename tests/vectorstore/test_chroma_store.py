from unittest.mock import patch

from langchain_core.documents import Document

from qa_bot_intelligent_assistant.config.settings import settings
from qa_bot_intelligent_assistant.vectorstore.chroma_store import (
    generate_chunk_id,
    get_embedding_model,
    get_vectorstore,
    upsert_documents,
)

def test_generate_chunk_id_is_deterministic_for_same_content():
    chunk = Document(page_content="hello world", metadata={"source": "a.pdf", "page": 2})

    assert generate_chunk_id(chunk) == generate_chunk_id(chunk)

def test_generate_chunk_id_differs_for_different_content():
    chunk_a = Document(page_content="hello", metadata={"source": "a.pdf", "page": 0})
    chunk_b = Document(page_content="world", metadata={"source": "a.pdf", "page": 0})

    assert generate_chunk_id(chunk_a) != generate_chunk_id(chunk_b)

def test_generate_chunk_id_defaults_missing_metadata():
    chunk = Document(page_content="hello", metadata={})

    chunk_id = generate_chunk_id(chunk)

    assert chunk_id.startswith("unknown:0:")

def test_get_embedding_model_uses_configured_settings():
    with patch("qa_bot_intelligent_assistant.vectorstore.chroma_store.OllamaEmbeddings") as mock_cls:
        get_embedding_model()

        mock_cls.assert_called_once_with(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )

def test_get_vectorstore_uses_configured_settings():
    with (
        patch("qa_bot_intelligent_assistant.vectorstore.chroma_store.Chroma") as mock_chroma_cls,
        patch(
            "qa_bot_intelligent_assistant.vectorstore.chroma_store.get_embedding_model"
        ) as mock_get_embedding_model,
    ):
        mock_get_embedding_model.return_value = "fake-embedding-model"

        get_vectorstore()

        mock_chroma_cls.assert_called_once_with(
            collection_name=settings.COLLECTION_NAME,
            embedding_function="fake-embedding-model",
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )

def test_upsert_documents_adds_chunks_with_generated_ids():
    chunks = [
        Document(page_content="hello", metadata={"source": "a.pdf", "page": 0}),
        Document(page_content="world", metadata={"source": "a.pdf", "page": 1}),
    ]

    with patch(
        "qa_bot_intelligent_assistant.vectorstore.chroma_store.get_vectorstore"
    ) as mock_get_vectorstore:
        mock_vectorstore = mock_get_vectorstore.return_value

        count = upsert_documents(chunks)

        expected_ids = [generate_chunk_id(chunk) for chunk in chunks]
        mock_vectorstore.add_documents.assert_called_once_with(chunks, ids=expected_ids)
        assert count == 2
