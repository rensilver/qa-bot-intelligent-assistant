from langchain_core.documents import Document

from qa_bot_intelligent_assistant.config.settings import settings
from qa_bot_intelligent_assistant.ingestion.text_splitter import (
    get_text_splitter,
    split_documents,
)

def test_get_text_splitter_uses_configured_chunk_settings():
    splitter = get_text_splitter()

    assert splitter._chunk_size == settings.CHUNK_SIZE
    assert splitter._chunk_overlap == settings.CHUNK_OVERLAP

def test_split_documents_splits_long_text_into_multiple_chunks():
    long_text = "word " * (settings.CHUNK_SIZE // len("word ") * 3)
    documents = [Document(page_content=long_text, metadata={"source": "a.pdf", "page": 0})]

    chunks = split_documents(documents)

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= settings.CHUNK_SIZE for chunk in chunks)
    assert all(chunk.metadata == {"source": "a.pdf", "page": 0} for chunk in chunks)

def test_split_documents_keeps_short_document_as_single_chunk():
    documents = [Document(page_content="short text", metadata={"source": "b.pdf", "page": 0})]

    chunks = split_documents(documents)

    assert len(chunks) == 1
    assert chunks[0].page_content == "short text"

def test_split_documents_with_no_documents_returns_empty_list():
    assert split_documents([]) == []
