from unittest.mock import patch

from langchain_core.documents import Document

from qa_bot_intelligent_assistant.chains.rag_chain import build_rag_chain, format_docs

def test_format_docs_joins_page_content_with_blank_line():
    docs = [Document(page_content="first"), Document(page_content="second")]

    assert format_docs(docs) == "first\n\nsecond"

def test_format_docs_with_no_docs_returns_empty_string():
    assert format_docs([]) == ""

def test_build_rag_chain_wires_retriever_and_llm():
    with (
        patch("qa_bot_intelligent_assistant.chains.rag_chain.get_retriever") as mock_get_retriever,
        patch("qa_bot_intelligent_assistant.chains.rag_chain.get_llm") as mock_get_llm,
    ):
        chain = build_rag_chain()

        mock_get_retriever.assert_called_once()
        mock_get_llm.assert_called_once()
        assert chain is not None
