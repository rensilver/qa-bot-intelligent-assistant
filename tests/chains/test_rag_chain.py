from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

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

def test_rag_chain_without_history_retrieves_using_original_question():
    received_queries = []

    def fake_retrieve(query):
        received_queries.append(query)
        return [Document(page_content="ctx")]

    with (
        patch(
            "qa_bot_intelligent_assistant.chains.rag_chain.get_retriever",
            return_value=RunnableLambda(fake_retrieve),
        ),
        patch(
            "qa_bot_intelligent_assistant.chains.rag_chain.get_llm",
            return_value=FakeListChatModel(responses=["final answer"]),
        ),
    ):
        chain = build_rag_chain()
        result = chain.invoke({"question": "What is X?", "chat_history": []})

    assert received_queries == ["What is X?"]
    assert result == "final answer"

def test_rag_chain_with_history_condenses_question_before_retrieval():
    received_queries = []

    def fake_retrieve(query):
        received_queries.append(query)
        return [Document(page_content="ctx")]

    with (
        patch(
            "qa_bot_intelligent_assistant.chains.rag_chain.get_retriever",
            return_value=RunnableLambda(fake_retrieve),
        ),
        patch(
            "qa_bot_intelligent_assistant.chains.rag_chain.get_llm",
            return_value=FakeListChatModel(
                responses=["What does page 2 say about X?", "final answer"]
            ),
        ),
    ):
        chain = build_rag_chain()
        result = chain.invoke(
            {
                "question": "what about page 2?",
                "chat_history": [HumanMessage("What is X?"), AIMessage("X is ...")],
            }
        )

    assert received_queries == ["What does page 2 say about X?"]
    assert result == "final answer"
