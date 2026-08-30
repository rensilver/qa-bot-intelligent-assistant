from unittest.mock import patch

from qa_bot_intelligent_assistant.config.settings import settings
from qa_bot_intelligent_assistant.retrieval.retriever import get_retriever

def test_get_retriever_uses_configured_top_k():
    with patch(
        "qa_bot_intelligent_assistant.retrieval.retriever.get_vectorstore"
    ) as mock_get_vectorstore:
        mock_vectorstore = mock_get_vectorstore.return_value

        get_retriever()

        mock_vectorstore.as_retriever.assert_called_once_with(
            search_kwargs={"k": settings.TOP_K}
        )
