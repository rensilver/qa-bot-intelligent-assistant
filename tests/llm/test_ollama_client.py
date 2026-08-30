from unittest.mock import patch

from qa_bot_intelligent_assistant.config.settings import settings
from qa_bot_intelligent_assistant.llm.ollama_client import get_llm

def test_get_llm_uses_configured_settings():
    with patch("qa_bot_intelligent_assistant.llm.ollama_client.ChatOllama") as mock_cls:
        get_llm()

        mock_cls.assert_called_once_with(
            model=settings.LLM_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.2,
        )
