from langchain_ollama import ChatOllama
from config.settings import settings

def get_llm():
    return ChatOllama(
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.2,
    )