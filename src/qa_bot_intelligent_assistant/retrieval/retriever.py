from langchain_core.vectorstores import VectorStoreRetriever
from qa_bot_intelligent_assistant.vectorstore.chroma_store import get_vectorstore
from qa_bot_intelligent_assistant.config.settings import settings

def get_retriever() -> VectorStoreRetriever:
    return get_vectorstore().as_retriever(search_kwargs={"k": settings.TOP_K})