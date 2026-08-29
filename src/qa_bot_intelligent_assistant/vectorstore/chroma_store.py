from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from config.settings import settings

def get_embedding_model():
    return OllamaEmbeddings(
        model=settings.EMBEDDING_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )

def get_vectorstore():
    return Chroma(
        collection_name=settings.COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )