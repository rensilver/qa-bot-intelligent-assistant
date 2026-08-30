import hashlib
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from qa_bot_intelligent_assistant.config.settings import settings

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

def generate_chunk_id(chunk: Document) -> str:
    source = chunk.metadata.get("source", "unknown")
    page = chunk.metadata.get("page", 0)
    content_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
    return f"{source}:{page}:{content_hash}"

def upsert_documents(chunks: list[Document]) -> int:
    vectorstore = get_vectorstore()
    ids = [generate_chunk_id(chunk) for chunk in chunks]
    vectorstore.add_documents(chunks, ids=ids)
    return len(chunks)