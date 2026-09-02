import hashlib
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from qa_bot_intelligent_assistant.config.settings import settings

def get_embedding_model() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.EMBEDDING_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )

def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )

def generate_chunk_id(chunk: Document) -> str:
    file_hash = chunk.metadata.get("file_hash", "unknown")
    page = chunk.metadata.get("page", 0)
    content_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
    return f"{file_hash}:{page}:{content_hash}"

def is_file_indexed(file_hash: str) -> bool:
    vectorstore = get_vectorstore()
    result = vectorstore.get(where={"file_hash": file_hash}, limit=1)
    return len(result.get("ids", [])) > 0

def upsert_documents(chunks: list[Document]) -> int:
    vectorstore = get_vectorstore()
    ids = [generate_chunk_id(chunk) for chunk in chunks]
    vectorstore.add_documents(chunks, ids=ids)
    return len(chunks)

def list_indexed_files() -> list[tuple[str, str]]:
    vectorstore = get_vectorstore()
    result = vectorstore.get(include=["metadatas"])
    seen = set()
    files = []
    for metadata in result.get("metadatas", []):
        file_hash = metadata.get("file_hash")
        if not file_hash or file_hash in seen:
            continue
        seen.add(file_hash)
        source = metadata.get("source")
        name = Path(source).name if source else file_hash
        files.append((file_hash, name))
    return files

def delete_file(file_hash: str) -> None:
    vectorstore = get_vectorstore()
    vectorstore.delete(where={"file_hash": file_hash})