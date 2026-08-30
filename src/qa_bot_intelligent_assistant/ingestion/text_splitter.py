from langchain_text_splitters import RecursiveCharacterTextSplitter
from qa_bot_intelligent_assistant.config.settings import settings

def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

def split_documents(documents: list) -> list:
    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)
    return chunks