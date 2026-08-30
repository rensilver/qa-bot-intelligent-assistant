from pathlib import Path
from langchain_core.documents import Document
from qa_bot_intelligent_assistant.ingestion.pdf_loader import load_pdf, load_pdfs_from_dir

def load_document(file_path: str | Path) -> list[Document]:
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return load_pdf(file_path)

    raise ValueError(f"File type not yet supported: {ext}")

def load_directory(directory: str | Path) -> list[Document]:
    return load_pdfs_from_dir(directory)