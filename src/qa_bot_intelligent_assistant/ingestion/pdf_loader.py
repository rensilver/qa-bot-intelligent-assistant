from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_pdf(file_path: str | Path) -> list[Document]:
    loader = PyPDFLoader(file_path)
    return loader.load()

def load_pdfs_from_dir(directory: str | Path) -> list[Document]:
    documents: list[Document] = []
    for pdf_path in sorted(Path(directory).glob("*.pdf")):
        documents.extend(load_pdf(str(pdf_path)))
    return documents