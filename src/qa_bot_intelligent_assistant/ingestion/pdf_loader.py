import hashlib
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def compute_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for bloco in iter(lambda: f.read(8192), b""):
            hasher.update(bloco)
    return hasher.hexdigest()

def load_pdf(file_path: str) -> list[Document]:
    file_hash = compute_file_hash(file_path)
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata["file_hash"] = file_hash
    return documents

def load_pdfs_from_dir(directory: str) -> list[Document]:
    documents: list[Document] = []
    for pdf_path in sorted(Path(directory).glob("*.pdf")):
        documents.extend(load_pdf(str(pdf_path)))
    return documents