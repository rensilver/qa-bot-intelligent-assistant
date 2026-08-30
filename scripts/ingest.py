import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qa_bot_intelligent_assistant.ingestion.loader import load_directory
from qa_bot_intelligent_assistant.ingestion.text_splitter import split_documents
from qa_bot_intelligent_assistant.vectorstore.chroma_store import upsert_documents
from qa_bot_intelligent_assistant.config.settings import settings

def ingest(source_dir: str) -> None:
    print(f"Reading documents from: {source_dir}")
    documents = load_directory(source_dir)

    if not documents:
        print("Document not found. Please check if the path is correct.")
        return

    print(f"{len(documents)} pages loaded. Splitting in chunks...")
    chunks = split_documents(documents)
    print(f"{len(chunks)} generated chunks. Indexing in ChromaDB...")

    upsert_documents(chunks)

    print(
        f"Ingestion completed. Collection '{settings.COLLECTION_NAME}'"
        f"persisted in '{settings.CHROMA_PERSIST_DIR}'."
    )

def main():
    parser = argparse.ArgumentParser(description="PDFs ingestion to ChromaDB")
    parser.add_argument(
        "--source",
        default="./data/raw",
        help="Diretório com os PDFs a serem indexados (default: ./data/raw)",
    )
    args = parser.parse_args()
    ingest(args.source)

if __name__ == "__main__":
    main()