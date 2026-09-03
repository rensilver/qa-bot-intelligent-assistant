# QA Bot Intelligent Assistant

A local, privacy-friendly Retrieval-Augmented Generation (RAG) chatbot that answers questions about your own PDF documents. Upload PDFs, index them into a vector database, and chat with an LLM that grounds its answers in the retrieved content — all running locally via [Ollama](https://ollama.com/).

## Features

- **PDF ingestion** — load one or more PDF files and split them into overlapping text chunks for indexing.
- **Vector search** — chunks are embedded and stored in [ChromaDB](https://www.trychroma.com/) for semantic similarity search.
- **RAG pipeline** — retrieved chunks are injected into a prompt so the LLM answers strictly from the indexed context (and admits when it doesn't know).
- **Local LLM & embeddings** — powered by [Ollama](https://ollama.com/) (default: `llama3.2` for generation, `nomic-embed-text` for embeddings). No external API calls, no data leaves your machine.
- **Web UI** — a [Gradio](https://www.gradio.app/) chat interface with drag-and-drop PDF upload and indexing progress feedback.
- **CLI ingestion script** — batch-index a whole directory of PDFs without opening the UI.
- **Idempotent indexing** — chunks are deduplicated via content hashing, so re-ingesting the same file won't create duplicate entries.

## Technologies

- **Python 3.11**
- **[LangChain](https://www.langchain.com/)** — orchestration of the RAG chain (`langchain`, `langchain-community`, `langchain-chroma`, `langchain-ollama`)
- **[Ollama](https://ollama.com/)** — local LLM inference and embeddings
- **[ChromaDB](https://www.trychroma.com/)** — vector store for document chunks
- **[Gradio](https://www.gradio.app/)** — web-based chat UI
- **[pypdf](https://pypi.org/project/pypdf/)** — PDF parsing
- **[uv](https://docs.astral.sh/uv/)** — dependency management and packaging
- **[Pydantic](https://docs.pydantic.dev/)** — configuration/data validation
- **Docker Compose** — runs Ollama and pulls the required models
- **pytest** — test suite

## Architecture

```
PDF files ──▶ loader ──▶ text splitter ──▶ embeddings ──▶ ChromaDB
                                                              │
User question ──▶ retriever ◀────────────────────────────────┘
                     │
                     ▼
              prompt + context ──▶ Ollama LLM ──▶ answer
```

Project layout:

```
src/qa_bot_intelligent_assistant/
├── main.py                # Gradio app entry point
├── config/settings.py      # Environment-based settings
├── ui/                     # Gradio ui theme
├── ingestion/               # PDF loading and text splitting
├── vectorstore/             # ChromaDB integration
├── retrieval/                # Retriever configuration
├── llm/                     # Ollama LLM client
└── chains/                  # RAG chain (prompt + retrieval + LLM)
scripts/ingest.py             # CLI to batch-index a directory of PDFs
```

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Docker](https://www.docker.com/) and Docker Compose (to run Ollama), **or** a local [Ollama](https://ollama.com/download) installation

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd qa-bot-intelligent-assistant
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Adjust the values in `.env` if needed:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `llama3.2:latest` | Model used for answer generation |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Model used for embeddings |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB persistence directory |
| `COLLECTION_NAME` | `documentos` | ChromaDB collection name |
| `CHUNK_SIZE` | `1000` | Text chunk size (characters) |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks (characters) |
| `TOP_K` | `4` | Number of chunks retrieved per query |

### 3. Start Ollama and pull the required models

```bash
docker compose up -d
```

This starts the Ollama server and automatically pulls `llama3.2:latest` and `nomic-embed-text`.

> Alternatively, if you have Ollama installed locally, run `ollama pull llama3.2` and `ollama pull nomic-embed-text` instead.

### 4. Install dependencies

```bash
uv sync
```

### 5. Run the app

```bash
uv run qa-bot-intelligent-assistant
```

The Gradio UI will be available at `http://127.0.0.1:7860`. From there you can upload PDFs, index them, and start asking questions.

### (Optional) Ingest PDFs via CLI

Instead of (or in addition to) uploading files through the UI, you can batch-index a directory:

```bash
uv run python scripts/ingest.py --source ./data/raw
```

`./data/raw` is the default source directory if `--source` is omitted.

## Running Tests

```bash
uv run pytest
```
