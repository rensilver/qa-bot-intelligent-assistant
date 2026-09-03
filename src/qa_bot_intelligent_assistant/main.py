from pathlib import Path
import gradio as gr
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from qa_bot_intelligent_assistant.chains.rag_chain import build_rag_chain
from qa_bot_intelligent_assistant.ingestion.pdf_loader import load_pdf, compute_file_hash
from qa_bot_intelligent_assistant.ingestion.text_splitter import split_documents
from qa_bot_intelligent_assistant.vectorstore.chroma_store import upsert_documents, is_file_indexed, list_indexed_files, delete_file
from qa_bot_intelligent_assistant.ui.theme import THEME, CSS

rag_chain = build_rag_chain()

def to_chat_history(history: list[dict[str, str]]) -> list[BaseMessage]:
    return [
        HumanMessage(turn["content"]) if turn["role"] == "user" else AIMessage(turn["content"])
        for turn in history
    ]

def respond(message: str, history: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    answer = rag_chain.invoke({"question": message, "chat_history": to_chat_history(history)})
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return "", history

def index_pdfs(files: list[str] | None, progress: gr.Progress = gr.Progress()) -> str:
    if not files:
        return "No file selected."

    total_files = len(files)
    total_chunks = 0
    processed, skipped = [], []

    for i, path in enumerate(files):
        name = Path(path).name
        progress(i / total_files, desc=f"Checking {name}...")

        file_hash = compute_file_hash(path)
        if is_file_indexed(file_hash):
            skipped.append(name)
            continue

        progress((i + 0.3) / total_files, desc=f"Reading {name}...")
        documents = load_pdf(path)

        progress((i + 0.6) / total_files, desc=f"Indexing: {name}")
        chunks = split_documents(documents)
        upsert_documents(chunks)

        total_chunks += len(chunks)
        processed.append(name)

    progress(1.0, desc="Done")

    parts = []
    if processed:
        parts.append(f"Indexed: {', '.join(processed)} ({total_chunks} chunks)")
    if skipped:
        parts.append(f"Already indexed, skipped: {', '.join(skipped)}")
    status = " | ".join(parts) if parts else "Nothing to do"

    return status

def remove_indexed_file(file_hash: str) -> list[tuple[str, str]]:
    delete_file(file_hash)
    return list_indexed_files()

with gr.Blocks(title="QA Intelligent Assistant - Llama 3.2 + RAG + ChromaDB") as demo:
    with gr.Column(elem_id="app-shell"):
        gr.HTML(
            """
            <div class="header-block">
                <p class="eyebrow">Local RAG Console</p>
                <div class="title-row">
                    <h1>QA Intelligent Assistant</h1>
                    <span class="status-dot" title="Online"></span>
                </div>
                <p class="subtitle">Llama 3.2 &middot; RAG &middot; ChromaDB</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["panel"]):
                gr.Markdown("### Documents", elem_classes=["panel-title"])
                upload = gr.File(
                    label="Upload one or more PDFs",
                    file_types=[".pdf"],
                    file_count="multiple",
                    type="filepath"
                )
                index_button = gr.Button("Index PDFs", variant="primary")
                status_output = gr.Textbox(label="Indexing status", interactive=False)

                gr.Markdown("### Indexed Documents", elem_classes=["panel-title"])
                indexed_files_state = gr.State([])

                @gr.render(inputs=indexed_files_state)
                def render_indexed_files(files):
                    if not files:
                        gr.Markdown("No documents indexed yet.")
                        return
                    for file_hash, name in files:
                        with gr.Row():
                            gr.Markdown(name)
                            delete_button = gr.Button("✕", scale=0, min_width=40)
                            delete_button.click(
                                fn=lambda fh=file_hash: remove_indexed_file(fh),
                                outputs=indexed_files_state,
                            )

            with gr.Column(scale=2, elem_classes=["panel"]):
                gr.Markdown("### Conversation", elem_classes=["panel-title"])
                chatbot = gr.Chatbot(label="Conversation", height=450)
                message = gr.Textbox(
                    label="Message",
                    placeholder="Ask something about the indexed documents...",
                )
                send_button = gr.Button("Send", variant="primary")

    index_button.click(fn=index_pdfs, inputs=upload, outputs=status_output).then(
        fn=lambda: None, outputs=upload
    ).then(
        fn=list_indexed_files, outputs=indexed_files_state
    )
    send_button.click(fn=respond, inputs=[message, chatbot], outputs=[message, chatbot])
    message.submit(fn=respond, inputs=[message, chatbot], outputs=[message, chatbot])

    demo.load(fn=list_indexed_files, outputs=indexed_files_state)

demo.queue()

def main() -> None:
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=THEME, css=CSS)

if __name__ == "__main__":
    main()