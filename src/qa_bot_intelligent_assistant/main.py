from pathlib import Path
import gradio as gr
from qa_bot_intelligent_assistant.chains.rag_chain import build_rag_chain
from qa_bot_intelligent_assistant.ingestion.pdf_loader import load_pdf, compute_file_hash
from qa_bot_intelligent_assistant.ingestion.text_splitter import split_documents
from qa_bot_intelligent_assistant.vectorstore.chroma_store import upsert_documents, is_file_indexed, list_indexed_files, delete_file

rag_chain = build_rag_chain()

def respond(message: str, history: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    answer = rag_chain.invoke(message)
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

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Space Grotesk"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "Consolas", "monospace"],
).set(
    body_background_fill="#10161B",
    body_background_fill_dark="#10161B",
    body_text_color="#E9EDEE",
    body_text_color_dark="#E9EDEE",
    body_text_color_subdued="#8CA0AC",
    body_text_color_subdued_dark="#8CA0AC",
    background_fill_primary="#1A2229",
    background_fill_primary_dark="#1A2229",
    background_fill_secondary="#161D22",
    background_fill_secondary_dark="#161D22",
    block_background_fill="#1A2229",
    block_background_fill_dark="#1A2229",
    block_border_color="#2A343B",
    block_border_color_dark="#2A343B",
    block_label_background_fill="#1A2229",
    block_label_background_fill_dark="#1A2229",
    block_label_text_color="#8CA0AC",
    block_label_text_color_dark="#8CA0AC",
    block_title_text_color="#E9EDEE",
    block_title_text_color_dark="#E9EDEE",
    border_color_primary="#2A343B",
    border_color_primary_dark="#2A343B",
    input_background_fill="#161D22",
    input_background_fill_dark="#161D22",
    input_border_color="#2A343B",
    input_border_color_dark="#2A343B",
    color_accent_soft="#2A2419",
    color_accent_soft_dark="#2A2419",
    button_primary_background_fill="#E8A33D",
    button_primary_background_fill_dark="#E8A33D",
    button_primary_background_fill_hover="#F0B25A",
    button_primary_background_fill_hover_dark="#F0B25A",
    button_primary_text_color="#1A1206",
    button_primary_text_color_dark="#1A1206",
    button_secondary_background_fill="#232D34",
    button_secondary_background_fill_dark="#232D34",
    button_secondary_text_color="#E9EDEE",
    button_secondary_text_color_dark="#E9EDEE",
)

CSS = """
#app-shell { max-width: 1080px; margin: 0 auto !important; padding: 32px 20px 48px; }

.header-block { text-align: center; margin-bottom: 12px; }
.header-block .eyebrow {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 12px; letter-spacing: 0.18em; text-transform: uppercase;
    color: #E8A33D; margin: 0 0 6px;
}
.header-block .title-row {
    display: flex; align-items: center; justify-content: center; gap: 10px;
}
.header-block h1 { font-size: 28px; font-weight: 600; margin: 0; color: #E9EDEE; }
.header-block .subtitle {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 12px; color: #8CA0AC; margin: 6px 0 0;
}
.status-dot {
    width: 8px; height: 8px; border-radius: 50%; background: #E8A33D; flex-shrink: 0;
    box-shadow: 0 0 0 0 rgba(232, 163, 61, 0.45);
    animation: status-pulse 2.4s infinite;
}
@keyframes status-pulse {
    0% { box-shadow: 0 0 0 0 rgba(232, 163, 61, 0.45); }
    70% { box-shadow: 0 0 0 8px rgba(232, 163, 61, 0); }
    100% { box-shadow: 0 0 0 0 rgba(232, 163, 61, 0); }
}
@media (prefers-reduced-motion: reduce) {
    .status-dot { animation: none; }
}

.panel { border: 1px solid #2A343B; border-radius: 12px; padding: 18px !important; }
.panel-title.prose h3, .panel-title h3 {
    font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase;
    color: #8CA0AC; border-left: 3px solid #E8A33D; padding-left: 8px; margin: 4px 0 14px;
}
"""

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