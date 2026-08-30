from pathlib import Path
import gradio as gr
from src.qa_bot_intelligent_assistant.chains.rag_chain import build_rag_chain
from src.qa_bot_intelligent_assistant.ingestion.pdf_loader import load_pdf
from src.qa_bot_intelligent_assistant.ingestion.text_splitter import split_documents
from src.qa_bot_intelligent_assistant.vectorstore.chroma_store import upsert_documents

rag_chain = build_rag_chain()

def respond(message, history):
    answer = rag_chain.invoke(message)
    history = history + [(message, answer)]
    return "", history

def index_pdfs(files, progress=gr.Progress()):
    if not files:
        return "No file selected."

    total_files = len(files)
    total_chunks = 0

    for i, path in enumerate(files):
        name = Path(path).name

        progress(i / total_files, desc=f"Reading {name}...")
        documents = load_pdf(path)

        progress((i + 0.4) / total_files, desc=f"Splitting into chunks: {name}")
        chunks = split_documents(documents)

        progress((i + 0.7) / total_files, desc=f"Indexing: {name}")
        upsert_documents(chunks)
        total_chunks += len(chunks)

    progress(1.0, desc="Done")
    return f"{total_files} PDF(s) indexed successfully ({total_chunks} chunks added)."
    
with gr.Blocks(title="QA Intelligent Assistant - Llama 3.2 + RAG + ChromaDB") as demo:
    gr.Markdown("## QA Intelligent Assistant - Llama 3.2 + RAG + ChromaDB")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Documents")
            upload = gr.File(
                label="Upload one or more PDFs",
                file_types=[".pdf"],
                file_count="multiple",
                type="filepath"
            )
            index_button = gr.Button("Index PDFs")
            index_status = gr.Textbox(label="Indexing status", interactive=False)

    with gr.Column(scale=2):
        chatbot = gr.Chatbot(label="Conversation", height=450)
        message = gr.Textbox(
            label="Message",
            placeholder="Ask something about the indexed documents...",
        )
        send_button = gr.Button("Send")

    index_button.click(fn=index_pdfs, inputs=upload, outputs=index_status)
    send_button.click(fn=respond, inputs=[message, chatbot], outputs=[message, chatbot])
    message.submit(fn=respond, inputs=[message, chatbot], outputs=[message, chatbot])

demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)