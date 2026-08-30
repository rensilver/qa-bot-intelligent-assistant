from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qa_bot_intelligent_assistant.llm.ollama_client import get_llm
from qa_bot_intelligent_assistant.retrieval.retriever import get_retriever

PROMPT_TEMPLATE = """Answer the question using only the context below.
If the answer don't fit the context, say that you don't know.

Context:
{context}

Question: {question}
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain():
    retriever = get_retriever()
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )