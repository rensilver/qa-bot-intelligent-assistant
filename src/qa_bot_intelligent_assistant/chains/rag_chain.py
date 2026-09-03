from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from qa_bot_intelligent_assistant.llm.ollama_client import get_llm
from qa_bot_intelligent_assistant.retrieval.retriever import get_retriever

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer the question using only the context below.\n"
        "If the answer doesn't fit the context, say that you don't know.\n\n"
        "Context:\n{context}",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("chat_history"),
    (
        "human",
        "Given the conversation above, rephrase the follow-up question below to be "
        "a standalone question that can be understood without the conversation.\n\n"
        "Follow-up question: {question}\nStandalone question:",
    ),
])

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain() -> Runnable[dict, str]:
    retriever = get_retriever()
    llm = get_llm()
    condense_question = CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()

    def resolve_search_query(inputs: dict) -> str:
        if inputs["chat_history"]:
            return condense_question.invoke(inputs)
        return inputs["question"]

    return (
        RunnablePassthrough.assign(
            context=RunnableLambda(resolve_search_query) | retriever | format_docs
        )
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )