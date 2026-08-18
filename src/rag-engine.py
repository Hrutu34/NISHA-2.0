from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.config import settings

def get_rag_chain():
    embeddings = OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url
    )
    
    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_directory
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # Fully local LLM via Ollama
    llm = ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.0
    )

    system_prompt = (
        "You are NISHA, an intelligent internal assistant for company onboarding and policy guidance. "
        "Use the retrieved company documentation below to provide direct, helpful, and empathetic answers. "
        "If you do not know the answer or if the policy is not documented, clearly direct the user to HR.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'Policy Doc')}]\n{doc.page_content}" 
            for doc in docs
        )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever