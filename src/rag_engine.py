import os
import chromadb
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

@st.cache_resource
def get_embedding_function():
    # Runs locally on free CPU (no API key needed)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_rag_chain():
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name="company_policies",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Retrieves free key from Streamlit Secrets or Environment
    groq_api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    
    llm = ChatGroq(
        model_name="llama3-8b-8192",  # Updated stable model ID
        groq_api_key=groq_api_key,
        temperature=0.0
    )

    system_prompt = (
        "You are NISHA, an intelligent company policy and onboarding assistant. "
        "Use the policy context below to give accurate, empathetic, and clear guidance. "
        "If the answer is not in the context, refer the employee to HR.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(
            f"[{doc.metadata.get('source', 'Policy Doc')}]:\n{doc.page_content}"
            for doc in docs
        )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever