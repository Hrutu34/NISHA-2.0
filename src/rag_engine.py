import re
import os
import chromadb
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def clean_chunk(text: str) -> str:
    # Remove HTML break tags
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    # Fix math trigger escaping
    text = text.replace('\\$', '$')
    return text

@st.cache_resource
def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_rag_chain():
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    embeddings = get_embedding_function()
    
    vectorstore = Chroma(
        collection_name="company_policies",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    groq_api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    
    # Enable streaming on ChatGroq
    llm = ChatGroq(
        model_name="openai/gpt-oss-20b",  # Updated active model ID
        groq_api_key=groq_api_key,
        temperature=0.0,
        streaming=True
    )

    system_prompt = (
        "You are NISHA, an internal HR and onboarding assistant for company employees.\n"
        "Provide direct, structured, and accurate guidance based strictly on the context below.\n\n"
        "### RULES:\n"
        "1. DO NOT use raw HTML tags (e.g., `<br>`, `<p>`). Use standard Markdown line breaks and bullet points.\n"
        "2. Do not enclose currency in LaTeX symbols ($3,000 should be plain text).\n"
        "3. Include structured headings, bullet points, and clearly state policy names and sections.\n"
        "4. If the context does not contain sufficient details to answer, state what is missing and direct the employee to HR at `hr-helpdesk@company.com`.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {os.path.basename(doc.metadata.get('source', 'Policy Doc'))}]\n{doc.page_content}"
            for doc in docs
        )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever