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

def clean_markdown_output(text: str) -> str:
    # Convert <br> or <br/> tags to simple line breaks
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    # Fix broken currency math triggers
    text = text.replace('\\$', '$')
    return text.strip()

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
    
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0.0
    )

    system_prompt = (
        "You are NISHA, an intelligent and empathetic internal HR and policy assistant for company employees and new joiners.\n"
        "Your task is to provide structured, accurate, and easily readable answers strictly grounded in the context provided below.\n\n"
        "### FORMATTING RULES:\n"
        "1. DO NOT use raw HTML tags like `<br>`, `<p>`, or `<span>`. Use standard Markdown bullet points and line breaks instead.\n"
        "2. When mentioning currency amounts (e.g., $3,000), write them clearly without LaTeX math formulas.\n"
        "3. When answering, structure your response with clear subheadings, bullet points, and an optional 'Next Steps' or 'Important Notes' section.\n"
        "4. Always mention the exact policy name and section number at the end of the explanation under a '📌 Citations' section.\n"
        "5. If the provided context does not contain enough information to answer definitively, clearly state what is missing and direct the employee to contact HR at `hr-helpdesk@company.com`.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(
            f"[Source Document: {doc.metadata.get('source', 'Company Policy')}]\n{doc.page_content}" 
            for doc in docs
        )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        | clean_markdown_output
    )

    return rag_chain, retriever