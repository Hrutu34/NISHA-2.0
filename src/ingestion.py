import os
import glob
import unicodedata
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "ft",
    "\ufb06": "st",
    "\ufffd": "",  # Strip raw replacement characters
}

def clean_extracted_text(text: str) -> str:
    for lig, rep in LIGATURE_MAP.items():
        text = text.replace(lig, rep)
    # Normalize unicode representations
    text = unicodedata.normalize("NFKD", text)
    # Strip non-printable/control characters
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\r\t ")
    return text

def load_documents_from_directory(directory_path: str = "./data/sample_policies") -> List[Document]:
    documents = []
    pdf_files = glob.glob(f"{directory_path}/*.pdf")
    md_files = glob.glob(f"{directory_path}/*.md")
    
    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        for doc in docs:
            doc.page_content = clean_extracted_text(doc.page_content)
        documents.extend(docs)
        
    for md_path in md_files:
        loader = TextLoader(md_path, encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.page_content = clean_extracted_text(doc.page_content)
        documents.extend(docs)
        
    return documents

def load_and_index_documents():
    docs = load_documents_from_directory()
    if not docs:
        print("No documents found to index.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="company_policies",
        persist_directory="./chroma_db"
    )
    return vectorstore

if __name__ == "__main__":
    load_and_index_documents()