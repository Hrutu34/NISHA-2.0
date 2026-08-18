import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def load_and_index_documents(data_dir: str = "./data"):
    # Load markdown and PDF documents
    loaders = [
        DirectoryLoader(data_dir, glob="**/*.md", loader_cls=TextLoader),
        DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader),
    ]
    
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    
    if not docs:
        raise ValueError(f"No documents found in directory: {data_dir}")

    # Chunk the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)

    # Use the free CPU-friendly HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Store them in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="company_policies",
        persist_directory="./chroma_db"
    )
    
    return vectorstore

if __name__ == "__main__":
    load_and_index_documents()