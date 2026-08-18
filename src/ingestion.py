import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from src.config import settings

def load_and_index_documents(data_dir: str = "./data"):
    loaders = [
        DirectoryLoader(data_dir, glob="**/*.md", loader_cls=TextLoader),
        DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader),
    ]
    
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    
    if not docs:
        raise ValueError(f"No documents found in directory: {data_dir}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)

    # Fully local embeddings via Ollama
    embeddings = OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url
    )
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=settings.collection_name,
        persist_directory=settings.chroma_persist_directory
    )
    
    return vectorstore

if __name__ == "__main__":
    load_and_index_documents()
    print("Ingestion complete. Local Vector store created.")