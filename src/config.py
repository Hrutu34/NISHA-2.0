from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    chroma_persist_directory: str = "./chroma_db"
    collection_name: str = "company_policies"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "llama3.2" 

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()