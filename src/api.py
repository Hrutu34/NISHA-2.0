from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.rag_engine import get_rag_chain
from src.ingestion import load_and_index_documents

app = FastAPI(title="NISHA 2.0 API", version="2.0.0")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

rag_chain = None

@app.on_event("startup")
def startup_event():
    global rag_chain
    try:
        rag_chain, _ = get_rag_chain()
    except Exception:
        pass

@app.post("/query", response_model=QueryResponse)
def query_policy(request: QueryRequest):
    if not rag_chain:
        raise HTTPException(status_code=503, detail="Vector store not initialized. Ingest data first.")
    response = rag_chain.invoke(request.question)
    return QueryResponse(answer=response)

@app.post("/ingest")
def trigger_ingestion():
    try:
        load_and_index_documents()
        global rag_chain
        rag_chain, _ = get_rag_chain()
        return {"status": "success", "message": "Documents indexed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))