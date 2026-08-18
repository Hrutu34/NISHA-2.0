from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from src.config import settings

def run_evaluation_suite(test_cases: list[dict]):
    eval_dataset = Dataset.from_list(test_cases)
    
    # Configure Ragas to use local models for evaluation
    evaluator_llm = ChatOllama(
        model=settings.llm_model, 
        base_url=settings.ollama_base_url,
        temperature=0.0
    )
    evaluator_embeddings = OllamaEmbeddings(
        model=settings.embedding_model, 
        base_url=settings.ollama_base_url
    )

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]

    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )
    
    return results.to_pandas()