from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.config import settings

def run_evaluation_suite(test_cases: list[dict]):
    """
    test_cases format:
    [
        {
            "question": "What is the policy on parental leave?",
            "contexts": ["..."],
            "answer": "...",
            "ground_truth": "..."
        }
    ]
    """
    eval_dataset = Dataset.from_list(test_cases)
    
    evaluator_llm = ChatOpenAI(model="gpt-4o", api_key=settings.openai_api_key)
    evaluator_embeddings = OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)

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