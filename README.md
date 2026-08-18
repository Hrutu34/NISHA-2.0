# NISHA 2.0: Newcomers' Integration, Support and Help Assistant

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/Orchestration-LangChain-green.svg)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/VectorStore-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![Ragas](https://img.shields.io/badge/Evaluations-Ragas-purple.svg)](https://github.com/explodinggradients/ragas)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An intelligent, hallucination-resistant Retrieval-Augmented Generation (RAG) assistant designed to help new joiners navigate company policies, benefits, travel rules, and workplace guidelines with transparent, chunk-level citations.

---

## Problem & Motivation

Navigating internal corporate documentation during onboarding is often overwhelming for new joiners. Generic LLMs cannot answer domain-specific internal policy queries reliably and are prone to hallucinations. 

**NISHA 2.0** solves this by:
* Grounding responses in actual company policy documents (Leave, Travel, Health Insurance, WFH, IT Assets, etc.).
* Providing verifiable inline source citations back to exact document chunks.
* Running on a **100% free, zero-cost stack** (deployable on Streamlit Community Cloud or locally via Ollama).
* Measuring retrieval quality and generation integrity through advanced evaluation metrics.

---

## System Architecture

```text
                               ┌─────────────────────────────┐
                               │  data/sample_policies/*.md  │
                               └──────────────┬──────────────┘
                                              │ Ingestion & Chunking
                                              ▼
                               ┌─────────────────────────────┐
                               │     HuggingFace / MiniLM    │
                               │     (Local Embeddings)      │
                               └──────────────┬──────────────┘
                                              │ Indexing
                                              ▼
                               ┌─────────────────────────────┐
                               │      ChromaDB (Vector)      │
                               └──────────────┬──────────────┘
                                              │
┌─────────────────────────┐                   │ Similarity Retrieval (Top-K)
│     Streamlit Web UI    │ ─── User Query ───┤
│  (Chat & Citation View) │                   ▼
│                         │ ◄─── Answer ───── ┌─────────────────────────────┐
└─────────────────────────┘      + Sources    │    LLM Generation Engine    │
                                              │  (Groq / LLaMA / Ollama)    │
                                              └─────────────────────────────┘
```

### 1. Ingestion Pipeline
* **Document Extraction:** Parses Markdown (`.md`) and PDF (`.pdf`) policy documents.
* **Semantic Chunking:** Splits text into 800-character chunks with a 150-character overlap using hierarchical header boundaries (`##`, `###`).
* **Vector Indexing:** Generates sentence embeddings via `all-MiniLM-L6-v2` / `nomic-embed-text` and persists to ChromaDB.

### 2. Retrieval & Generation Engine
* **Hybrid Context Matching:** Performs similarity search across policy embeddings to retrieve the top $K=3$ most relevant chunks.
* **Context-Bound Prompting:** Injects retrieved chunks into a system prompt that explicitly instructs the model to refuse unsupported questions and refer employees to HR.
* **Zero-Cost Inference:** Powered by Groq's high-speed free tier (`llama-3.1-8b-instant`) for cloud deployment, or local execution via Ollama (`llama3.2`).

---

## Tech Stack

| Layer | Technology | Purpose / Advantage |
| :--- | :--- | :--- |
| **User Interface** | [Streamlit](https://streamlit.io/) | Lightweight, interactive chat interface with source citation accordion |
| **Framework** | [LangChain](https://www.langchain.com/) | Modular orchestration of retrievers, document loaders, and chains |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) | Persistent, lightweight, embedded vector database |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Open-source, CPU-friendly embeddings (runs at zero cost) |
| **LLM Inference** | [Groq](https://groq.com/) / [Ollama](https://ollama.com/) | Ultra-fast free cloud inference or 100% private local execution |
| **Evaluation** | [Ragas](https://github.com/explodinggradients/ragas) | Quantitative scoring of the RAG Triad (Faithfulness, Relevancy, Recall) |

---

## Repository Structure

```text
nisha-2.0/
├── data/
│   └── sample_policies/                  # Sample markdown policy documents
│       ├── asset_policy.md               # IT hardware & usage rules
│       ├── health_insurance_policy.md    # Coverage & dependent guidelines
│       ├── hybrid_and_wfh_policy.md      # Hybrid schedule & stipend
│       ├── learning_and_development_policy.md # Certifications & budget
│       ├── leave_policy.md               # Annual, sick, and parental leaves
│       ├── probation_and_confirmation_policy.md # 90-day review roadmap
│       ├── shift_allowance_policy.md     # Shift differential & on-call rates
│       ├── transfer_policy.md            # Relocation assistance
│       ├── travel_policy.md              # Per diem & booking rules
│       └── workplace_conduct_policy.md   # Ethics, POSH, & grievance escalation
├── src/
│   ├── __init__.py
│   ├── config.py                         # Environment settings & config
│   ├── ingestion.py                      # Vector store indexing workflow
│   ├── rag_engine.py                     # Retrieval chain & prompt formatting
│   ├── evals.py                          # Automated Ragas evaluation suite
│   └── api.py                            # Optional FastAPI endpoints
├── app.py                                # Streamlit chat application
├── requirements.txt                      # Project dependencies
├── .env.example                          # Environment template
└── README.md
```

---

## Quickstart Guide

### Option A: Local Run (Free via Groq API)

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/hrutu34/nisha-2.0.git](https://github.com/hrutu34/nisha-2.0.git)
   cd nisha-2.0
   ```

2. **Create a virtual environment & install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   Add your free Groq API key from [Groq Console](https://console.groq.com/):
   ```env
   GROQ_API_KEY=gsk_your_free_key_here
   ```

4. **Index policies & launch the app:**
   ```bash
   python -m src.ingestion
   streamlit run app.py
   ```

---

### Option B: 100% Local Run (No External APIs via Ollama)

1. **Install and run Ollama:**
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

2. **Update your `.env`:**
   ```env
   OLLAMA_BASE_URL=http://localhost:11434
   LLM_MODEL=llama3.2
   EMBEDDING_MODEL=nomic-embed-text
   ```

3. **Run Streamlit:**
   ```bash
   streamlit run app.py
   ```

---

## Evaluation & Quality Benchmarks

NISHA 2.0 integrates automated testing using the **Ragas** framework to evaluate model hallucination and retrieval quality across 4 key dimensions:

```text
                  ┌──────────────────────┐
                  │      RAG TRIAD       │
                  └──────────┬───────────┘
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Faithfulness   │ │Answer Relevance │ │Context Precision│
│ (No Hallucinate)│ │ (Direct Answer) │ │  (Signal/Noise) │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

| Evaluation Metric | Target Score | Description |
| :--- | :---: | :--- |
| **Faithfulness** | $> 0.90$ | Measures if the response is strictly derived from retrieved text. |
| **Answer Relevancy** | $> 0.85$ | Evaluates how directly the answer addresses the user's prompt. |
| **Context Precision** | $> 0.85$ | Measures if the most relevant chunks are ranked at the top. |
| **Context Recall** | $> 0.90$ | Measures whether all necessary facts from ground truth were retrieved. |

To run the automated evaluation suite:
```bash
python -m src.evals
```

---

## Free Cloud Deployment (Streamlit Community Cloud)

1. Push your repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and connect your repository (`hrutu34/nisha-2.0`).
3. Set the main file path to `app.py`.
4. Add your secrets under **App Settings > Secrets**:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
5. Click **Deploy** for a zero-cost, always-on web demo.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

> Author : Hrutu Surve
