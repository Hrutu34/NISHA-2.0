# NISHA 2.0: Newcomers' Integration, Support and Help Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nisha-2.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/orchestration-LangChain-green.svg)](https://python.langchain.com/)
[![Groq GPT-OSS-20B](https://img.shields.io/badge/LLM-GPT--OSS--20B%20(Groq)-orange.svg)](https://groq.com/)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An intelligent, hallucination-resistant Retrieval-Augmented Generation (RAG) assistant designed to help new joiners navigate company policies, benefits, travel rules, and workplace guidelines with transparent, chunk-level citations.

---

## ✨ Key Features

- **⚡ Real-Time Response Streaming:** Responses stream smoothly character-by-character using `st.write_stream` for an interactive, natural conversational feel.
- **📄 Native In-App PDF Document Viewer:** Employees can read original, official corporate PDF policy documents directly in the browser with instant download support.
- **📌 Transparent Source Citations:** Every answer cites the exact policy document and excerpt retrieved from the knowledge base, eliminating hallucinations.
- **🏢 Enterprise Knowledge Base:** Grounded in 11 comprehensive, interconnected corporate policies spanning HR, legal, travel, compensation, IT security, and executive governance.
- **🚀 Zero-Cost Production Stack:** Optimized for cost efficiency using HuggingFace embeddings (`all-MiniLM-L6-v2`), local Chroma vector storage, and Groq's high-speed inference engine running OpenAI's open-weight **GPT-OSS 20B** model.
- **🌐 Dual Interface:** Interactive Streamlit UI for employees and a standalone FastAPI backend (`src/api.py`) for enterprise portal integration.

---

## Problem & Motivation

Navigating internal corporate documentation during onboarding is often overwhelming for new joiners. Generic LLMs cannot answer domain-specific internal policy queries reliably and are prone to hallucinations. 

**NISHA 2.0** solves this by:
* Grounding responses in actual company policy documents (Leave, Travel, Health Insurance, WFH, IT Assets, etc.).
* Providing verifiable inline source citations back to exact document chunks.
* Running on a **100% free, zero-cost stack** (deployable on Streamlit Community Cloud and uses Groq API).
* Measuring retrieval quality and generation integrity through advanced evaluation metrics.

---

## System Architecture

```text
                               ┌─────────────────────────────┐
                               │  data/sample_policies/*.pdf │
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
* **Document Extraction:** Parses PDF (`.pdf`) policy documents.
* **Semantic Chunking:** Splits text into 600-character chunks with an 80-character overlap using recursive text separators.
* **Vector Indexing:** Generates sentence embeddings via `all-MiniLM-L6-v2` / `nomic-embed-text` and persists to ChromaDB.

### 2. Retrieval & Generation Engine
* **Hybrid Context Matching:** Performs similarity search across policy embeddings to retrieve the top $K=4$ most relevant chunks.
* **Context-Bound Prompting:** Injects retrieved chunks into a system prompt that explicitly instructs the model to refuse unsupported questions and refer employees to HR.
* **Zero-Cost Inference:** Powered by Groq's high-speed free tier (`gpt-oss-20b`) for cloud deployment.

---
##Project Structure

```
nisha-2.0/
├── data/
│   └── sample_policies/        # Multi-page corporate policy PDF documents
│       ├── global_travel_and_expense_policy_v4.pdf
│       ├── health_insurance_and_wellness_policy_v6.pdf
│       ├── hierarchy_and_leadership_directory.pdf
│       ├── hybrid_and_wfh_policy_v4.pdf
│       ├── it_asset_management_policy_v3.pdf
│       ├── learning_and_development_policy_v2.pdf
│       ├── leave_and_attendance_policy_v5.pdf
│       ├── probation_and_confirmation_policy_v5.pdf
│       ├── shift_allowance_policy_v3.pdf
│       ├── transfer_and_relocation_policy_v4.pdf
│       └── workplace_conduct_and_posh_policy_v4.pdf
├── src/
│   ├── __init__.py
│   ├── api.py                  # FastAPI REST API endpoints
│   ├── config.py               # Centralized path and model configurations
│   ├── evals.py                # Ragas & evaluation benchmarking suite
│   ├── ingestion.py            # PDF document loading, chunking, and ChromaDB vector indexing
│   └── rag_engine.py           # LangChain Groq streaming chain & retriever setup
├── app.py                      # Main Streamlit web application with PDF viewer
├── requirements.txt            # Python dependencies
├── LICENSE
└── README.md
```

---

## Tech Stack

| Layer | Technology | Purpose / Advantage |
| :--- | :--- | :--- |
| **User Interface** | [Streamlit](https://streamlit.io/) | Lightweight, interactive chat interface with source citation accordion |
| **Framework** | [LangChain](https://www.langchain.com/) | Modular orchestration of retrievers, document loaders, and chains |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) | Persistent, lightweight, embedded vector database |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Open-source, CPU-friendly embeddings (runs at zero cost) |
| **LLM Inference** | [Groq](https://groq.com/)| Ultra-fast free cloud inference |
| **Evaluation** | [Ragas](https://github.com/explodinggradients/ragas) | Quantitative scoring of the RAG Triad (Faithfulness, Relevancy, Recall) |

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

3. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Configure Environment Variables
   Create a .env file in the root directory (or configure secrets in Streamlit Cloud):
   ```bash
   Ini, TOML
   GROQ_API_KEY=your_groq_api_key_here
   ```
   
5. Launch the Application
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
