# NISHA-2.0

Upload a set of documents/policies (PDFs, Markdowns, API specs) and ask natural-language questions about them from a new joiner`s perspective, as they have lesser knowledge about existing company policies. Answers are grounded in the actual content, with inline citations back to the exact source chunk — not a black box that might be making things up.

**🔗 Live demo:** `<Work in Progress>`
**🔗 API docs:** `<Work in Progress>/docs` 

> Note: the backend runs on Render's free tier and spins down after ~15 minutes of inactivity. The first request after idle time may take 30-50 seconds to wake up — please be patient on first load.

---

## Problem

Generic LLM chat answers questions from training data, which means it can't answer questions about custom policy documents, and it can't tell you where an answer came from. This project builds a retrieval-augmented pipeline that grounds every answer in retrieved source chunks, surfaces those sources transparently, and explicitly declines to answer when retrieval confidence is low — rather than confidently hallucinating.

## Architecture

```
┌─────────────┐        ┌───────────────────────┐
│   React     │  HTTP  │      FastAPI API       │
│  Frontend   │◄──────►│  ───────────────────   │
│  (Vercel)   │        │  Upload · Query · Eval │
└─────────────┘        └──────────┬────────────┘
                                   │
                 ┌─────────────────┼─────────────────┐
                 ▼                 ▼                 ▼
        ┌────────────────┐ ┌──────────────┐  ┌──────────────┐
        │  Chunking +    │ │   Pinecone    │  │   LLM API    │
        │  Embedding     │ │ (vector store)│  │ (OpenAI/Groq)│
        │  Pipeline      │ └──────────────┘  └──────────────┘
        └────────────────┘
```

**Ingestion flow:** Document uploaded → text extracted → split into overlapping chunks (~500 tokens, 50-token overlap) → each chunk embedded → stored in Pinecone with metadata (source file, page number).

**Query flow:** User question → embedded → top-k similar chunks retrieved from Pinecone → chunks + question assembled into a prompt → LLM generates an answer with citations → frontend renders the answer with expandable source references.

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Backend | Python, FastAPI | Fast to build, rich RAG/embedding ecosystem |
| Document parsing | `unstructured` / `pypdf` | Extracts text from PDFs, Markdown, plain text |
| Embeddings | OpenAI `text-embedding-3-small` (or `sentence-transformers` locally for zero-cost dev) | Cheap, solid quality for a portfolio-scale corpus |
| Vector store | Pinecone (free tier) | Managed similarity search, no infra to run |
| Generation | OpenAI / Groq | Composes grounded answers from retrieved chunks |
| Frontend | React (Vite) | Upload UI + chat interface with citation badges |
| Hosting | Render (backend), Vercel (frontend), Pinecone (vectors), Supabase Storage (optional file persistence) | Free tiers, GitHub auto-deploy |
| CI | GitHub Actions | Runs `pytest` on every push |

## Key Design Decisions

- **Chunking strategy is deliberate, not default.** 500-token chunks with 50-token overlap were chosen to balance context completeness against retrieval precision — naive chunking (e.g., fixed character splits with no overlap) is the most common reason RAG systems retrieve incomplete or fragmented context.
- **Citations are first-class, not an afterthought.** Every answer links back to the specific source file/chunk it drew from. This is what separates a usable internal tool from a demo — it lets a user actually verify the answer instead of trusting it blindly.
- **Explicit "I don't know" behavior.** If retrieval similarity scores fall below a confidence threshold, the system tells the user it couldn't find a grounded answer instead of letting the LLM fill the gap with a plausible-sounding guess. This directly extends hallucination-mitigation work from an earlier LLM itinerary-planning project.
- **(Optional, if implemented) Re-ranking.** A lightweight re-ranking pass over the top-k retrieved chunks before generation, to push the most relevant chunk to the top of context rather than relying on raw embedding similarity alone.

## Evaluation

A small hand-built eval set, run against my own uploaded test documents:

| Question | Expected source | Retrieved correctly? | Answer quality (1-5) |
|---|---|---|---|
| "What authentication method does the API use?" | `api-spec.md`, section 3 | ✅ | 5 |
| "What's the rate limit for the /users endpoint?" | `api-spec.md`, section 5 | ✅ | 4 |
| "What's the pricing model?" (not in the docs) | — should decline | ✅ correctly declined | 5 |

## Running Locally

```bash
# Backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add OpenAI key, Pinecone key/environment
uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

Requires: Python 3.11+, Node 18+, an OpenAI (or Groq) API key, a Pinecone account/index.

## Known Limitations

- Free-tier Pinecone index limits corpus size — fine for a portfolio demo, not for production document volumes.
- Uploaded files are processed transiently by default (not persisted across sessions unless Supabase Storage is wired in).
- Free-tier Render backend cold-starts after inactivity (~30-50s first request).
- No automatic re-indexing if a source document is updated — currently requires re-upload.

## Roadmap / What I'd Add Next

- Re-ranking with a cross-encoder or Cohere's rerank endpoint for higher precision on ambiguous queries.
- Support for incremental re-indexing when a document changes.
- Multi-document comparison queries ("how does the auth flow differ between v1 and v2 of the API spec?").

---

**Author:** Hrutu Surve 