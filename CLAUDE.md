# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Required environment variables:**
- `OPENAI_API_KEY` — for OpenAI GPT-4.1 and embeddings (always required)
- `HF_TOKEN` — for HuggingFace gated Llama models (required for local/DPO modes)
- `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` — for GraphRAG (optional, defaults to localhost)
- `JWT_SECRET` — for auth token signing (defaults to `change-me-in-production`)
- `DATABASE_URL` — SQLite locally (`sqlite:///data/users.db`), PostgreSQL on Cloud Run
- `REDIS_URL` — for caching (defaults to `redis://localhost:6379`; app runs without it)
- `LANGCHAIN_API_KEY` — LangSmith tracing and online evals (optional; tracing disabled if unset)
- `LANGCHAIN_PROJECT` — LangSmith project name (defaults to `medical-chatbot`)

## Running the Application

| Command | Purpose |
|---------|---------|
| `python -m uvicorn server:app --reload` | Web UI at http://localhost:8000 (auth + Gradio) |
| `python3 main.py` | CLI chat with OpenAI GPT-4.1 |
| `python3 main.py --model local` | CLI chat with Llama base model |
| `python3 main.py --model dpo` | CLI chat with DPO fine-tuned model |
| `python3 main.py --collect-preferences` | Preference collection mode |
| `python3 training/dpo_trainer.py` | Run DPO fine-tuning on collected preferences |
| `python3 graph/builder.py` | Build Neo4j knowledge graph from CSV |
| `./deploy.sh` | Deploy to Google Cloud Run |

**CLI interactive commands:** `quit`, `clear`, `stats` (preference mode only)

## Package Structure

```
project/
├── core/
│   ├── config.py          # All config — DB, Redis, GCP, models, paths, LangSmith
│   ├── chatbot.py         # MedicalChatbot — agent, tools, streaming, eval hooks
│   ├── guardrails.py      # GuardrailsManager — input/output safety (PII, toxicity, off-topic)
│   ├── llm.py             # LLMManager — OpenAI / local Llama factory
│   └── llm_local.py       # Llama loader with optional LoRA adapter
│
├── rag/
│   ├── pipeline.py        # RAGPipeline — FAISS, CSV loading, GCS support
│   └── cache.py           # CacheManager — Redis embedding + context cache
│
├── graph/
│   ├── builder.py         # Builds Neo4j graph from CSV
│   └── retrieval.py       # GraphRAGPipeline — Cypher queries
│
├── evals/
│   ├── langsmith_evals.py # LLM-as-judge evaluator functions (relevance, faithfulness, safety)
│   └── online_evaluator.py # OnlineEvaluator — async eval on every live response
│
├── api/
│   └── auth.py            # AuthManager — bcrypt, JWT, register/login
│
├── db/
│   ├── database.py        # SQLAlchemy models: User, ChatSession, Message
│   └── session_store.py   # Session CRUD helpers
│
├── ui/
│   └── app.py             # Gradio UI — streaming, sidebar, session mgmt
│
├── training/
│   ├── dpo_trainer.py     # DPO fine-tuning (4-bit quant + LoRA via TRL)
│   └── preference_collector.py  # Chosen/rejected pair collection
│
├── server.py              # FastAPI app — auth routes, middleware, Gradio mount
├── main.py                # CLI entry point
├── Dockerfile             # python:3.12-slim, amd64
└── deploy.sh              # Cloud Build + Cloud Run one-command deploy
```

## Architecture

A production-deployed medical Q&A chatbot with user auth, three retrieval strategies, Redis caching, streaming responses, persistent chat history, input/output guardrails, online LLM-as-judge evaluation, and a DPO fine-tuning pipeline.

### Data Flow

```
Browser
  │
  ▼
FastAPI (server.py)
  │
  ├── Unauthenticated → /login or /register
  │       └── AuthManager (api/auth.py)
  │             bcrypt password hash → SQLAlchemy → DB
  │             JWT token → httpOnly cookie
  │
  └── Authenticated → /app (Gradio UI mounted here)
          │
          ▼
      Gradio UI (ui/app.py)
          │   per-tab state: user_id, session_id, lc_history
          │
          ▼
      respond() — streaming generator
          │
          ├── GuardrailsManager.check_input_sync()   ← PII / toxicity / off-topic gate
          │       blocked → return refusal message immediately
          │
          ├── CacheManager.get_context()  ←── Redis (rag/cache.py)
          │       hit → skip retrieval entirely
          │       miss → continue
          │
          ▼
      MedicalChatbot.stream_with_history() (core/chatbot.py)
          │   captures root run_id + tool outputs from astream_events
          │
          ▼
      LangChain Tool-Calling Agent  ──── traced by LangSmith
          │
          ├── rag_retrieval ──────► RAGPipeline (rag/pipeline.py)
          │                         CacheManager.get_embedding() ← Redis
          │                         OpenAI text-embedding-3-small
          │                         FAISS similarity search → Top-K chunks
          │
          ├── graph_retrieval ────► GraphRAGPipeline (graph/retrieval.py)
          │                         Neo4j Cypher queries
          │                         concept match + co-occurrence lookup
          │
          └── web_search ─────────► DuckDuckGoSearchRun
          │
          ▼
      GPT-4.1 streams tokens via LangChain astream_events
          │
          ▼
      GuardrailsManager.check_output_sync()  ← toxicity gate on full response
          │
          ▼
      Response delivered to user
          │
          ▼  (daemon thread — zero latency impact)
      OnlineEvaluator (evals/online_evaluator.py)
          ├── answer_relevance  → gpt-4o-mini judge → score
          ├── faithfulness      → grounding check vs retrieved context → score
          └── medical_safety    → dangerous advice check → score
                    └── all scores posted to LangSmith trace as feedback
          │
          ▼
      On completion:
        SessionStore.save_message()  →  SQLite (local) / PostgreSQL (Cloud Run)
        CacheManager.set_context()   →  Redis (24h TTL)
```

### Agent Tool Selection

| Query type | Tool |
|---|---|
| Factual — "what is X", "symptoms of X", "how is X treated" | `rag_retrieval` first |
| Relational — "what's associated with X", "what co-occurs with X" | `graph_retrieval` first |
| Neither returned useful results | `web_search` |

Queries are passed **verbatim** to tools (not rephrased) — enforced in system prompt to maximise Redis cache hit rate.

### Guardrails (`core/guardrails.py`)

Three input validators run before every LLM call, one output validator runs after:

| Check | Method | Gate |
|---|---|---|
| PII detection | Regex (SSN, phone, credit card) | Input |
| Toxic content | OpenAI Moderation API | Input + Output |
| Off-topic | Keyword heuristics + GPT-4o-mini fallback | Input |

`GuardrailsManager` is a singleton and fails open — if `guardrails-ai` is not installed or any check errors unexpectedly, the chatbot continues normally.

### Online Evaluation (`evals/`)

Every live response is scored asynchronously in a daemon thread. No curated dataset or offline step required.

| Evaluator | What it checks | Judge |
|---|---|---|
| `answer_relevance` | Does the answer address the question? | gpt-4o-mini |
| `faithfulness` | Is the answer grounded in retrieved context? | gpt-4o-mini |
| `medical_safety` | Does it avoid dangerous medical advice? | gpt-4o-mini |

Scores are posted back to the LangSmith trace as feedback via `Client.create_feedback()`. Requires `LANGCHAIN_API_KEY` — silently disabled otherwise.

### Neo4j Knowledge Graph Schema

```
(QAPair {id, question, answer})
    └──[:TAGGED_WITH]──► (MedicalConcept {name})
                              └──[:CO_OCCURS_WITH {weight}]──► (MedicalConcept)
```

Built by `graph/builder.py` from CSV rows where `label=1.0`. Run `python3 graph/builder.py` to populate.

### Redis Cache Layers

| Layer | Key prefix | TTL | Stores |
|---|---|---|---|
| Embedding cache | `medchat:emb:` | None | OpenAI embedding vectors |
| Context cache | `medchat:ctx:` | 24h | Post-RAG context strings |

Cache key = SHA-256 of normalised query (lowercase, punctuation stripped, stop words removed, words sorted). App runs without Redis — caching degrades gracefully.

### Database Models (`db/database.py`)

- `User` — `user_id`, `email`, `username`, `password_hash`, `is_active`
- `ChatSession` — `session_id`, `user_id`, `title`, `model_type`
- `Message` — `message_id`, `session_id`, `role`, `content`, `tools_used` (JSON list)

SQLite locally (WAL mode), PostgreSQL on Cloud Run. Switched via `DATABASE_URL` env var — no code changes.

### Configuration Defaults (`core/config.py`)

```
LLM_MODEL = "gpt-4.1"
EMBEDDING_MODEL = "text-embedding-3-small"
LOCAL_BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
LOCAL_ADAPTER_PATH = "models/dpo_medical_chatbot"
CSV_PATH = "data/train_data_chatbot_small.csv"
VECTOR_STORE_PATH = "data/vector_store"
PREFERENCES_PATH = "data/preferences.jsonl"
CHUNK_SIZE = 1000, CHUNK_OVERLAP = 200, TOP_K_RESULTS = 3
DATABASE_URL = "sqlite:///data/users.db"
REDIS_URL = "redis://localhost:6379"
CONTEXT_CACHE_TTL = 86400  # 24h
JWT_EXPIRE_HOURS = 24
LANGCHAIN_PROJECT = "medical-chatbot"
```

### Training Workflow

1. Collect preference pairs: `python3 main.py --collect-preferences`
   (user picks better of 2 responses → saved to `data/preferences.jsonl`)
2. Fine-tune: `python3 training/dpo_trainer.py` → saves LoRA adapter to `models/dpo_medical_chatbot/`
3. Use trained model: `python3 main.py --model dpo`

Minimum ~500 preference pairs recommended for measurable DPO improvement on Llama-3.1-8B with LoRA r=16.

### Cloud Deployment (`deploy.sh`)

Builds image on Cloud Build (no local cross-compile) and deploys to Cloud Run:
- **Runtime**: Cloud Run, `us-central1`, 2Gi RAM, 1 CPU, 0–2 instances
- **Secrets**: Secret Manager (`OPENAI_API_KEY`, `JWT_SECRET`, `NEO4J_PASSWORD`, etc.)
- **Database**: Cloud SQL PostgreSQL 15 via Unix socket
- **FAISS index**: loaded from GCS (`gs://<project>-assets/vector_store`)
- **Container**: stateless — SQLite data is wiped on redeploy; use Cloud SQL for persistence

### Data Files

- `data/train_data_chatbot_small.csv` — primary medical Q&A dataset (3.3MB, default); columns: `short_question`, `short_answer`, `tags`, `label`
- `data/train_data_chatbot.csv` — larger version (34MB)
- `data/vector_store/` — cached FAISS index (auto-created on first run, or loaded from GCS)
- `data/preferences.jsonl` — DPO training data
- `data/users.db` — SQLite database (local only)

### Hardware Notes

- **OpenAI mode**: internet connection only
- **Local Llama (CPU)**: 8GB+ RAM, slow
- **DPO training**: GPU with 8GB+ VRAM recommended
