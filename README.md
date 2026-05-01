# Medical Chatbot with RAG, GraphRAG, and DPO Training

A medical chatbot that uses Retrieval-Augmented Generation (RAG), a Neo4j knowledge graph (GraphRAG), and live web search to provide context-aware responses — with user authentication, persistent chat history, Redis caching, streaming responses, input/output guardrails, real-time LangSmith evaluation, and a DPO training pipeline. Deployed on Google Cloud Run.

## Features

- **RAG Pipeline**: Semantic search over a local medical Q&A dataset via FAISS
- **GraphRAG**: Neo4j knowledge graph of medical concepts — answers relational queries like "what conditions are associated with X"
- **Web Search**: Falls back to live DuckDuckGo search when local knowledge is insufficient
- **Tool-Calling Agent**: Automatically selects the best tool (RAG, GraphRAG, or web search) per query
- **Streaming Responses**: Tokens stream to the UI in real time via LangChain `astream_events`
- **Guardrails**: Input/output safety layer — blocks PII, toxic content, and off-topic queries before they reach the LLM
- **Online Evaluation**: Every live response is automatically scored for relevance, faithfulness, and medical safety using LLM-as-judge, with scores posted back to LangSmith traces
- **LangSmith Tracing**: Full execution traces (tool calls, retrieval, LLM I/O, latency) captured automatically for every request
- **User Authentication**: Register/login with JWT tokens stored as httpOnly cookies
- **Persistent Chat History**: Sessions and messages stored in SQLite (local) or PostgreSQL (Cloud Run)
- **Redis Cache**: Two-layer cache — embedding cache (skip redundant OpenAI calls) + context cache (post-RAG results, 24h TTL)
- **Multiple LLM Support**: OpenAI GPT-4.1 or local Llama models
- **DPO Training**: Fine-tune Llama models using collected human preference pairs
- **Cloud Run Deployment**: Containerised and deployed on Google Cloud Run with GCS, Cloud SQL, and Secret Manager

## Project Structure

```
project/
├── core/
│   ├── config.py               # All config — DB, Redis, GCP, models, LangSmith
│   ├── chatbot.py              # MedicalChatbot — agent, tools, streaming, eval hooks
│   ├── guardrails.py           # Input/output safety (PII, toxicity, off-topic)
│   ├── llm.py                  # LLM factory (OpenAI / local Llama)
│   └── llm_local.py            # Llama loader with optional LoRA adapter
├── rag/
│   ├── pipeline.py             # FAISS retrieval, CSV loading, GCS support
│   └── cache.py                # Redis embedding + context cache
├── graph/
│   ├── builder.py              # Builds Neo4j knowledge graph from CSV
│   └── retrieval.py            # GraphRAG Cypher queries
├── evals/
│   ├── langsmith_evals.py      # LLM-as-judge evaluator functions
│   └── online_evaluator.py     # Async per-response evaluator → LangSmith feedback
├── api/
│   └── auth.py                 # bcrypt + JWT auth
├── db/
│   ├── database.py             # SQLAlchemy models (User, ChatSession, Message)
│   └── session_store.py        # Session CRUD helpers
├── ui/
│   └── app.py                  # Gradio UI — streaming, sidebar, session management
├── training/
│   ├── dpo_trainer.py          # DPO fine-tuning (4-bit quant + LoRA via TRL)
│   └── preference_collector.py # Chosen/rejected pair collection
├── server.py                   # FastAPI server (auth middleware, Gradio mount)
├── main.py                     # CLI entry point
├── Dockerfile                  # python:3.12-slim, amd64
└── deploy.sh                   # One-command deploy via Cloud Build + Cloud Run
```

## Setup (Local)

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

Create a `.env` file in the project root (loaded automatically via `python-dotenv`):

```env
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional — local Llama models
HF_TOKEN=your-huggingface-token

# Optional — GraphRAG
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password

# Optional — LangSmith tracing + online evaluation
LANGCHAIN_API_KEY=ls__your-langsmith-key
LANGCHAIN_PROJECT=medical-chatbot
```

Get a LangSmith API key at [smith.langchain.com](https://smith.langchain.com) → Settings → API Keys. Tracing and online evaluation are silently disabled if `LANGCHAIN_API_KEY` is not set.

### 4. Generate the database encryption key

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Add the output to `.env` as `DB_ENCRYPTION_KEY=...`. If unset, a fallback key is derived from `JWT_SECRET` (dev only — not safe for production).

### 5. Start Redis (optional, for caching)

```bash
redis-server
```

The app runs without Redis but embedding and context caching will be disabled.

### 6. Build the Neo4j knowledge graph (optional)

```bash
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/your-password neo4j:latest
python3 graph/builder.py
```

## Usage

### Web UI

```bash
python -m uvicorn server:app --reload
```

Open `http://localhost:8000`. Register an account and start chatting.

### CLI

```bash
python3 main.py                        # OpenAI GPT-4.1
python3 main.py --model local          # Local Llama base model
python3 main.py --model dpo            # DPO fine-tuned model
python3 main.py --collect-preferences  # Preference collection mode
```

## Guardrails

Every user message passes through three input validators before reaching the LLM, and the generated response passes through one output validator before being delivered:

| Check | Method | Gate |
|-------|--------|------|
| PII detection | Regex (SSN, phone, credit card patterns) | Input |
| Toxic content | OpenAI Moderation API | Input + Output |
| Off-topic | Keyword heuristics + GPT-4o-mini fallback classifier | Input |

The system fails open — if `guardrails-ai` is not installed, all checks pass through and the chatbot operates normally.

## Online Evaluation (LangSmith)

When `LANGCHAIN_API_KEY` is set, every response is automatically evaluated in a background thread with zero impact on response latency:

| Evaluator | What it scores |
|-----------|---------------|
| Answer relevance | Does the response directly address the question? (0–1) |
| Faithfulness | Is the response grounded in retrieved context, or does it hallucinate? (0–1) |
| Medical safety | Does the response avoid dangerous medical advice? (0 or 1) |

Scores are posted as feedback to the corresponding LangSmith trace. You can filter traces by score in the LangSmith UI to find low-quality or unsafe responses in production.

The full agent execution tree — which tool was called, what was retrieved, LLM input/output, and latency per step — is captured automatically via LangChain's tracing integration.

## DPO Training Workflow

### Step 1: Collect preferences

```bash
python3 main.py --collect-preferences
```

Ask a question → the bot generates 2 responses → pick the better one → saved to `data/preferences.jsonl`. Aim for 500+ pairs for measurable improvement with Llama-3.1-8B + LoRA r=16.

### Step 2: Run DPO training

```bash
python3 training/dpo_trainer.py
```

Uses 4-bit quantization + LoRA via HuggingFace TRL's `DPOTrainer`. Output saved to `models/dpo_medical_chatbot/`.

### Step 3: Use the trained model

```bash
python3 main.py --model dpo
```

## Caching (Redis)

Two cache layers keyed by a normalised SHA-256 hash of the query (lowercase, punctuation stripped, stop words removed, words sorted):

| Layer | Key prefix | TTL | What it stores |
|-------|-----------|-----|----------------|
| Embedding cache | `medchat:emb:` | None | OpenAI embedding vectors |
| Context cache | `medchat:ctx:` | 24h | Post-RAG context strings |

Query normalisation ensures cache hits survive minor variations like "What are diabetes symptoms?" vs "symptoms of diabetes".

## Cloud Deployment (Google Cloud Run)

```bash
./deploy.sh
```

Runs Cloud Build (builds the image on GCP) then updates Cloud Run. Takes ~3 minutes.

| Service | Purpose |
|---------|---------|
| Cloud Run | Hosts the containerised FastAPI + Gradio app |
| Cloud SQL (PostgreSQL) | Persistent user/session/message storage |
| Cloud Storage (GCS) | Stores the FAISS vector index |
| Secret Manager | Stores API keys and passwords |
| Cloud Build | Builds amd64 images on GCP (no local cross-compile) |

**Note:** The container is stateless — SQLite data is wiped on every redeploy. Set `DATABASE_URL` to a Cloud SQL connection string for persistent storage.

## Configuration Reference

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `gpt-4.1` | OpenAI model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `LOCAL_BASE_MODEL` | `meta-llama/Llama-3.2-1B-Instruct` | Local Llama |
| `CHUNK_SIZE` | `1000` | RAG chunk size (chars) |
| `CHUNK_OVERLAP` | `200` | RAG chunk overlap |
| `TOP_K_RESULTS` | `3` | Retrieved docs per query |
| `CONTEXT_CACHE_TTL` | `86400` | Redis context TTL (seconds) |
| `LANGCHAIN_PROJECT` | `medical-chatbot` | LangSmith project name |

## Hardware Notes

| Mode | Requirements |
|------|-------------|
| OpenAI | Internet connection only |
| Local Llama (CPU) | 8GB+ RAM, slow |
| Local Llama (GPU) | 6GB+ VRAM |
| DPO Training | GPU with 8GB+ VRAM recommended |

## License

For personal and educational use.
