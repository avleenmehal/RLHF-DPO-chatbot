# Medical Chatbot with RAG, GraphRAG, and DPO Training

A medical chatbot that uses Retrieval-Augmented Generation (RAG), a Neo4j knowledge graph (GraphRAG), and live web search to provide context-aware responses — with user authentication, persistent chat history, Redis caching, streaming responses, and a DPO training pipeline. Deployed on Google Cloud Run.

## Features

- **RAG Pipeline**: Semantic search over a local medical Q&A dataset via FAISS
- **GraphRAG**: Neo4j knowledge graph of medical concepts — answers relational queries like "what conditions are associated with X"
- **Web Search**: Falls back to live DuckDuckGo search when local knowledge is insufficient
- **Tool-Calling Agent**: Automatically selects the best tool (RAG, GraphRAG, or web search) per query
- **Streaming Responses**: Tokens stream to the UI in real time via LangChain `astream_events`
- **User Authentication**: Register/login with JWT tokens stored as httpOnly cookies
- **Persistent Chat History**: Sessions and messages stored in SQLite (local) or PostgreSQL (Cloud Run)
- **Redis Cache**: Two-layer cache — embedding cache (skip redundant OpenAI calls) + context cache (post-RAG results, 24h TTL)
- **Multiple LLM Support**: OpenAI GPT-4.1 or local Llama models
- **DPO Training**: Fine-tune Llama models using collected human preference pairs
- **Cloud Run Deployment**: Containerised and deployed on Google Cloud Run with GCS, Cloud SQL, and Secret Manager

## Project Structure

```
ChatbotMedical/
├── config.py                 # Configuration (models, paths, DB, Redis, GCP)
├── llm.py                    # LLM manager (OpenAI/Local)
├── llm_local.py              # Local Llama model loader with LoRA support
├── rag.py                    # RAG pipeline (FAISS, GCS loading, embedding cache)
├── graph_builder.py          # Builds Neo4j knowledge graph from CSV data
├── graph_retrieval.py        # GraphRAG retrieval pipeline
├── chatbot.py                # Tool-calling agent (RAG + GraphRAG + web search)
├── cache.py                  # Redis CacheManager (embedding + context cache)
├── app.py                    # Gradio web UI (streaming, session sidebar)
├── server.py                 # FastAPI server (auth middleware, /health, /login)
├── auth.py                   # JWT auth + bcrypt password hashing
├── database.py               # SQLAlchemy models (User, ChatSession, Message)
├── session_store.py          # Session CRUD helpers
├── main.py                   # CLI entry point
├── preference_collector.py   # Preference data collection
├── dpo_trainer.py            # DPO training script
├── Dockerfile                # Serving container (python:3.12-slim, amd64)
├── deploy.sh                 # One-command deploy via Cloud Build + Cloud Run
├── requirements.txt          # Serving dependencies
├── requirements-training.txt # Training dependencies (torch, trl, peft, etc.)
├── data/
│   ├── train_data_chatbot_small.csv  # Medical Q&A dataset (3.3MB, default)
│   ├── train_data_chatbot.csv        # Larger dataset (34MB)
│   ├── preferences.jsonl             # Collected DPO preference pairs
│   └── vector_store/                 # FAISS index (auto-created, or loaded from GCS)
└── models/
    └── dpo_medical_chatbot/          # Trained LoRA adapter output
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

```bash
export OPENAI_API_KEY="sk-your-key-here"

# Optional — only for local Llama models
export HF_TOKEN="your-huggingface-token"

# Optional — only for GraphRAG
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your-neo4j-password"
```

Or create a `.env` file — it is loaded automatically via `python-dotenv`.

### 4. Start Redis (for caching)

```bash
redis-server
```

The app runs without Redis but caching is disabled. Embeddings and RAG context will not be cached.

### 5. (Optional) Build the Neo4j knowledge graph

```bash
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/your-password neo4j:latest
python3 graph_builder.py
```

## Usage

### Web UI (recommended)

```bash
uvicorn server:app --reload
```

Open `http://localhost:8000` — you will be redirected to `/login`. Register an account, then start chatting.

### CLI

```bash
python3 main.py                        # OpenAI GPT-4.1
python3 main.py --model local          # Local Llama base model
python3 main.py --model dpo            # DPO fine-tuned model
python3 main.py --collect-preferences  # Preference collection mode
```

### CLI commands

| Command | Description |
|---------|-------------|
| `quit` | Exit |
| `clear` | Clear conversation history |
| `stats` | View preference count (preference mode only) |

## Authentication

- Register at `/register`, login at `/login`
- JWT token stored as an httpOnly cookie (`access_token`)
- All `/app/*` routes are protected — unauthenticated requests redirect to `/login`
- Logout at `/logout`

## Caching (Redis)

Two cache layers keyed by a normalised, stop-word-stripped SHA-256 hash of the query:

| Layer | Key prefix | TTL | What it stores |
|-------|-----------|-----|----------------|
| Embedding cache | `medchat:emb:` | None | OpenAI embedding vectors |
| Context cache | `medchat:ctx:` | 24h | Post-RAG context strings |

Query normalisation (lowercase, punctuation stripped, stop words removed, words sorted) ensures cache hits survive minor query variations like "What are diabetes symptoms?" vs "symptoms of diabetes".

## Streaming Responses

Responses stream token-by-token using LangChain `astream_events` with the `on_chat_model_stream` event. The async generator runs in a daemon thread and feeds tokens through a `Queue` to Gradio's generator-based `respond()` function.

## Persistent Chat History

| Environment | Database |
|-------------|----------|
| Local | SQLite (`data/users.db`) with WAL mode |
| Cloud Run | PostgreSQL via Cloud SQL (set `DATABASE_URL`) |

SQLAlchemy handles both. Switching from SQLite to PostgreSQL requires only a `DATABASE_URL` environment variable change.

Sessions are created lazily — only when the first message is sent, not on page load.

## Redeployment

Every time you make code changes, redeploy with:

```bash
./deploy.sh
```

This runs Cloud Build (builds the image on GCP) and then updates Cloud Run. Takes ~3 minutes.

### Important: data is wiped on every redeploy

The current setup uses **SQLite inside the container**. Since the container is replaced on every deploy, all user accounts and chat history are deleted. This is fine for testing but not for production.

| Data | Current behaviour | Fix for production |
|------|------------------|--------------------|
| User accounts | Wiped on redeploy | Add Cloud SQL (PostgreSQL) |
| Chat history | Wiped on redeploy | Add Cloud SQL (PostgreSQL) |
| FAISS index | Survives (loaded from GCS) | Already solved |
| Redis cache | No cache (not set up) | Add Cloud Memorystore |

To add Cloud SQL and make data persistent:
```bash
# Create the database instance (~5 mins, ~$7/month)
gcloud sql instances create medchat-db --database-version=POSTGRES_15 --tier=db-f1-micro --region=us-central1 --project=medical-chat-prod
gcloud sql databases create medchat --instance=medchat-db
gcloud sql users create medchat-user --instance=medchat-db --password=YOUR_PASSWORD

# Then update deploy.sh to add:
# --add-cloudsql-instances=medical-chat-prod:us-central1:medchat-db
# --set-env-vars="DATABASE_URL=postgresql+psycopg2://medchat-user:YOUR_PASSWORD@/medchat?host=/cloudsql/medical-chat-prod:us-central1:medchat-db"
```

## How the Agent Selects Tools

| Query type | Tool used |
|-----------|-----------|
| Factual ("what is X", "symptoms of X") | `rag_retrieval` |
| Relational ("what co-occurs with X") | `graph_retrieval` |
| Insufficient local results | `web_search` |

## DPO Training Workflow

### Step 1: Collect preferences

```bash
python3 main.py --collect-preferences
```

Ask a question → bot generates 2 responses → pick the better one → saved to `data/preferences.jsonl`. Aim for 100+ pairs.

### Step 2: Run DPO training

```bash
pip install -r requirements-training.txt
python3 dpo_trainer.py
```

Uses 4-bit quantization + LoRA via HuggingFace TRL's `DPOTrainer`. Output saved to `models/dpo_medical_chatbot/`.

### Step 3: Use the trained model

```bash
python3 main.py --model dpo
```

## Cloud Deployment (Google Cloud Run)

### GCP services used

| Service | Purpose |
|---------|---------|
| **Cloud Run** | Hosts the containerised FastAPI + Gradio app |
| **Artifact Registry** | Stores Docker images |
| **Cloud Build** | Builds amd64 images on GCP (no local cross-compile) |
| **Cloud SQL (PostgreSQL)** | Persistent user/session/message storage |
| **Cloud Storage (GCS)** | Stores the FAISS vector index |
| **Secret Manager** | Stores API keys and passwords securely |
| **IAM / Service Accounts** | Scoped permissions for Cloud Run |

### Deploy

```bash
./deploy.sh
```

This runs `gcloud builds submit` (builds on GCP) then `gcloud run deploy`. First-time setup requires the phases in the deployment section below.

### Architecture on GCP

```
User → Cloud Run (FastAPI + Gradio)
           ├── Cloud SQL        (users, sessions, messages)
           ├── GCS              (FAISS vector index)
           └── Secret Manager   (OPENAI_API_KEY, JWT_SECRET, DB password)
```

### Environment variables (Cloud Run)

| Variable | Source | Description |
|----------|--------|-------------|
| `OPENAI_API_KEY` | Secret Manager | OpenAI API key |
| `JWT_SECRET` | Secret Manager | JWT signing secret |
| `NEO4J_PASSWORD` | Secret Manager | Neo4j password |
| `DATABASE_URL` | Env var | PostgreSQL connection string |
| `VECTOR_STORE_PATH` | Env var | `gs://bucket/vector_store` |
| `GCP_PROJECT` | Env var | GCP project ID |

## Configuration Reference

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `gpt-4.1` | OpenAI model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `LOCAL_BASE_MODEL` | `meta-llama/Llama-3.2-1B-Instruct` | Local Llama |
| `CHUNK_SIZE` | `1000` | RAG chunk size |
| `CHUNK_OVERLAP` | `200` | RAG chunk overlap |
| `TOP_K_RESULTS` | `3` | Retrieved docs per query |
| `CONTEXT_CACHE_TTL` | `86400` | Redis context cache TTL (seconds) |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection |
| `DATABASE_URL` | `sqlite:///data/users.db` | Database connection |

## Hardware Notes

| Mode | Requirements |
|------|-------------|
| OpenAI | Internet connection only |
| Local Llama (CPU) | 8GB+ RAM, slow |
| Local Llama (GPU) | 6GB+ VRAM |
| DPO Training | GPU with 8GB+ VRAM recommended |

## License

For personal and educational use.
