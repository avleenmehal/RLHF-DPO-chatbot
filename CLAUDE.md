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

## Running the Application

| Command | Purpose |
|---------|---------|
| `python app.py` | Gradio web UI at http://localhost:7860 |
| `python3 main.py` | CLI chat with OpenAI GPT-4.1 |
| `python3 main.py --model local` | CLI chat with Llama base model |
| `python3 main.py --model dpo` | CLI chat with DPO fine-tuned model |
| `python3 main.py --collect-preferences` | Preference collection mode (generates 2 responses per query for user to rank) |
| `python3 dpo_trainer.py` | Run DPO fine-tuning on collected preferences |

**CLI interactive commands:** `quit`, `clear`, `stats` (preference mode only)

## Architecture

This is a medical Q&A chatbot with RAG + web search, multi-model support, and a DPO training pipeline.

### Data Flow
User query → `MedicalChatbot` (chatbot.py) → LangChain agent selects tool → RAG or web search → context + query → LLM → response

The agent uses two tools:
1. **rag_retrieval** — semantic search over local medical CSV data via FAISS
2. **web_search** — DuckDuckGo fallback when local knowledge is insufficient

### Key Modules

- **chatbot.py** — `MedicalChatbot` class; LangChain agent orchestration, conversation history, preference collection mode
- **rag.py** — `RAGPipeline`; loads CSV data, chunks, embeds via OpenAI, stores/retrieves via FAISS
- **llm.py** — `LLMManager` singleton; factory for OpenAI or local Llama models; always uses OpenAI embeddings
- **llm_local.py** — `LocalLLM` + `LocalLLMWrapper`; Llama loading with optional LoRA adapter, LangChain integration
- **config.py** — All configuration (model names, paths, chunk sizes); reads from environment variables
- **preference_collector.py** — Appends chosen/rejected response pairs to `data/preferences.jsonl`
- **dpo_trainer.py** — 4-bit quantization + LoRA fine-tuning via HuggingFace TRL's `DPOTrainer`
- **app.py** — Gradio web interface (single-turn, uses OpenAI + RAG only)

### Configuration Defaults (config.py)

```
LLM_MODEL = "gpt-4.1"
EMBEDDING_MODEL = "text-embedding-3-small"
LOCAL_BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
LOCAL_ADAPTER_PATH = "models/dpo_medical_chatbot"
CSV_PATH = "data/train_data_chatbot_small.csv"
VECTOR_STORE_PATH = "data/vector_store"
PREFERENCES_PATH = "data/preferences.jsonl"
CHUNK_SIZE = 1000, CHUNK_OVERLAP = 200, TOP_K_RESULTS = 3
```

### Training Workflow

1. Collect preference pairs: `python3 main.py --collect-preferences` (user picks better of 2 responses → saved to `data/preferences.jsonl`)
2. Fine-tune: `python3 dpo_trainer.py` → saves LoRA adapter to `models/dpo_medical_chatbot/`
3. Use trained model: `python3 main.py --model dpo`

### Data Files

- `data/train_data_chatbot_small.csv` — primary medical Q&A dataset (3.3MB, used by default)
- `data/train_data_chatbot.csv` — larger version (34MB)
- `data/vector_store/` — cached FAISS index (auto-created on first run)
- `data/preferences.jsonl` — DPO training data

### Hardware Notes

- **OpenAI mode**: internet only
- **Local Llama (CPU)**: 8GB+ RAM, slow
- **DPO training**: GPU with 8GB+ VRAM recommended
