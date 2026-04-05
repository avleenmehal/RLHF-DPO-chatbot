# Medical Chatbot with RAG, GraphRAG, and DPO Training

A medical chatbot that uses Retrieval-Augmented Generation (RAG), a Neo4j knowledge graph (GraphRAG), and live web search to provide context-aware responses, with support for fine-tuning via Direct Preference Optimization (DPO).

## Features

- **RAG Pipeline**: Semantic search over a local medical Q&A dataset via FAISS
- **GraphRAG**: Neo4j knowledge graph of medical concepts and their relationships — answers relational queries like "what conditions are associated with X"
- **Web Search**: Falls back to live DuckDuckGo search when local knowledge is insufficient
- **Tool-Calling Agent**: Automatically selects the best tool (RAG, GraphRAG, or web search) per query
- **Multiple LLM Support**: OpenAI GPT-4.1 or local Llama models
- **Preference Collection**: Built-in system to collect human preferences for DPO training
- **DPO Training**: Fine-tune Llama models using collected preferences
- **Conversation Memory**: Maintains chat history for context-aware responses

## Project Structure

```
ChatbotMedical/
├── config.py                 # Configuration settings (models, paths, Neo4j)
├── llm.py                    # LLM manager (OpenAI/Local)
├── llm_local.py              # Local Llama model loader with LoRA support
├── rag.py                    # RAG pipeline (FAISS vector store)
├── graph_builder.py          # Builds Neo4j knowledge graph from CSV data
├── graph_retrieval.py        # GraphRAG retrieval pipeline
├── chatbot.py                # Tool-calling agent (RAG + GraphRAG + web search)
├── app.py                    # Gradio web UI
├── main.py                   # CLI entry point
├── preference_collector.py   # Preference data collection
├── dpo_trainer.py            # DPO training script
├── requirements.txt          # Dependencies
├── data/
│   ├── train_data_chatbot_small.csv  # Medical Q&A dataset (3.3MB, default)
│   ├── train_data_chatbot.csv        # Larger dataset (34MB)
│   ├── preferences.jsonl             # Collected DPO preference pairs
│   └── vector_store/                 # FAISS vector store (auto-created)
└── models/
    └── dpo_medical_chatbot/          # Trained LoRA adapter output
```

## Setup

### 1. Create and Activate a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

```bash
# Required for OpenAI
export OPENAI_API_KEY="sk-your-key-here"

# Required for local Llama models
export HF_TOKEN="your-huggingface-token"

# Required for GraphRAG
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your-neo4j-password"
```

### 4. (Optional) Set Up Neo4j and Build the Knowledge Graph

GraphRAG requires a running Neo4j instance. Install [Neo4j Desktop](https://neo4j.com/download/) or use Docker:

```bash
docker run -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-password \
  neo4j:latest
```

Then build the knowledge graph from the CSV data:

```bash
python3 graph_builder.py
```

This creates:
- **QAPair** nodes — one per doctor-patient Q&A (label=1.0 rows only)
- **MedicalConcept** nodes — one per unique medical tag
- **TAGGED_WITH** edges — linking Q&A pairs to their concepts
- **CO_OCCURS_WITH** edges — linking concepts that appear together (with co-occurrence weight)

### 5. Web Search

Web search uses DuckDuckGo — no API key required. To verify it works:

```bash
python3 -c "from langchain_community.tools import DuckDuckGoSearchRun; print(DuckDuckGoSearchRun().run('common cold symptoms'))"
```

### 6. (Optional) Login to Hugging Face

Required only if using local Llama models:

```bash
huggingface-cli login
```

## Usage

### Web UI

```bash
python app.py
```

Then open `http://localhost:7860` in your browser.

### Basic Chat (OpenAI)

```bash
python3 main.py
```

### Choose Model

```bash
# OpenAI GPT-4.1 (default)
python3 main.py --model openai

# Local Llama base model
python3 main.py --model local

# Local Llama with DPO fine-tuning
python3 main.py --model dpo
```

### Collect Preferences for Training

```bash
python3 main.py --collect-preferences
```

In this mode:
1. Ask a question
2. Bot generates 2 responses
3. Pick the better one (0 or 1)
4. Preference saved to `data/preferences.jsonl`

### Chat Commands

- `quit` — Exit the chatbot
- `clear` — Clear conversation history
- `stats` — View collected preferences count (preference mode only)

## How the Agent Selects Tools

The LangChain agent follows this decision guide:

| Query type | Tool used |
|---|---|
| Factual ("what is X", "symptoms of X", "how is X treated") | `rag_retrieval` first |
| Relational ("what's associated with X", "what co-occurs with X") | `graph_retrieval` first |
| Neither local tool has sufficient results | `web_search` |

## GraphRAG Architecture

`graph_builder.py` reads the medical CSV and populates Neo4j with:

```
(QAPair) -[:TAGGED_WITH]-> (MedicalConcept)
(MedicalConcept) -[:CO_OCCURS_WITH {weight}]- (MedicalConcept)
```

`graph_retrieval.py` (`GraphRAGPipeline`) exposes three methods:
- `retrieve_by_concept(concept)` — finds Q&A pairs tagged with a concept
- `retrieve_related_concepts(concept)` — finds co-occurring concepts ranked by weight
- `retrieve(query)` — main entry point; extracts the best-matching concept from the query, runs both lookups, and formats a single context string for the LLM

## DPO Training Workflow

### Step 1: Collect Preferences

```bash
python3 main.py --collect-preferences
```

Aim for 100+ preference pairs for meaningful training.

### Step 2: Run DPO Training

```bash
python3 dpo_trainer.py
```

Options:
```bash
python3 dpo_trainer.py --model meta-llama/Llama-3.2-1B-Instruct \
                       --preferences data/preferences.jsonl \
                       --output models/dpo_medical_chatbot
```

Uses 4-bit quantization + LoRA via HuggingFace TRL's `DPOTrainer`.

### Step 3: Use Trained Model

```bash
python3 main.py --model dpo
```

## Configuration

Edit `config.py` to customize:

| Setting | Description | Default |
|---------|-------------|---------|
| `LLM_MODEL` | OpenAI model | `gpt-4.1` |
| `LOCAL_BASE_MODEL` | Llama model | `meta-llama/Llama-3.2-1B-Instruct` |
| `CHUNK_SIZE` | RAG chunk size | `1000` |
| `TOP_K_RESULTS` | Retrieved docs per query | `3` |
| `NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7687` |
| `NEO4J_USERNAME` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | *(from env)* |

## Requirements

- Python 3.9+
- OpenAI API key (for OpenAI mode)
- Hugging Face account (for local Llama)
- Neo4j instance (for GraphRAG)
- GPU recommended for local models and DPO training

## Hardware Notes

| Mode | Requirements |
|------|--------------|
| OpenAI | Internet connection |
| Local Llama (CPU) | 8GB+ RAM, slower |
| Local Llama (GPU) | 6GB+ VRAM |
| DPO Training | GPU with 8GB+ VRAM recommended |

## License

For personal/educational use.
