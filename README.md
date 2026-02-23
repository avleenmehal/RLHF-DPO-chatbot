# Medical Chatbot with RAG and DPO Training

A medical chatbot that uses Retrieval-Augmented Generation (RAG) and live web search to provide context-aware responses, with support for training via Direct Preference Optimization (DPO).

## Features

- **RAG Pipeline**: Retrieves relevant doctor-patient conversations to provide contextual answers
- **Web Search**: Falls back to live DuckDuckGo search when local knowledge is insufficient
- **Tool-Calling Agent**: Automatically decides when to use RAG vs web search per query
- **Multiple LLM Support**: OpenAI GPT-4.1 or local Llama models
- **Preference Collection**: Built-in system to collect human preferences for training
- **DPO Training**: Fine-tune Llama models using collected preferences
- **Conversation Memory**: Maintains chat history for context-aware responses

## Project Structure

```
ChatbotMedical/
├── config.py                 # Configuration settings
├── llm.py                    # LLM manager (OpenAI/Local)
├── llm_local.py              # Local Llama model loader
├── rag.py                    # RAG pipeline
├── chatbot.py                # Tool-calling agent with RAG + web search
├── app.py                    # Gradio web UI
├── main.py                   # CLI entry point
├── preference_collector.py   # Preference data collection
├── dpo_trainer.py            # DPO training script
├── requirements.txt          # Dependencies
├── data/
│   ├── train_data_chatbot_small.csv  # Medical conversations
│   ├── preferences.jsonl             # Collected preferences
│   └── vector_store/                 # FAISS vector store
└── models/
    └── dpo_medical_chatbot/          # Trained model output
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
```

### 4. Web Search

Web search uses DuckDuckGo and works out of the box — no API key required. The agent automatically searches the web when the local RAG knowledge base does not contain a sufficient answer.

To verify it is working:

```bash
python3 -c "from langchain_community.tools import DuckDuckGoSearchRun; print(DuckDuckGoSearchRun().run('common cold symptoms'))"
```

### 5. (Optional) Login to Hugging Face

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

# Local Llama with DPO training
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

- `quit` - Exit the chatbot
- `clear` - Clear conversation history
- `stats` - View collected preferences count (in preference mode)

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
| `TOP_K_RESULTS` | Number of retrieved docs | `3` |

## Requirements

- Python 3.9+ (3.14 recommended)
- OpenAI API key (for OpenAI mode)
- Hugging Face account (for local Llama)
- GPU recommended for local models and training

## Hardware Notes

| Mode | Requirements |
|------|--------------|
| OpenAI | Internet connection |
| Local Llama (CPU) | 8GB+ RAM, slower |
| Local Llama (GPU) | 6GB+ VRAM |
| DPO Training | GPU with 8GB+ VRAM recommended |

## License

For personal/educational use.
