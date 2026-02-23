"""Configuration and environment variables."""

import os


class Config:
    """Application configuration."""

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face token for gated models

    # Model settings
    LLM_MODEL = "gpt-4.1"
    EMBEDDING_MODEL = "text-embedding-3-small"
    TEMPERATURE = 0.7

    # Local model settings (for DPO-trained model)
    LOCAL_BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
    LOCAL_ADAPTER_PATH = "models/dpo_medical_chatbot"
    USE_LOCAL_MODEL = False  # Set to True after DPO training

    # RAG settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RESULTS = 3

    # Data paths
    CSV_PATH = "data/train_data_chatbot_small.csv"
    VECTOR_STORE_PATH = "data/vector_store"
    PREFERENCES_PATH = "data/preferences.jsonl"

    @classmethod
    def validate(cls, require_openai: bool = True):
        """Validate required configuration."""
        if require_openai and not cls.USE_LOCAL_MODEL and not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Run: export OPENAI_API_KEY='your-key'"
            )
