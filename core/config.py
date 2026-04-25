"""Configuration and environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration."""

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face token for gated models

    # Auth / DB settings
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/users.db")
    JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
    JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))
    # Message encryption key — must be a 32-byte URL-safe base64 string.
    # Generate one with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    # If unset in dev a deterministic fallback is derived from JWT_SECRET (NOT safe for production).
    DB_ENCRYPTION_KEY = os.getenv("DB_ENCRYPTION_KEY", "")

    # Redis cache
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    CONTEXT_CACHE_TTL = int(os.getenv("CONTEXT_CACHE_TTL", str(60 * 60 * 24)))  # 24h

    # Google Cloud
    GCP_PROJECT = os.getenv("GCP_PROJECT", "")
    GCS_BUCKET = os.getenv("GCS_BUCKET", "")                          # e.g. medchat-assets-prod
    CLOUD_SQL_CONNECTION_NAME = os.getenv("CLOUD_SQL_CONNECTION_NAME", "")  # project:region:instance

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

    # Neo4j settings
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

    @classmethod
    def validate(cls, require_openai: bool = True):
        """Validate required configuration."""
        if require_openai and not cls.USE_LOCAL_MODEL and not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Run: export OPENAI_API_KEY='your-key'"
            )
