"""LLM initialization and management."""

from enum import Enum
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config import Config


class ModelType(Enum):
    """Available model types."""
    OPENAI = "openai"
    LOCAL_BASE = "local_base"      # Llama without DPO training
    LOCAL_DPO = "local_dpo"        # Llama with DPO training


class LLMManager:
    """Manages LLM and embedding models."""

    _llm = None
    _local_llm = None
    _embeddings = None
    _model_type = ModelType.OPENAI

    @classmethod
    def set_model_type(cls, model_type: ModelType):
        """Set which model to use."""
        cls._model_type = model_type
        cls._llm = None
        cls._local_llm = None

    @classmethod
    def get_llm(cls, model_type: ModelType = None):
        """Get or create the LLM instance."""
        if model_type is not None:
            cls._model_type = model_type

        if cls._model_type == ModelType.OPENAI:
            return cls._get_openai_llm()
        elif cls._model_type == ModelType.LOCAL_BASE:
            return cls._get_local_llm(use_adapter=False)
        elif cls._model_type == ModelType.LOCAL_DPO:
            return cls._get_local_llm(use_adapter=True)

    @classmethod
    def _get_openai_llm(cls) -> ChatOpenAI:
        """Get OpenAI LLM."""
        if cls._llm is None:
            Config.validate(require_openai=True)
            cls._llm = ChatOpenAI(
                model=Config.LLM_MODEL,
                temperature=Config.TEMPERATURE,
                api_key=Config.OPENAI_API_KEY,
            )
        return cls._llm

    @classmethod
    def _get_local_llm(cls, use_adapter: bool = False):
        """Get local LLM (base or DPO-trained)."""
        if cls._local_llm is None:
            from llm_local import LocalLLM, LocalLLMWrapper

            adapter_path = Config.LOCAL_ADAPTER_PATH if use_adapter else None
            model_desc = "DPO-trained" if use_adapter else "base"
            print(f"Loading local {model_desc} model: {Config.LOCAL_BASE_MODEL}")

            local = LocalLLM(
                base_model=Config.LOCAL_BASE_MODEL,
                adapter_path=adapter_path,
            )
            cls._local_llm = LocalLLMWrapper(local)
        return cls._local_llm

    @classmethod
    def get_embeddings(cls) -> OpenAIEmbeddings:
        """Get or create the embeddings instance."""
        if cls._embeddings is None:
            Config.validate(require_openai=True)
            cls._embeddings = OpenAIEmbeddings(
                model=Config.EMBEDDING_MODEL,
                api_key=Config.OPENAI_API_KEY,
            )
        return cls._embeddings
