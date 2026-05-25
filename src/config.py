"""
Central configuration — loaded once from .env via pydantic-settings.
Import `settings` anywhere in the codebase.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Vector store
    vector_store_path: str = "./data/vector_store"
    collection_name: str = "rag_documents"

    # Retrieval
    top_k_retrieval: int = 5

    # Pipeline
    max_retries: int = 3

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Logging
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
