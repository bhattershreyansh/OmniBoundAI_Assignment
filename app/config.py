from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    LLM_MODEL: str = "gpt-4.1-mini-2025-04-14"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_TIMEOUT_SECONDS: float = 30.0
    
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "omnibound-index"

    FETCH_TIMEOUT_SECONDS: float = 15.0
    FETCH_MAX_BYTES: int = 5_000_000
    TOP_K: int = 5
    CHUNK_SIZE_TOKENS: int = 400
    CHUNK_OVERLAP_TOKENS: int = 50
    
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ENVIRONMENT: str = "development"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
