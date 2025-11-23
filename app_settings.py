from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import AnyUrl, Field


class Settings(BaseSettings):
    database_url: str = Field("sqlite:////data/tasks.db", env="DATABASE_URL")
    checkpoints_path: str = Field("/data/checkpoints.sqlite", env="CHECKPOINTS_PATH")
    ollama_base_url: AnyUrl = Field("http://host.docker.internal:11434", env="OLLAMA_BASE_URL")
    ollama_reasoning_model: str = Field(
        "deepseek-r1:8b-llama-distill-q4_K_M", env="OLLAMA_REASONING_MODEL"
    )
    ollama_coding_model: str = Field(
        "qwen2.5-coder:7b-instruct-q6_K", env="OLLAMA_CODING_MODEL"
    )
    ollama_model: str | None = Field(
        "gemma3:4b", env="OLLAMA_MODEL"
    )
    perplexity_api_key: str | None = Field(default=None, env="PERPLEXITY_API_KEY")
    perplexity_api_url: AnyUrl = Field(
        "https://api.perplexity.ai/chat/completions", env="PERPLEXITY_API_URL"
    )
    run_locally: bool = Field(False, env="RUN_LOCALLY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[arg-type]
