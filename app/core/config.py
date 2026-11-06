"""
Centralized application configuration.

All settings are loaded from environment variables (or a local .env file).
No secrets are hardcoded here — see .env.example for the expected keys.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application metadata ---
    app_name: str = "DevPilot"
    environment: str = "development"
    debug: bool = False

    # --- API server ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- GitHub (read-only repository access) ---
    # Optional: without a token, GitHub's unauthenticated rate limit (60 req/hour) applies
    # and only public repositories are accessible. With a token, private repos work and
    # the limit rises to 5000 req/hour. Token only needs read access — no write/admin scopes.
    github_token: Optional[str] = Field(default=None)
    github_request_timeout_seconds: float = Field(default=10.0)
    github_max_file_size_bytes: int = Field(default=500_000)
    github_max_files: int = Field(default=500)
    github_max_concurrent_requests: int = Field(default=10)

    # --- Google GenAI ---
    google_api_key: Optional[str] = Field(default=None)
    google_embedding_model: str = Field(default="gemini-embedding-001")
    # gemini-2.5-flash turned out to be retired for new API keys (confirmed via a
    # live 404 from the Gemini API, which pointed to this replacement). Override
    # via GOOGLE_CHAT_MODEL if Google ships a newer recommended model later.
    google_chat_model: str = Field(default="gemini-3.8-flash")
    # Low by default: this assistant answers grounded questions about a specific
    # repository, where factual accuracy matters more than creative variation.
    google_temperature: float = Field(default=0.2)

    # --- Pinecone ---
    pinecone_api_key: Optional[str] = Field(default=None)
    pinecone_index_name: str = Field(default="devpilot-code")
    pinecone_cloud: str = Field(default="aws")
    pinecone_region: str = Field(default="us-east-1")

    # --- Chunking ---
    chunk_size: int = Field(default=1500, description="Target chunk size in characters.")
    chunk_overlap: int = Field(default=200, description="Overlap between consecutive chunks, in characters.")

    # --- Agent (tool-calling loop) ---
    # Counts graph super-steps (agent + tools nodes), not tool calls directly — this
    # caps runaway loops (e.g. the model repeatedly calling tools without converging)
    # while still allowing several sequential tool calls for multi-step questions.
    agent_recursion_limit: int = Field(default=15)

    # --- Bug investigation (Milestone 6) ---
    # Outer bound on verify_evidence -> continue_investigation -> investigate_agent
    # cycles — independent of the inner ReAct tool-call loop, so an investigation
    # that can't converge stops with an honest "inconclusive" report instead of
    # looping indefinitely.
    investigation_max_rounds: int = Field(default=3)
    # Graph-level safety net (covers both the inner ReAct loop and the outer
    # verification loop across all rounds); higher than agent_recursion_limit
    # since this graph has more stages and can legitimately run longer.
    investigation_recursion_limit: int = Field(default=60)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance so the .env file is parsed once."""
    return Settings()
