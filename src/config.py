"""Configuration module for Financial Detective.

This module provides configuration management using Pydantic models.
All configuration values are read from environment variables.
Supports loading from .env file via python-dotenv.
"""

import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

# Load environment variables from .env file (if present)
load_dotenv()


class Config(BaseModel):
    """Application configuration loaded from environment variables.

    Attributes:
        llm_provider: The LLM provider to use. Must be "openai", "ollama", or "gemini".
        openai_api_key: API key for OpenAI. Required when llm_provider is "openai".
        ollama_model: Model name for Ollama. Defaults to "llama3:latest".
        ollama_base_url: Base URL for Ollama API. Defaults to "http://localhost:11434".
        gemini_api_key: API key for Google Gemini. Required when llm_provider is "gemini".
        gemini_model: Model name for Gemini. Defaults to "gemini-1.5-pro".
        chunk_enabled: Whether to enable document chunking for large texts.
        chunk_size_tokens: Target number of tokens per chunk.
        chunk_overlap_tokens: Number of tokens to overlap between chunks.
    """

    model_config = ConfigDict(extra="forbid")

    llm_provider: Literal["openai", "ollama", "gemini"] = Field(
        default="openai",
        description="LLM provider to use for extraction",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key (required for openai provider)",
    )
    ollama_model: str = Field(
        default="llama3:latest",
        description="Ollama model name",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )
    gemini_api_key: str | None = Field(
        default=None,
        description="Google Gemini API key (required for gemini provider)",
    )
    gemini_model: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model name",
    )
    chunk_enabled: bool = Field(
        default=True,
        description="Enable document chunking for large texts",
    )
    chunk_size_tokens: int = Field(
        default=4000,
        description="Target number of tokens per chunk",
        gt=0,
    )
    chunk_overlap_tokens: int = Field(
        default=200,
        description="Number of tokens to overlap between chunks",
        ge=0,
    )


def load_config() -> Config:
    """Load configuration from environment variables.

    Reads the following environment variables:
    - LLM_PROVIDER: "openai", "ollama", or "gemini" (default: "openai")
    - OPENAI_API_KEY: API key for OpenAI
    - OLLAMA_MODEL: Model name for Ollama (default: "llama3:latest")
    - OLLAMA_BASE_URL: Ollama API URL (default: "http://localhost:11434")
    - GEMINI_API_KEY: API key for Google Gemini
    - GEMINI_MODEL: Model name for Gemini (default: "gemini-2.0-flash")
    - CHUNK_ENABLED: Enable document chunking (default: "true")
    - CHUNK_SIZE_TOKENS: Target tokens per chunk (default: 4000)
    - CHUNK_OVERLAP_TOKENS: Overlap tokens between chunks (default: 200)

    Returns:
        A validated Config instance.

    Raises:
        pydantic.ValidationError: If environment values fail validation.
    """
    # Parse CHUNK_ENABLED as boolean
    chunk_enabled_str = os.environ.get("CHUNK_ENABLED", "true").lower()
    chunk_enabled = chunk_enabled_str in ("true", "1", "yes")

    return Config(
        llm_provider=os.environ.get("LLM_PROVIDER", "openai"),  # type: ignore[arg-type]
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "llama3:latest"),
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        chunk_enabled=chunk_enabled,
        chunk_size_tokens=int(os.environ.get("CHUNK_SIZE_TOKENS", "4000")),
        chunk_overlap_tokens=int(os.environ.get("CHUNK_OVERLAP_TOKENS", "200")),
    )

