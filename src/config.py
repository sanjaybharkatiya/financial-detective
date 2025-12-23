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


def load_config() -> Config:
    """Load configuration from environment variables.

    Reads the following environment variables:
    - LLM_PROVIDER: "openai", "ollama", or "gemini" (default: "openai")
    - OPENAI_API_KEY: API key for OpenAI
    - OLLAMA_MODEL: Model name for Ollama (default: "llama3:latest")
    - OLLAMA_BASE_URL: Ollama API URL (default: "http://localhost:11434")
    - GEMINI_API_KEY: API key for Google Gemini
    - GEMINI_MODEL: Model name for Gemini (default: "gemini-2.0-flash")

    Returns:
        A validated Config instance.

    Raises:
        pydantic.ValidationError: If environment values fail validation.
    """
    return Config(
        llm_provider=os.environ.get("LLM_PROVIDER", "openai"),  # type: ignore[arg-type]
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "llama3:latest"),
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
    )

