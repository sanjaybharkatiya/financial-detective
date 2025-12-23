"""Factory for creating LLM extractor instances.

This module provides a factory function that creates the appropriate
LLM extractor based on application configuration.
"""

from src.config import load_config
from src.extractor.base import LLMExtractor
from src.extractor.gemini_llm import GeminiExtractor
from src.extractor.ollama_llm import OllamaExtractor
from src.extractor.openai_llm import OpenAIExtractor


def create_extractor() -> LLMExtractor:
    """Create an LLM extractor based on configuration.

    Reads the LLM_PROVIDER from environment configuration and returns
    the appropriate extractor instance.

    Returns:
        An LLMExtractor instance configured for the selected provider.

    Raises:
        ValueError: If the configured LLM provider is not supported.
    """
    config = load_config()

    if config.llm_provider == "openai":
        return OpenAIExtractor(api_key=config.openai_api_key)

    if config.llm_provider == "ollama":
        return OllamaExtractor(
            model=config.ollama_model,
            base_url=config.ollama_base_url,
        )

    if config.llm_provider == "gemini":
        return GeminiExtractor(
            api_key=config.gemini_api_key,
            model=config.gemini_model,
        )

    raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")

