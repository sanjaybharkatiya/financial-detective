"""Unit tests for the extractor factory module.

Tests create_extractor function:
- Returns OpenAIExtractor when LLM_PROVIDER=openai
- Returns OllamaExtractor when LLM_PROVIDER=ollama
- Uses mocked environment variables (no real API keys or services required)
"""

from unittest.mock import patch

import pytest

from src.extractor.factory import create_extractor
from src.extractor.ollama_llm import OllamaExtractor
from src.extractor.openai_llm import OpenAIExtractor


class TestCreateExtractor:
    """Tests for create_extractor factory function."""

    @patch.dict(
        "os.environ",
        {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test-api-key"},
        clear=True,
    )
    def test_openai_provider_returns_openai_extractor(self) -> None:
        """LLM_PROVIDER=openai should return an OpenAIExtractor instance."""
        extractor = create_extractor()

        assert isinstance(extractor, OpenAIExtractor)

    @patch.dict(
        "os.environ",
        {"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": "llama3"},
        clear=True,
    )
    def test_ollama_provider_returns_ollama_extractor(self) -> None:
        """LLM_PROVIDER=ollama should return an OllamaExtractor instance."""
        extractor = create_extractor()

        assert isinstance(extractor, OllamaExtractor)

    @patch.dict(
        "os.environ",
        {"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": "mistral", "OLLAMA_BASE_URL": "http://custom:11434"},
        clear=True,
    )
    def test_ollama_extractor_uses_config_values(self) -> None:
        """OllamaExtractor should be configured with environment values."""
        extractor = create_extractor()

        assert isinstance(extractor, OllamaExtractor)
        assert extractor.model == "mistral"
        assert extractor.base_url == "http://custom:11434"

    @patch.dict(
        "os.environ",
        {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "my-secret-key"},
        clear=True,
    )
    def test_openai_extractor_uses_api_key(self) -> None:
        """OpenAIExtractor should be configured with the API key."""
        extractor = create_extractor()

        assert isinstance(extractor, OpenAIExtractor)
        assert extractor.api_key == "my-secret-key"

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "test-key"},
        clear=True,
    )
    def test_defaults_to_openai_provider(self) -> None:
        """When LLM_PROVIDER is not set, should default to openai."""
        extractor = create_extractor()

        assert isinstance(extractor, OpenAIExtractor)

    @patch.dict(
        "os.environ",
        {"LLM_PROVIDER": "openai"},
        clear=True,
    )
    def test_openai_without_api_key_raises_error(self) -> None:
        """OpenAI provider without API key should raise ValueError."""
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            create_extractor()

