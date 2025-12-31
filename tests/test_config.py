"""Unit tests for the config module.

Tests configuration loading from environment variables:
- Default values
- Environment variable overrides
- Boolean parsing for CHUNK_ENABLED
- Provider selection
"""

from unittest.mock import patch

import pytest

from src.config import Config, load_config


class TestConfig:
    """Tests for the Config model."""

    def test_default_llm_provider(self) -> None:
        """Default LLM provider should be 'openai'."""
        config = Config()
        assert config.llm_provider == "openai"

    def test_default_ollama_model(self) -> None:
        """Default Ollama model should be 'llama3:latest'."""
        config = Config()
        assert config.ollama_model == "llama3:latest"

    def test_default_ollama_base_url(self) -> None:
        """Default Ollama base URL should be localhost."""
        config = Config()
        assert config.ollama_base_url == "http://localhost:11434"

    def test_default_gemini_model(self) -> None:
        """Default Gemini model should be 'gemini-2.0-flash'."""
        config = Config()
        assert config.gemini_model == "gemini-2.0-flash"

    def test_default_chunk_enabled(self) -> None:
        """Chunking should be enabled by default."""
        config = Config()
        assert config.chunk_enabled is True

    def test_default_chunk_size(self) -> None:
        """Default chunk size should be 4000 tokens."""
        config = Config()
        assert config.chunk_size_tokens == 4000

    def test_default_chunk_overlap(self) -> None:
        """Default chunk overlap should be 200 tokens."""
        config = Config()
        assert config.chunk_overlap_tokens == 200

    def test_api_keys_default_to_none(self) -> None:
        """API keys should default to None."""
        config = Config()
        assert config.openai_api_key is None
        assert config.gemini_api_key is None

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields should raise validation error."""
        with pytest.raises(Exception):
            Config(unknown_field="value")  # type: ignore[call-arg]


class TestLoadConfig:
    """Tests for the load_config function."""

    @patch.dict("os.environ", {}, clear=True)
    def test_load_config_with_defaults(self) -> None:
        """load_config should return config with defaults when no env vars set."""
        config = load_config()

        assert config.llm_provider == "openai"
        assert config.ollama_model == "llama3:latest"
        assert config.gemini_model == "gemini-2.0-flash"
        assert config.chunk_enabled is True

    @patch.dict(
        "os.environ",
        {"LLM_PROVIDER": "ollama"},
        clear=True,
    )
    def test_load_config_llm_provider(self) -> None:
        """load_config should read LLM_PROVIDER from environment."""
        config = load_config()

        assert config.llm_provider == "ollama"

    @patch.dict(
        "os.environ",
        {"LLM_PROVIDER": "gemini"},
        clear=True,
    )
    def test_load_config_gemini_provider(self) -> None:
        """load_config should read gemini as LLM_PROVIDER."""
        config = load_config()

        assert config.llm_provider == "gemini"

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "sk-test-key-123"},
        clear=True,
    )
    def test_load_config_openai_api_key(self) -> None:
        """load_config should read OPENAI_API_KEY from environment."""
        config = load_config()

        assert config.openai_api_key == "sk-test-key-123"

    @patch.dict(
        "os.environ",
        {"GEMINI_API_KEY": "gemini-key-456"},
        clear=True,
    )
    def test_load_config_gemini_api_key(self) -> None:
        """load_config should read GEMINI_API_KEY from environment."""
        config = load_config()

        assert config.gemini_api_key == "gemini-key-456"

    @patch.dict(
        "os.environ",
        {"OLLAMA_MODEL": "mistral:latest"},
        clear=True,
    )
    def test_load_config_ollama_model(self) -> None:
        """load_config should read OLLAMA_MODEL from environment."""
        config = load_config()

        assert config.ollama_model == "mistral:latest"

    @patch.dict(
        "os.environ",
        {"OLLAMA_BASE_URL": "http://custom:8080"},
        clear=True,
    )
    def test_load_config_ollama_base_url(self) -> None:
        """load_config should read OLLAMA_BASE_URL from environment."""
        config = load_config()

        assert config.ollama_base_url == "http://custom:8080"

    @patch.dict(
        "os.environ",
        {"GEMINI_MODEL": "gemini-1.5-pro"},
        clear=True,
    )
    def test_load_config_gemini_model(self) -> None:
        """load_config should read GEMINI_MODEL from environment."""
        config = load_config()

        assert config.gemini_model == "gemini-1.5-pro"

    @patch.dict(
        "os.environ",
        {"CHUNK_ENABLED": "false"},
        clear=True,
    )
    def test_load_config_chunk_enabled_false(self) -> None:
        """load_config should parse CHUNK_ENABLED=false as boolean False."""
        config = load_config()

        assert config.chunk_enabled is False

    @patch.dict(
        "os.environ",
        {"CHUNK_ENABLED": "true"},
        clear=True,
    )
    def test_load_config_chunk_enabled_true(self) -> None:
        """load_config should parse CHUNK_ENABLED=true as boolean True."""
        config = load_config()

        assert config.chunk_enabled is True

    @patch.dict(
        "os.environ",
        {"CHUNK_ENABLED": "1"},
        clear=True,
    )
    def test_load_config_chunk_enabled_one(self) -> None:
        """load_config should parse CHUNK_ENABLED=1 as boolean True."""
        config = load_config()

        assert config.chunk_enabled is True

    @patch.dict(
        "os.environ",
        {"CHUNK_ENABLED": "yes"},
        clear=True,
    )
    def test_load_config_chunk_enabled_yes(self) -> None:
        """load_config should parse CHUNK_ENABLED=yes as boolean True."""
        config = load_config()

        assert config.chunk_enabled is True

    @patch.dict(
        "os.environ",
        {"CHUNK_ENABLED": "0"},
        clear=True,
    )
    def test_load_config_chunk_enabled_zero(self) -> None:
        """load_config should parse CHUNK_ENABLED=0 as boolean False."""
        config = load_config()

        assert config.chunk_enabled is False

    @patch.dict(
        "os.environ",
        {"CHUNK_SIZE_TOKENS": "8000"},
        clear=True,
    )
    def test_load_config_chunk_size(self) -> None:
        """load_config should read CHUNK_SIZE_TOKENS from environment."""
        config = load_config()

        assert config.chunk_size_tokens == 8000

    @patch.dict(
        "os.environ",
        {"CHUNK_OVERLAP_TOKENS": "500"},
        clear=True,
    )
    def test_load_config_chunk_overlap(self) -> None:
        """load_config should read CHUNK_OVERLAP_TOKENS from environment."""
        config = load_config()

        assert config.chunk_overlap_tokens == 500

    @patch.dict(
        "os.environ",
        {
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "key",
            "GEMINI_MODEL": "gemini-1.5-flash",
            "CHUNK_ENABLED": "false",
            "CHUNK_SIZE_TOKENS": "2000",
        },
        clear=True,
    )
    def test_load_config_all_values(self) -> None:
        """load_config should read all environment variables correctly."""
        config = load_config()

        assert config.llm_provider == "gemini"
        assert config.gemini_api_key == "key"
        assert config.gemini_model == "gemini-1.5-flash"
        assert config.chunk_enabled is False
        assert config.chunk_size_tokens == 2000

    @patch.dict(
        "os.environ",
        {"CHUNK_ENABLED": "FALSE"},
        clear=True,
    )
    def test_load_config_chunk_enabled_case_insensitive(self) -> None:
        """CHUNK_ENABLED parsing should be case-insensitive."""
        config = load_config()

        assert config.chunk_enabled is False

    @patch.dict(
        "os.environ",
        {"CHUNK_ENABLED": "TRUE"},
        clear=True,
    )
    def test_load_config_chunk_enabled_uppercase_true(self) -> None:
        """CHUNK_ENABLED=TRUE should be parsed as True."""
        config = load_config()

        assert config.chunk_enabled is True

