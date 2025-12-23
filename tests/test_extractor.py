"""Unit tests for the extractor module.

Tests extract_knowledge_graph function with mocked factory:
- Verifies delegation to LLMExtractor.extract
- Verifies KnowledgeGraph is returned unchanged
- Tests are provider-agnostic (no OpenAI/Ollama mocking)
"""

from unittest.mock import MagicMock, patch

import pytest

from src.extractor import extract_knowledge_graph
from src.schema import KnowledgeGraph, Node, Relationship


class TestExtractKnowledgeGraph:
    """Tests for extract_knowledge_graph function."""

    @patch("src.extractor.create_extractor")
    def test_delegates_to_extractor(self, mock_create_extractor: MagicMock) -> None:
        """extract_knowledge_graph should delegate to extractor.extract."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        expected_graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="company_1", type="Company", name="Acme Corp"),
            ],
            relationships=[],
        )
        mock_extractor.extract.return_value = expected_graph

        result = extract_knowledge_graph("Sample financial text")

        mock_create_extractor.assert_called_once()
        mock_extractor.extract.assert_called_once_with("Sample financial text")
        assert result is expected_graph

    @patch("src.extractor.create_extractor")
    def test_returns_knowledge_graph_unchanged(self, mock_create_extractor: MagicMock) -> None:
        """KnowledgeGraph from extractor should be returned unchanged."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        expected_graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="company_1", type="Company", name="Acme Corp"),
                Node(id="risk_1", type="RiskFactor", name="Market Volatility"),
                Node(id="amount_1", type="DollarAmount", name="$1,000,000"),
            ],
            relationships=[
                Relationship(source="company_1", target="risk_1", relation="HAS_RISK"),
                Relationship(source="company_1", target="amount_1", relation="REPORTS_AMOUNT"),
            ],
        )
        mock_extractor.extract.return_value = expected_graph

        result = extract_knowledge_graph("Financial report text")

        assert result.schema_version == "1.0.0"
        assert len(result.nodes) == 3
        assert len(result.relationships) == 2
        assert result.nodes[0].name == "Acme Corp"
        assert result.relationships[0].relation == "HAS_RISK"

    @patch("src.extractor.create_extractor")
    def test_propagates_extractor_exceptions(self, mock_create_extractor: MagicMock) -> None:
        """Exceptions from extractor.extract should propagate."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor
        mock_extractor.extract.side_effect = ValueError("LLM returned empty response")

        with pytest.raises(ValueError, match="LLM returned empty response"):
            extract_knowledge_graph("Sample text")

    @patch("src.extractor.create_extractor")
    def test_propagates_factory_exceptions(self, mock_create_extractor: MagicMock) -> None:
        """Exceptions from create_extractor should propagate."""
        mock_create_extractor.side_effect = ValueError("OPENAI_API_KEY environment variable is not set")

        with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is not set"):
            extract_knowledge_graph("Sample text")

    @patch("src.extractor.create_extractor")
    def test_passes_text_to_extractor(self, mock_create_extractor: MagicMock) -> None:
        """Input text should be passed to extractor.extract unchanged."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor
        mock_extractor.extract.return_value = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c1", type="Company", name="Test")],
            relationships=[],
        )

        input_text = "Reliance Industries reported revenue of ₹9,500 crore."
        extract_knowledge_graph(input_text)

        mock_extractor.extract.assert_called_once_with(input_text)
