"""Unit tests for the extractor module.

Tests extract_knowledge_graph function with mocked factory:
- Verifies delegation to LLMExtractor.extract
- Verifies KnowledgeGraph is returned unchanged
- Tests are provider-agnostic (no OpenAI/Ollama mocking)
- Tests chunking integration
"""

from unittest.mock import MagicMock, patch, call

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


class TestExtractKnowledgeGraphChunking:
    """Tests for extract_knowledge_graph with chunking enabled."""

    @patch("src.extractor.load_config")
    @patch("src.extractor.create_extractor")
    def test_small_text_not_chunked(
        self,
        mock_create_extractor: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Text smaller than chunk size should not be chunked."""
        mock_config = MagicMock()
        mock_config.chunk_enabled = True
        mock_config.chunk_size_tokens = 1000
        mock_config.chunk_overlap_tokens = 100
        mock_load_config.return_value = mock_config

        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        expected_graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c1", type="Company", name="Test")],
            relationships=[],
        )
        mock_extractor.extract.return_value = expected_graph

        # Small text (less than 1000 tokens = 4000 chars)
        result = extract_knowledge_graph("Short text")

        # Should call extract once (no chunking)
        mock_extractor.extract.assert_called_once()
        assert result is expected_graph

    @patch("src.extractor.load_config")
    @patch("src.extractor.create_extractor")
    def test_large_text_chunked(
        self,
        mock_create_extractor: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Text larger than chunk size should be chunked."""
        mock_config = MagicMock()
        mock_config.chunk_enabled = True
        mock_config.chunk_size_tokens = 50  # Small for testing (200 chars)
        mock_config.chunk_overlap_tokens = 10
        mock_load_config.return_value = mock_config

        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        # Return different graphs for each chunk
        graph1 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c1", type="Company", name="Company A")],
            relationships=[],
        )
        graph2 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c2", type="Company", name="Company B")],
            relationships=[],
        )
        mock_extractor.extract.side_effect = [graph1, graph2]

        # Large text (more than 200 chars = 50 tokens)
        large_text = "This is a large text. " * 50  # ~1100 chars

        result = extract_knowledge_graph(large_text)

        # Should call extract multiple times (chunking)
        assert mock_extractor.extract.call_count >= 2
        # Result should be merged graph
        assert len(result.nodes) >= 1

    @patch("src.extractor.load_config")
    @patch("src.extractor.create_extractor")
    def test_chunking_disabled(
        self,
        mock_create_extractor: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """When chunking is disabled, large text should not be chunked."""
        mock_config = MagicMock()
        mock_config.chunk_enabled = False  # Disabled
        mock_load_config.return_value = mock_config

        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        expected_graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c1", type="Company", name="Test")],
            relationships=[],
        )
        mock_extractor.extract.return_value = expected_graph

        # Large text
        large_text = "This is a large text. " * 100

        result = extract_knowledge_graph(large_text)

        # Should call extract once (no chunking)
        mock_extractor.extract.assert_called_once()
        assert result is expected_graph

    @patch("src.extractor.load_config")
    @patch("src.extractor.create_extractor")
    def test_failed_chunk_continues_processing(
        self,
        mock_create_extractor: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Failed chunks should be skipped, processing should continue."""
        mock_config = MagicMock()
        mock_config.chunk_enabled = True
        mock_config.chunk_size_tokens = 50
        mock_config.chunk_overlap_tokens = 10
        mock_load_config.return_value = mock_config

        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        # First chunk fails, second succeeds
        graph2 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c1", type="Company", name="Success")],
            relationships=[],
        )
        mock_extractor.extract.side_effect = [
            ValueError("First chunk failed"),
            graph2,
        ]

        large_text = "This is a large text. " * 50

        # Should not raise, should return partial result
        result = extract_knowledge_graph(large_text)

        assert len(result.nodes) >= 1
        assert result.nodes[0].name == "Success"

    @patch("src.extractor.load_config")
    @patch("src.extractor.create_extractor")
    def test_all_chunks_fail_raises_error(
        self,
        mock_create_extractor: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """If all chunks fail, should raise error."""
        mock_config = MagicMock()
        mock_config.chunk_enabled = True
        mock_config.chunk_size_tokens = 50
        mock_config.chunk_overlap_tokens = 10
        mock_load_config.return_value = mock_config

        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        # All chunks fail
        mock_extractor.extract.side_effect = ValueError("All failed")

        large_text = "This is a large text. " * 50

        with pytest.raises(ValueError, match="chunks failed"):
            extract_knowledge_graph(large_text)

    @patch("src.extractor.load_config")
    @patch("src.extractor.create_extractor")
    def test_progress_callback_invoked(
        self,
        mock_create_extractor: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Progress callback should be invoked for each chunk."""
        mock_config = MagicMock()
        mock_config.chunk_enabled = True
        mock_config.chunk_size_tokens = 50
        mock_config.chunk_overlap_tokens = 10
        mock_load_config.return_value = mock_config

        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c1", type="Company", name="Test")],
            relationships=[],
        )
        mock_extractor.extract.return_value = graph

        callback = MagicMock()
        large_text = "This is a large text. " * 50

        extract_knowledge_graph(large_text, on_chunk_complete=callback)

        # Callback should be called at least once
        assert callback.call_count >= 1

    @patch("src.extractor.load_config")
    @patch("src.extractor.create_extractor")
    def test_overlap_exceeds_chunk_size_raises_error(
        self,
        mock_create_extractor: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Overlap >= chunk size should raise ValueError."""
        mock_config = MagicMock()
        mock_config.chunk_enabled = True
        mock_config.chunk_size_tokens = 100
        mock_config.chunk_overlap_tokens = 100  # Invalid: overlap >= chunk size
        mock_load_config.return_value = mock_config

        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        large_text = "This is a large text. " * 100

        with pytest.raises(ValueError, match="CHUNK_OVERLAP_TOKENS must be less than"):
            extract_knowledge_graph(large_text)

    @patch("src.extractor.load_config")
    @patch("src.extractor.create_extractor")
    def test_merged_graph_deduplicates_nodes(
        self,
        mock_create_extractor: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Merged graph should deduplicate nodes with same (type, name)."""
        mock_config = MagicMock()
        mock_config.chunk_enabled = True
        mock_config.chunk_size_tokens = 50
        mock_config.chunk_overlap_tokens = 10
        mock_load_config.return_value = mock_config

        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        # Both chunks have same company
        graph1 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c1", type="Company", name="Shared Corp")],
            relationships=[],
        )
        graph2 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c1", type="Company", name="Shared Corp")],  # Duplicate
            relationships=[],
        )
        mock_extractor.extract.side_effect = [graph1, graph2]

        large_text = "This is a large text. " * 50

        result = extract_knowledge_graph(large_text)

        # Should have only 1 node (deduplicated)
        assert len(result.nodes) == 1
        assert result.nodes[0].name == "Shared Corp"
