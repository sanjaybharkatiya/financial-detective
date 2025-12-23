"""Unit tests for the visualizer module.

Tests render_graph function:
- Graph image file is created in specified directory
- Output file is a valid PNG image

Note: These tests require networkx which may have compatibility issues
with Python 3.14. Tests are skipped if networkx cannot be imported.
"""

import sys
from pathlib import Path

import pytest

from src.schema import KnowledgeGraph, Node, Relationship

# Skip all tests in this module if networkx has compatibility issues
try:
    from src.visualizer import render_graph
    NETWORKX_AVAILABLE = True
except (ImportError, AttributeError):
    NETWORKX_AVAILABLE = False
    render_graph = None  # type: ignore

pytestmark = pytest.mark.skipif(
    not NETWORKX_AVAILABLE,
    reason="networkx not compatible with Python 3.14"
)


class TestRenderGraph:
    """Tests for render_graph function."""

    def test_graph_image_is_created(self, tmp_path: Path) -> None:
        """render_graph should create an image file at the specified path."""
        graph = KnowledgeGraph(
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

        output_path = tmp_path / "output" / "graph.png"

        render_graph(graph, output_path)

        assert output_path.exists()
        assert output_path.is_file()

    def test_output_file_is_valid_png(self, tmp_path: Path) -> None:
        """render_graph should create a valid PNG file with correct header."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="company_1", type="Company", name="Acme Corp"),
            ],
            relationships=[],
        )

        output_path = tmp_path / "graph.png"

        render_graph(graph, output_path)

        # PNG files start with these magic bytes
        png_header = b"\x89PNG\r\n\x1a\n"
        with open(output_path, "rb") as f:
            file_header = f.read(8)

        assert file_header == png_header

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """render_graph should create parent directories if they don't exist."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="company_1", type="Company", name="Acme Corp"),
            ],
            relationships=[],
        )

        nested_path = tmp_path / "deep" / "nested" / "dir" / "graph.png"

        render_graph(graph, nested_path)

        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_graph_with_all_node_types(self, tmp_path: Path) -> None:
        """render_graph should handle all node types without error."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="company_1", type="Company", name="Acme Corp"),
                Node(id="company_2", type="Company", name="Beta Inc"),
                Node(id="risk_1", type="RiskFactor", name="Market Risk"),
                Node(id="risk_2", type="RiskFactor", name="Credit Risk"),
                Node(id="amount_1", type="DollarAmount", name="$500,000"),
                Node(id="amount_2", type="DollarAmount", name="$1,000,000"),
            ],
            relationships=[
                Relationship(source="company_1", target="company_2", relation="OWNS"),
                Relationship(source="company_1", target="risk_1", relation="HAS_RISK"),
                Relationship(source="company_2", target="risk_2", relation="HAS_RISK"),
                Relationship(source="company_1", target="amount_1", relation="REPORTS_AMOUNT"),
                Relationship(source="company_2", target="amount_2", relation="REPORTS_AMOUNT"),
            ],
        )

        output_path = tmp_path / "full_graph.png"

        render_graph(graph, output_path)

        assert output_path.exists()
        # File should have some content (not empty)
        assert output_path.stat().st_size > 0
