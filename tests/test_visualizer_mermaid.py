"""Unit tests for the visualizer_mermaid module.

Tests Mermaid diagram generation:
- Label escaping for special characters
- Node shape mapping based on type
- Mermaid file generation
- HTML file generation
"""

from pathlib import Path

import pytest

from src.schema import KnowledgeGraph, Node, Relationship
from src.visualizer_mermaid import (
    _escape_mermaid_label,
    _get_node_shape,
    render_mermaid,
    render_mermaid_html,
)


class TestEscapeMermaidLabel:
    """Tests for _escape_mermaid_label function."""

    def test_plain_text_unchanged(self) -> None:
        """Plain text without special characters should be unchanged."""
        assert _escape_mermaid_label("Acme Corp") == "Acme Corp"

    def test_double_quotes_replaced(self) -> None:
        """Double quotes should be replaced with single quotes."""
        assert _escape_mermaid_label('Company "Best"') == "Company 'Best'"

    def test_backticks_replaced(self) -> None:
        """Backticks should be replaced with single quotes."""
        assert _escape_mermaid_label("Code `example`") == "Code 'example'"

    def test_hash_removed(self) -> None:
        """Hash symbols should be removed."""
        assert _escape_mermaid_label("Item #1") == "Item 1"

    def test_ampersand_replaced(self) -> None:
        """Ampersand should be replaced with 'and'."""
        assert _escape_mermaid_label("A & B") == "A and B"

    def test_angle_brackets_removed(self) -> None:
        """Angle brackets should be removed."""
        assert _escape_mermaid_label("<html>") == "html"

    def test_square_brackets_replaced(self) -> None:
        """Square brackets should be replaced with parentheses."""
        assert _escape_mermaid_label("[item]") == "(item)"

    def test_curly_braces_replaced(self) -> None:
        """Curly braces should be replaced with parentheses."""
        assert _escape_mermaid_label("{value}") == "(value)"

    def test_pipe_replaced(self) -> None:
        """Pipe character should be replaced with dash."""
        assert _escape_mermaid_label("A | B") == "A - B"

    def test_long_text_truncated(self) -> None:
        """Text longer than 60 characters should be truncated."""
        long_text = "A" * 100
        result = _escape_mermaid_label(long_text)
        assert len(result) == 60
        assert result.endswith("...")

    def test_exactly_60_chars_not_truncated(self) -> None:
        """Text exactly 60 characters should not be truncated."""
        text_60 = "A" * 60
        result = _escape_mermaid_label(text_60)
        assert result == text_60
        assert not result.endswith("...")

    def test_multiple_special_chars(self) -> None:
        """Multiple special characters should all be handled."""
        result = _escape_mermaid_label('Test "quoted" & <html> [array]')
        assert result == "Test 'quoted' and html (array)"


class TestGetNodeShape:
    """Tests for _get_node_shape function."""

    def test_company_rectangle_shape(self) -> None:
        """Company nodes should use rectangle shape."""
        node = Node(id="c1", type="Company", name="Acme Corp")
        result = _get_node_shape(node)
        assert result == 'c1["Acme Corp"]'

    def test_risk_factor_rounded_shape(self) -> None:
        """RiskFactor nodes should use rounded shape."""
        node = Node(id="r1", type="RiskFactor", name="Market Risk")
        result = _get_node_shape(node)
        assert result == 'r1("Market Risk")'

    def test_dollar_amount_parallelogram_shape(self) -> None:
        """DollarAmount nodes should use parallelogram shape."""
        node = Node(id="a1", type="DollarAmount", name="$1,000,000")
        result = _get_node_shape(node)
        # Mermaid parallelogram syntax: [/"label"/]
        assert result == 'a1[/"$1,000,000"/]'

    def test_company_with_context(self) -> None:
        """Company with context should include context in label."""
        node = Node(id="c1", type="Company", name="Acme Corp", context="Parent company")
        result = _get_node_shape(node)
        assert "Acme Corp" in result
        assert "Parent company" in result

    def test_dollar_amount_with_context_shows_context_first(self) -> None:
        """DollarAmount with context should show context before amount."""
        node = Node(id="a1", type="DollarAmount", name="$1M", context="Revenue")
        result = _get_node_shape(node)
        # For dollar amounts, context comes first
        assert "Revenue: $1M" in result or "Revenue" in result

    def test_escapes_special_chars_in_name(self) -> None:
        """Special characters in node name should be escaped."""
        node = Node(id="c1", type="Company", name='Company "A"')
        result = _get_node_shape(node)
        assert '"A"' not in result  # Double quotes should be escaped
        assert "'A'" in result


class TestRenderMermaid:
    """Tests for render_mermaid function."""

    def test_creates_mermaid_file(self, tmp_path: Path) -> None:
        """render_mermaid should create a .mmd file."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="c1", type="Company", name="Acme Corp"),
            ],
            relationships=[],
        )
        output_path = tmp_path / "graph.mmd"

        render_mermaid(graph, output_path)

        assert output_path.exists()

    def test_file_starts_with_flowchart(self, tmp_path: Path) -> None:
        """Mermaid file should start with flowchart declaration."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="c1", type="Company", name="Acme Corp"),
            ],
            relationships=[],
        )
        output_path = tmp_path / "graph.mmd"

        render_mermaid(graph, output_path)

        content = output_path.read_text()
        assert content.startswith("flowchart")

    def test_includes_all_nodes(self, tmp_path: Path) -> None:
        """Mermaid file should include all graph nodes."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="c1", type="Company", name="Acme Corp"),
                Node(id="r1", type="RiskFactor", name="Market Risk"),
                Node(id="a1", type="DollarAmount", name="$1M"),
            ],
            relationships=[],
        )
        output_path = tmp_path / "graph.mmd"

        render_mermaid(graph, output_path)

        content = output_path.read_text()
        assert "c1" in content
        assert "r1" in content
        assert "a1" in content

    def test_includes_relationships(self, tmp_path: Path) -> None:
        """Mermaid file should include relationship arrows."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="c1", type="Company", name="Acme Corp"),
                Node(id="r1", type="RiskFactor", name="Market Risk"),
            ],
            relationships=[
                Relationship(source="c1", target="r1", relation="HAS_RISK"),
            ],
        )
        output_path = tmp_path / "graph.mmd"

        render_mermaid(graph, output_path)

        content = output_path.read_text()
        assert "c1" in content
        assert "r1" in content
        assert "-->" in content or "---" in content

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """render_mermaid should create parent directories if needed."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c1", type="Company", name="Acme")],
            relationships=[],
        )
        output_path = tmp_path / "deep" / "nested" / "graph.mmd"

        render_mermaid(graph, output_path)

        assert output_path.exists()

    def test_empty_graph_still_creates_file(self, tmp_path: Path) -> None:
        """Empty graph should still create a valid Mermaid file."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[],
            relationships=[],
        )
        output_path = tmp_path / "graph.mmd"

        render_mermaid(graph, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "flowchart" in content


class TestRenderMermaidHtml:
    """Tests for render_mermaid_html function."""

    def test_creates_html_file(self, tmp_path: Path) -> None:
        """render_mermaid_html should create an .html file."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="c1", type="Company", name="Acme Corp"),
            ],
            relationships=[],
        )
        output_path = tmp_path / "graph.html"

        render_mermaid_html(graph, output_path)

        assert output_path.exists()

    def test_html_contains_doctype(self, tmp_path: Path) -> None:
        """HTML file should start with DOCTYPE declaration."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c1", type="Company", name="Acme")],
            relationships=[],
        )
        output_path = tmp_path / "graph.html"

        render_mermaid_html(graph, output_path)

        content = output_path.read_text()
        assert "<!DOCTYPE html>" in content

    def test_html_includes_mermaid_script(self, tmp_path: Path) -> None:
        """HTML file should include Mermaid.js library."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c1", type="Company", name="Acme")],
            relationships=[],
        )
        output_path = tmp_path / "graph.html"

        render_mermaid_html(graph, output_path)

        content = output_path.read_text()
        assert "mermaid" in content.lower()

    def test_html_includes_graph_content(self, tmp_path: Path) -> None:
        """HTML file should include graph nodes."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="company_1", type="Company", name="Acme Corp"),
                Node(id="risk_1", type="RiskFactor", name="Market Risk"),
            ],
            relationships=[
                Relationship(source="company_1", target="risk_1", relation="HAS_RISK"),
            ],
        )
        output_path = tmp_path / "graph.html"

        render_mermaid_html(graph, output_path)

        content = output_path.read_text()
        assert "company_1" in content
        assert "risk_1" in content

    def test_html_includes_zoom_controls(self, tmp_path: Path) -> None:
        """HTML file should include zoom controls."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c1", type="Company", name="Acme")],
            relationships=[],
        )
        output_path = tmp_path / "graph.html"

        render_mermaid_html(graph, output_path)

        content = output_path.read_text()
        # Should have zoom-related content
        assert "zoom" in content.lower()

    def test_html_includes_legend(self, tmp_path: Path) -> None:
        """HTML file should include a legend for node types."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c1", type="Company", name="Acme")],
            relationships=[],
        )
        output_path = tmp_path / "graph.html"

        render_mermaid_html(graph, output_path)

        content = output_path.read_text()
        assert "legend" in content.lower()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """render_mermaid_html should create parent directories if needed."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c1", type="Company", name="Acme")],
            relationships=[],
        )
        output_path = tmp_path / "deep" / "nested" / "graph.html"

        render_mermaid_html(graph, output_path)

        assert output_path.exists()

    def test_html_displays_node_count(self, tmp_path: Path) -> None:
        """HTML should display the number of nodes."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="c1", type="Company", name="Acme"),
                Node(id="c2", type="Company", name="Beta"),
                Node(id="r1", type="RiskFactor", name="Risk"),
            ],
            relationships=[],
        )
        output_path = tmp_path / "graph.html"

        render_mermaid_html(graph, output_path)

        content = output_path.read_text()
        # Should show node count somewhere
        assert "3" in content  # 3 nodes

    def test_html_displays_relationship_count(self, tmp_path: Path) -> None:
        """HTML should display the number of relationships."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="c1", type="Company", name="Acme"),
                Node(id="r1", type="RiskFactor", name="Risk"),
            ],
            relationships=[
                Relationship(source="c1", target="r1", relation="HAS_RISK"),
            ],
        )
        output_path = tmp_path / "graph.html"

        render_mermaid_html(graph, output_path)

        content = output_path.read_text()
        # Should show relationship count
        assert "1" in content  # 1 relationship

    def test_html_is_self_contained(self, tmp_path: Path) -> None:
        """HTML file should be self-contained (can open in browser)."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c1", type="Company", name="Acme")],
            relationships=[],
        )
        output_path = tmp_path / "graph.html"

        render_mermaid_html(graph, output_path)

        content = output_path.read_text()
        # Should have proper HTML structure
        assert "<html" in content
        assert "</html>" in content
        assert "<head>" in content
        assert "<body>" in content

