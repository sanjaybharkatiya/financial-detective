"""Unit tests for the validator module.

Tests validate_knowledge_graph function for:
- Valid graph acceptance
- Duplicate node ID rejection
- Invalid relationship reference rejection
"""

import pytest

from src.schema import KnowledgeGraph, Node, Relationship
from src.validator import validate_knowledge_graph


class TestValidateKnowledgeGraph:
    """Tests for validate_knowledge_graph function."""

    def test_valid_graph_passes(self) -> None:
        """A valid KnowledgeGraph with unique IDs and valid references should pass."""
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

        # Should not raise any exception
        validate_knowledge_graph(graph)

    def test_duplicate_node_ids_fail(self) -> None:
        """KnowledgeGraph with duplicate node IDs should raise ValueError."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="company_1", type="Company", name="Acme Corp"),
                Node(id="company_1", type="Company", name="Beta Inc"),  # Duplicate ID
                Node(id="risk_1", type="RiskFactor", name="Market Volatility"),
            ],
            relationships=[],
        )

        with pytest.raises(ValueError, match="Duplicate node IDs found"):
            validate_knowledge_graph(graph)

    def test_invalid_relationship_source_fails(self) -> None:
        """Relationship with non-existent source ID should raise ValueError."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="company_1", type="Company", name="Acme Corp"),
                Node(id="risk_1", type="RiskFactor", name="Market Volatility"),
            ],
            relationships=[
                Relationship(source="nonexistent", target="risk_1", relation="HAS_RISK"),
            ],
        )

        with pytest.raises(ValueError, match="Invalid node references in relationships"):
            validate_knowledge_graph(graph)

    def test_invalid_relationship_target_fails(self) -> None:
        """Relationship with non-existent target ID should raise ValueError."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="company_1", type="Company", name="Acme Corp"),
                Node(id="risk_1", type="RiskFactor", name="Market Volatility"),
            ],
            relationships=[
                Relationship(source="company_1", target="nonexistent", relation="HAS_RISK"),
            ],
        )

        with pytest.raises(ValueError, match="Invalid node references in relationships"):
            validate_knowledge_graph(graph)

    def test_empty_nodes_fails(self) -> None:
        """KnowledgeGraph with no nodes should raise ValueError."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[],
            relationships=[],
        )

        with pytest.raises(ValueError, match="KnowledgeGraph must contain at least one node"):
            validate_knowledge_graph(graph)

