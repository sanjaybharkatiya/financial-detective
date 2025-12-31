"""Unit tests for the validator module.

Tests validate_knowledge_graph function for:
- Valid graph acceptance
- Duplicate node ID rejection
- Invalid relationship reference rejection

Tests validate_and_repair_graph function for:
- Auto-removal of invalid relationships
- Type constraint enforcement
"""

import pytest

from src.schema import KnowledgeGraph, Node, Relationship
from src.validator import validate_knowledge_graph, validate_and_repair_graph


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


class TestValidateAndRepairGraph:
    """Tests for validate_and_repair_graph function."""

    def test_valid_graph_unchanged(self) -> None:
        """A valid graph should be returned with all relationships intact."""
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

        result = validate_and_repair_graph(graph)

        assert len(result.relationships) == 2

    def test_removes_has_risk_to_company(self) -> None:
        """HAS_RISK relationship targeting a Company should be removed."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="company_1", type="Company", name="Acme Corp"),
                Node(id="company_2", type="Company", name="Beta Inc"),
            ],
            relationships=[
                Relationship(source="company_1", target="company_2", relation="HAS_RISK"),
            ],
        )

        result = validate_and_repair_graph(graph)

        assert len(result.relationships) == 0

    def test_removes_has_risk_to_dollar_amount(self) -> None:
        """HAS_RISK relationship targeting a DollarAmount should be removed."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="company_1", type="Company", name="Acme Corp"),
                Node(id="amount_1", type="DollarAmount", name="$1M"),
            ],
            relationships=[
                Relationship(source="company_1", target="amount_1", relation="HAS_RISK"),
            ],
        )

        result = validate_and_repair_graph(graph)

        assert len(result.relationships) == 0

    def test_removes_reports_amount_to_company(self) -> None:
        """REPORTS_AMOUNT relationship targeting a Company should be removed."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="company_1", type="Company", name="Acme Corp"),
                Node(id="company_2", type="Company", name="Beta Inc"),
            ],
            relationships=[
                Relationship(source="company_1", target="company_2", relation="REPORTS_AMOUNT"),
            ],
        )

        result = validate_and_repair_graph(graph)

        assert len(result.relationships) == 0

    def test_removes_reports_amount_to_risk(self) -> None:
        """REPORTS_AMOUNT relationship targeting a RiskFactor should be removed."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="company_1", type="Company", name="Acme Corp"),
                Node(id="risk_1", type="RiskFactor", name="Market Risk"),
            ],
            relationships=[
                Relationship(source="company_1", target="risk_1", relation="REPORTS_AMOUNT"),
            ],
        )

        result = validate_and_repair_graph(graph)

        assert len(result.relationships) == 0

    def test_removes_owns_with_non_company_source(self) -> None:
        """OWNS relationship with non-Company source should be removed."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="risk_1", type="RiskFactor", name="Risk"),
                Node(id="company_1", type="Company", name="Acme Corp"),
            ],
            relationships=[
                Relationship(source="risk_1", target="company_1", relation="OWNS"),
            ],
        )

        result = validate_and_repair_graph(graph)

        assert len(result.relationships) == 0

    def test_removes_owns_with_non_company_target(self) -> None:
        """OWNS relationship with non-Company target should be removed."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="company_1", type="Company", name="Acme Corp"),
                Node(id="risk_1", type="RiskFactor", name="Risk"),
            ],
            relationships=[
                Relationship(source="company_1", target="risk_1", relation="OWNS"),
            ],
        )

        result = validate_and_repair_graph(graph)

        assert len(result.relationships) == 0

    def test_valid_owns_between_companies_preserved(self) -> None:
        """OWNS relationship between two Companies should be preserved."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="company_1", type="Company", name="Parent Corp"),
                Node(id="company_2", type="Company", name="Subsidiary Inc"),
            ],
            relationships=[
                Relationship(source="company_1", target="company_2", relation="OWNS"),
            ],
        )

        result = validate_and_repair_graph(graph)

        assert len(result.relationships) == 1
        assert result.relationships[0].relation == "OWNS"

    def test_removes_relationships_with_missing_source(self) -> None:
        """Relationships with missing source node should be removed."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="risk_1", type="RiskFactor", name="Market Risk"),
            ],
            relationships=[
                Relationship(source="nonexistent", target="risk_1", relation="HAS_RISK"),
            ],
        )

        result = validate_and_repair_graph(graph)

        assert len(result.relationships) == 0

    def test_removes_relationships_with_missing_target(self) -> None:
        """Relationships with missing target node should be removed."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="company_1", type="Company", name="Acme Corp"),
            ],
            relationships=[
                Relationship(source="company_1", target="nonexistent", relation="HAS_RISK"),
            ],
        )

        result = validate_and_repair_graph(graph)

        assert len(result.relationships) == 0

    def test_empty_graph_raises_error(self) -> None:
        """Graph with no nodes should raise ValueError."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[],
            relationships=[],
        )

        with pytest.raises(ValueError, match="KnowledgeGraph must contain at least one node"):
            validate_and_repair_graph(graph)

    def test_preserves_valid_while_removing_invalid(self) -> None:
        """Valid relationships should be preserved while invalid ones are removed."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="company_1", type="Company", name="Acme Corp"),
                Node(id="risk_1", type="RiskFactor", name="Market Risk"),
                Node(id="amount_1", type="DollarAmount", name="$1M"),
            ],
            relationships=[
                Relationship(source="company_1", target="risk_1", relation="HAS_RISK"),  # Valid
                Relationship(source="company_1", target="amount_1", relation="HAS_RISK"),  # Invalid: amount not risk
                Relationship(source="company_1", target="amount_1", relation="REPORTS_AMOUNT"),  # Valid
            ],
        )

        result = validate_and_repair_graph(graph)

        assert len(result.relationships) == 2
        relations = {r.relation for r in result.relationships}
        assert "HAS_RISK" in relations
        assert "REPORTS_AMOUNT" in relations

    def test_other_relations_not_constrained(self) -> None:
        """Relationships like OPERATES, IMPACTED_BY should not be type-constrained."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="company_1", type="Company", name="Acme Corp"),
                Node(id="risk_1", type="RiskFactor", name="Regulation"),
            ],
            relationships=[
                Relationship(source="company_1", target="risk_1", relation="IMPACTED_BY"),
            ],
        )

        result = validate_and_repair_graph(graph)

        assert len(result.relationships) == 1
        assert result.relationships[0].relation == "IMPACTED_BY"

    def test_preserves_node_structure(self) -> None:
        """Nodes should be preserved in the repaired graph."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="company_1", type="Company", name="Acme Corp", context="Parent company"),
                Node(id="risk_1", type="RiskFactor", name="Market Risk"),
            ],
            relationships=[],
        )

        result = validate_and_repair_graph(graph)

        assert len(result.nodes) == 2
        assert result.nodes[0].context == "Parent company"

    def test_preserves_schema_version(self) -> None:
        """Schema version should be preserved in the repaired graph."""
        graph = KnowledgeGraph(
            schema_version="2.0.0",
            nodes=[
                Node(id="company_1", type="Company", name="Acme Corp"),
            ],
            relationships=[],
        )

        result = validate_and_repair_graph(graph)

        assert result.schema_version == "2.0.0"

