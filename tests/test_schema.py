"""Unit tests for the schema module.

Tests Pydantic schema validation:
- Node model validation
- Relationship model validation
- KnowledgeGraph model validation
- Extra field rejection (extra="forbid")
"""

import pytest
from pydantic import ValidationError

from src.schema import KnowledgeGraph, Node, Relationship


class TestNode:
    """Tests for the Node model."""

    def test_valid_company_node(self) -> None:
        """Valid Company node should be created successfully."""
        node = Node(id="company_1", type="Company", name="Acme Corp")

        assert node.id == "company_1"
        assert node.type == "Company"
        assert node.name == "Acme Corp"
        assert node.context is None

    def test_valid_risk_factor_node(self) -> None:
        """Valid RiskFactor node should be created successfully."""
        node = Node(id="risk_1", type="RiskFactor", name="Market Volatility")

        assert node.type == "RiskFactor"

    def test_valid_dollar_amount_node(self) -> None:
        """Valid DollarAmount node should be created successfully."""
        node = Node(id="amount_1", type="DollarAmount", name="$1,000,000")

        assert node.type == "DollarAmount"

    def test_context_is_optional(self) -> None:
        """Context field should be optional."""
        node = Node(id="c1", type="Company", name="Corp")

        assert node.context is None

    def test_context_can_be_provided(self) -> None:
        """Context field should accept a string value."""
        node = Node(
            id="c1",
            type="Company",
            name="Acme Corp",
            context="Parent holding company",
        )

        assert node.context == "Parent holding company"

    def test_invalid_type_rejected(self) -> None:
        """Invalid node type should raise ValidationError."""
        with pytest.raises(ValidationError):
            Node(id="x1", type="InvalidType", name="Test")  # type: ignore[arg-type]

    def test_missing_id_rejected(self) -> None:
        """Node without id should raise ValidationError."""
        with pytest.raises(ValidationError):
            Node(type="Company", name="Test")  # type: ignore[call-arg]

    def test_missing_name_rejected(self) -> None:
        """Node without name should raise ValidationError."""
        with pytest.raises(ValidationError):
            Node(id="c1", type="Company")  # type: ignore[call-arg]

    def test_missing_type_rejected(self) -> None:
        """Node without type should raise ValidationError."""
        with pytest.raises(ValidationError):
            Node(id="c1", name="Test")  # type: ignore[call-arg]

    def test_extra_field_rejected(self) -> None:
        """Extra fields should be rejected (extra='forbid')."""
        with pytest.raises(ValidationError) as exc_info:
            Node(
                id="c1",
                type="Company",
                name="Test",
                extra_field="not allowed",  # type: ignore[call-arg]
            )

        assert "extra_field" in str(exc_info.value).lower() or "extra" in str(exc_info.value).lower()


class TestRelationship:
    """Tests for the Relationship model."""

    def test_valid_has_risk_relationship(self) -> None:
        """Valid HAS_RISK relationship should be created."""
        rel = Relationship(source="c1", target="r1", relation="HAS_RISK")

        assert rel.source == "c1"
        assert rel.target == "r1"
        assert rel.relation == "HAS_RISK"
        assert rel.confidence is None

    def test_valid_reports_amount_relationship(self) -> None:
        """Valid REPORTS_AMOUNT relationship should be created."""
        rel = Relationship(source="c1", target="a1", relation="REPORTS_AMOUNT")

        assert rel.relation == "REPORTS_AMOUNT"

    def test_valid_owns_relationship(self) -> None:
        """Valid OWNS relationship should be created."""
        rel = Relationship(source="c1", target="c2", relation="OWNS")

        assert rel.relation == "OWNS"

    def test_all_valid_relation_types(self) -> None:
        """All defined relation types should be accepted."""
        valid_relations = [
            "OWNS",
            "HAS_RISK",
            "REPORTS_AMOUNT",
            "OPERATES",
            "IMPACTED_BY",
            "DECLINED_DUE_TO",
            "SUPPORTED_BY",
            "PARTNERED_WITH",
            "JOINT_VENTURE_WITH",
            "RAISED_CAPITAL",
            "INVESTED_IN",
            "COMMITTED_CAPEX",
            "TARGETS",
            "PLANS_TO",
            "ON_TRACK_TO",
            "COMMITTED_TO",
            "COMPLIES_WITH",
            "SUBJECT_TO",
        ]

        for relation in valid_relations:
            rel = Relationship(source="a", target="b", relation=relation)  # type: ignore[arg-type]
            assert rel.relation == relation

    def test_invalid_relation_rejected(self) -> None:
        """Invalid relation type should raise ValidationError."""
        with pytest.raises(ValidationError):
            Relationship(source="a", target="b", relation="INVALID_RELATION")  # type: ignore[arg-type]

    def test_confidence_is_optional(self) -> None:
        """Confidence field should be optional."""
        rel = Relationship(source="a", target="b", relation="OWNS")

        assert rel.confidence is None

    def test_confidence_can_be_provided(self) -> None:
        """Confidence field should accept a float value."""
        rel = Relationship(source="a", target="b", relation="OWNS", confidence=0.95)

        assert rel.confidence == 0.95

    def test_confidence_edge_values(self) -> None:
        """Confidence should accept edge values (0.0, 1.0)."""
        rel_zero = Relationship(source="a", target="b", relation="OWNS", confidence=0.0)
        rel_one = Relationship(source="a", target="b", relation="OWNS", confidence=1.0)

        assert rel_zero.confidence == 0.0
        assert rel_one.confidence == 1.0

    def test_missing_source_rejected(self) -> None:
        """Relationship without source should raise ValidationError."""
        with pytest.raises(ValidationError):
            Relationship(target="b", relation="OWNS")  # type: ignore[call-arg]

    def test_missing_target_rejected(self) -> None:
        """Relationship without target should raise ValidationError."""
        with pytest.raises(ValidationError):
            Relationship(source="a", relation="OWNS")  # type: ignore[call-arg]

    def test_missing_relation_rejected(self) -> None:
        """Relationship without relation should raise ValidationError."""
        with pytest.raises(ValidationError):
            Relationship(source="a", target="b")  # type: ignore[call-arg]

    def test_extra_field_rejected(self) -> None:
        """Extra fields should be rejected (extra='forbid')."""
        with pytest.raises(ValidationError):
            Relationship(
                source="a",
                target="b",
                relation="OWNS",
                extra_field="not allowed",  # type: ignore[call-arg]
            )


class TestKnowledgeGraph:
    """Tests for the KnowledgeGraph model."""

    def test_valid_knowledge_graph(self) -> None:
        """Valid KnowledgeGraph should be created successfully."""
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

        assert graph.schema_version == "1.0.0"
        assert len(graph.nodes) == 2
        assert len(graph.relationships) == 1

    def test_empty_nodes_and_relationships(self) -> None:
        """Empty nodes and relationships lists should be valid."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[],
            relationships=[],
        )

        assert len(graph.nodes) == 0
        assert len(graph.relationships) == 0

    def test_missing_schema_version_rejected(self) -> None:
        """KnowledgeGraph without schema_version should raise ValidationError."""
        with pytest.raises(ValidationError):
            KnowledgeGraph(nodes=[], relationships=[])  # type: ignore[call-arg]

    def test_missing_nodes_rejected(self) -> None:
        """KnowledgeGraph without nodes should raise ValidationError."""
        with pytest.raises(ValidationError):
            KnowledgeGraph(schema_version="1.0.0", relationships=[])  # type: ignore[call-arg]

    def test_missing_relationships_rejected(self) -> None:
        """KnowledgeGraph without relationships should raise ValidationError."""
        with pytest.raises(ValidationError):
            KnowledgeGraph(schema_version="1.0.0", nodes=[])  # type: ignore[call-arg]

    def test_extra_field_rejected(self) -> None:
        """Extra fields should be rejected (extra='forbid')."""
        with pytest.raises(ValidationError):
            KnowledgeGraph(
                schema_version="1.0.0",
                nodes=[],
                relationships=[],
                extra_field="not allowed",  # type: ignore[call-arg]
            )

    def test_invalid_node_in_list_rejected(self) -> None:
        """Invalid node in nodes list should raise ValidationError."""
        with pytest.raises(ValidationError):
            KnowledgeGraph(
                schema_version="1.0.0",
                nodes=[{"id": "x", "type": "InvalidType", "name": "Test"}],  # type: ignore[list-item]
                relationships=[],
            )

    def test_invalid_relationship_in_list_rejected(self) -> None:
        """Invalid relationship in list should raise ValidationError."""
        with pytest.raises(ValidationError):
            KnowledgeGraph(
                schema_version="1.0.0",
                nodes=[],
                relationships=[{"source": "a", "target": "b", "relation": "INVALID"}],  # type: ignore[list-item]
            )

    def test_graph_with_all_node_types(self) -> None:
        """Graph with all node types should be valid."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="c1", type="Company", name="Acme Corp"),
                Node(id="r1", type="RiskFactor", name="Market Risk"),
                Node(id="a1", type="DollarAmount", name="$1M"),
            ],
            relationships=[
                Relationship(source="c1", target="r1", relation="HAS_RISK"),
                Relationship(source="c1", target="a1", relation="REPORTS_AMOUNT"),
            ],
        )

        types = {n.type for n in graph.nodes}
        assert types == {"Company", "RiskFactor", "DollarAmount"}

    def test_graph_with_context(self) -> None:
        """Graph nodes with context should be valid."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(
                    id="c1",
                    type="Company",
                    name="Acme",
                    context="Parent holding company",
                ),
            ],
            relationships=[],
        )

        assert graph.nodes[0].context == "Parent holding company"

    def test_model_dump_json(self) -> None:
        """KnowledgeGraph should serialize to valid JSON."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c1", type="Company", name="Acme")],
            relationships=[],
        )

        json_str = graph.model_dump_json()

        assert "schema_version" in json_str
        assert "nodes" in json_str
        assert "relationships" in json_str
        assert "Acme" in json_str

    def test_model_validate_from_dict(self) -> None:
        """KnowledgeGraph should deserialize from valid dict."""
        data = {
            "schema_version": "1.0.0",
            "nodes": [
                {"id": "c1", "type": "Company", "name": "Acme"},
            ],
            "relationships": [],
        }

        graph = KnowledgeGraph.model_validate(data)

        assert graph.schema_version == "1.0.0"
        assert len(graph.nodes) == 1
        assert graph.nodes[0].name == "Acme"

