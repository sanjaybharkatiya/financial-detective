"""Unit tests for the graph_merger module.

Tests graph merging functionality:
- Merging multiple graphs
- Single graph passthrough
- Empty graph handling
- Schema version preservation
- ID re-numbering and deduplication
"""

import pytest

from src.graph_merger import merge_graphs
from src.schema import KnowledgeGraph, Node, Relationship


class TestMergeGraphs:
    """Tests for merge_graphs function."""

    def test_empty_list_raises_error(self) -> None:
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot merge empty list of graphs"):
            merge_graphs([])

    def test_single_graph_returned_unchanged(self) -> None:
        """Single graph should be returned as-is."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c1", type="Company", name="Acme Corp")],
            relationships=[],
        )
        
        result = merge_graphs([graph])
        
        assert result is graph  # Same object, not a copy

    def test_two_graphs_merged_correctly(self) -> None:
        """Two graphs should merge nodes and relationships."""
        graph1 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c1", type="Company", name="Acme Corp")],
            relationships=[],
        )
        graph2 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c2", type="Company", name="Beta Inc")],
            relationships=[
                Relationship(source="c2", target="c1", relation="OWNS")
            ],
        )
        
        result = merge_graphs([graph1, graph2])
        
        assert result.schema_version == "1.0.0"
        assert len(result.nodes) == 2
        assert len(result.relationships) == 1

    def test_multiple_graphs_merged_correctly(self) -> None:
        """Multiple graphs should merge all nodes and relationships."""
        graphs = [
            KnowledgeGraph(
                schema_version="1.0.0",
                nodes=[Node(id=f"c{i}", type="Company", name=f"Company {i}")],
                relationships=[],
            )
            for i in range(5)
        ]
        
        result = merge_graphs(graphs)
        
        assert len(result.nodes) == 5
        assert len(result.relationships) == 0

    def test_schema_version_from_first_graph(self) -> None:
        """Schema version should come from the first graph."""
        graph1 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[],
            relationships=[],
        )
        graph2 = KnowledgeGraph(
            schema_version="2.0.0",
            nodes=[],
            relationships=[],
        )
        
        result = merge_graphs([graph1, graph2])
        
        assert result.schema_version == "1.0.0"

    def test_nodes_renumbered_with_type_prefix(self) -> None:
        """Nodes should be renumbered with type-based prefixes."""
        graph1 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="n1", type="Company", name="First"),
                Node(id="n2", type="Company", name="Second"),
            ],
            relationships=[],
        )
        graph2 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="n3", type="Company", name="Third"),
            ],
            relationships=[],
        )
        
        result = merge_graphs([graph1, graph2])
        
        assert len(result.nodes) == 3
        # IDs are renumbered with type prefix
        assert result.nodes[0].id == "company_1"
        assert result.nodes[1].id == "company_2"
        assert result.nodes[2].id == "company_3"
        # Names are preserved
        assert result.nodes[0].name == "First"
        assert result.nodes[1].name == "Second"
        assert result.nodes[2].name == "Third"

    def test_relationships_concatenated_in_order(self) -> None:
        """Relationships should be concatenated in graph order."""
        graph1 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[],
            relationships=[
                Relationship(source="a", target="b", relation="OWNS"),
            ],
        )
        graph2 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[],
            relationships=[
                Relationship(source="c", target="d", relation="HAS_RISK"),
            ],
        )
        
        result = merge_graphs([graph1, graph2])
        
        assert len(result.relationships) == 2
        assert result.relationships[0].source == "a"
        assert result.relationships[1].source == "c"

    def test_empty_graphs_merge_to_empty(self) -> None:
        """Merging empty graphs should produce empty graph."""
        graphs = [
            KnowledgeGraph(schema_version="1.0.0", nodes=[], relationships=[]),
            KnowledgeGraph(schema_version="1.0.0", nodes=[], relationships=[]),
        ]
        
        result = merge_graphs(graphs)
        
        assert len(result.nodes) == 0
        assert len(result.relationships) == 0

    def test_mixed_empty_and_nonempty_graphs(self) -> None:
        """Empty and non-empty graphs should merge correctly."""
        graph1 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[],
            relationships=[],
        )
        graph2 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c1", type="Company", name="Acme")],
            relationships=[],
        )
        graph3 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[],
            relationships=[],
        )
        
        result = merge_graphs([graph1, graph2, graph3])
        
        assert len(result.nodes) == 1
        # ID is renumbered
        assert result.nodes[0].id == "company_1"
        assert result.nodes[0].name == "Acme"

    def test_duplicate_nodes_deduplicated_by_type_and_name(self) -> None:
        """Duplicate nodes (same type and name) should be deduplicated."""
        graph1 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c1", type="Company", name="Acme Corp")],
            relationships=[],
        )
        graph2 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c1", type="Company", name="Acme Corp")],
            relationships=[],
        )
        
        result = merge_graphs([graph1, graph2])
        
        # Nodes with same (type, name) are deduplicated
        assert len(result.nodes) == 1
        assert result.nodes[0].name == "Acme Corp"

    def test_different_names_not_deduplicated(self) -> None:
        """Nodes with different names should not be deduplicated."""
        graph1 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c1", type="Company", name="Acme Corp")],
            relationships=[],
        )
        graph2 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c1", type="Company", name="Beta Inc")],
            relationships=[],
        )
        
        result = merge_graphs([graph1, graph2])
        
        # Different names = different nodes
        assert len(result.nodes) == 2

    def test_returns_new_graph_instance(self) -> None:
        """Merged result should be a new KnowledgeGraph instance."""
        graph1 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c1", type="Company", name="Acme")],
            relationships=[],
        )
        graph2 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c2", type="Company", name="Beta")],
            relationships=[],
        )
        
        result = merge_graphs([graph1, graph2])
        
        assert result is not graph1
        assert result is not graph2

    def test_relationship_references_updated_after_renumbering(self) -> None:
        """Relationship source/target within same chunk should be updated to new IDs."""
        # Both nodes and relationship in same graph/chunk
        graph1 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="old_c1", type="Company", name="Parent Corp"),
                Node(id="old_c2", type="Company", name="Child Inc"),
            ],
            relationships=[
                Relationship(source="old_c2", target="old_c1", relation="OWNS")
            ],
        )
        graph2 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[Node(id="c3", type="Company", name="Third Corp")],
            relationships=[],
        )
        
        result = merge_graphs([graph1, graph2])
        
        assert len(result.relationships) == 1
        # Relationship references should be updated to new IDs
        assert result.relationships[0].source == "company_2"
        assert result.relationships[0].target == "company_1"

    def test_deduplication_updates_relationship_references(self) -> None:
        """Relationships should reference deduplicated node IDs."""
        # Both chunks have the same company
        graph1 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="c1", type="Company", name="Shared Corp"),
                Node(id="r1", type="RiskFactor", name="Market Risk"),
            ],
            relationships=[
                Relationship(source="c1", target="r1", relation="HAS_RISK")
            ],
        )
        graph2 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="c1", type="Company", name="Shared Corp"),  # Duplicate
                Node(id="a1", type="DollarAmount", name="$1M"),
            ],
            relationships=[
                Relationship(source="c1", target="a1", relation="REPORTS_AMOUNT")
            ],
        )
        
        result = merge_graphs([graph1, graph2])
        
        # 3 unique nodes (company deduplicated, risk, amount)
        assert len(result.nodes) == 3
        # Both relationships should point to company_1
        assert len(result.relationships) == 2
        assert result.relationships[0].source == "company_1"
        assert result.relationships[1].source == "company_1"

    def test_different_node_types_have_different_prefixes(self) -> None:
        """Different node types should have different ID prefixes."""
        graph = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="x", type="Company", name="Corp A"),
                Node(id="y", type="RiskFactor", name="Risk A"),
                Node(id="z", type="DollarAmount", name="$100"),
            ],
            relationships=[],
        )
        # Use merge with two identical graphs to trigger renumbering
        result = merge_graphs([graph, graph])
        
        # Due to deduplication, should have 3 nodes
        assert len(result.nodes) == 3
        
        # Check prefixes
        ids = {n.id for n in result.nodes}
        assert "company_1" in ids
        assert "risk_1" in ids
        assert "amount_1" in ids

    def test_duplicate_relationships_deduplicated(self) -> None:
        """Duplicate relationships (same source, target, relation) should be deduplicated."""
        graph1 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="c1", type="Company", name="Corp"),
                Node(id="r1", type="RiskFactor", name="Risk"),
            ],
            relationships=[
                Relationship(source="c1", target="r1", relation="HAS_RISK")
            ],
        )
        graph2 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[
                Node(id="c1", type="Company", name="Corp"),
                Node(id="r1", type="RiskFactor", name="Risk"),
            ],
            relationships=[
                Relationship(source="c1", target="r1", relation="HAS_RISK")  # Duplicate
            ],
        )
        
        result = merge_graphs([graph1, graph2])
        
        # Duplicate relationship should be deduplicated
        assert len(result.relationships) == 1
