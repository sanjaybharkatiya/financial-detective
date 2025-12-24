"""Unit tests for the graph_merger module.

Tests graph merging functionality:
- Merging multiple graphs
- Single graph passthrough
- Empty graph handling
- Schema version preservation
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

    def test_nodes_concatenated_in_order(self) -> None:
        """Nodes should be concatenated in graph order."""
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
        assert result.nodes[0].id == "n1"
        assert result.nodes[1].id == "n2"
        assert result.nodes[2].id == "n3"

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
        assert result.nodes[0].id == "c1"

    def test_duplicate_nodes_preserved(self) -> None:
        """Duplicate nodes should be preserved (no deduplication)."""
        node = Node(id="c1", type="Company", name="Acme Corp")
        graph1 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[node],
            relationships=[],
        )
        graph2 = KnowledgeGraph(
            schema_version="1.0.0",
            nodes=[node],
            relationships=[],
        )
        
        result = merge_graphs([graph1, graph2])
        
        # Both nodes preserved, no deduplication
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
