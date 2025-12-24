"""Graph merging module for combining multiple Knowledge Graphs.

This module provides functionality to merge multiple KnowledgeGraph instances
into a single unified graph. Used when extracting from chunked documents.
"""

from src.schema import KnowledgeGraph


def merge_graphs(graphs: list[KnowledgeGraph]) -> KnowledgeGraph:
    """Merge multiple Knowledge Graphs into a single unified graph.

    Combines nodes and relationships from all input graphs using a simple
    concatenation strategy. No deduplication is performed - duplicate entities
    across chunks will result in multiple nodes.

    The schema_version is taken from the first graph in the list.

    Args:
        graphs: List of KnowledgeGraph instances to merge.
            Must contain at least one graph.

    Returns:
        A new KnowledgeGraph containing all nodes and relationships
        from the input graphs.

    Raises:
        ValueError: If graphs list is empty.

    Example:
        >>> graph1 = KnowledgeGraph(
        ...     schema_version="1.0.0",
        ...     nodes=[Node(id="n1", type="Company", name="Corp A")],
        ...     relationships=[]
        ... )
        >>> graph2 = KnowledgeGraph(
        ...     schema_version="1.0.0",
        ...     nodes=[Node(id="n2", type="Company", name="Corp B")],
        ...     relationships=[]
        ... )
        >>> merged = merge_graphs([graph1, graph2])
        >>> len(merged.nodes)
        2
    """
    if not graphs:
        raise ValueError("Cannot merge empty list of graphs")

    if len(graphs) == 1:
        return graphs[0]

    # Use schema version from first graph
    schema_version = graphs[0].schema_version

    # Combine all nodes from all graphs
    all_nodes = []
    for graph in graphs:
        all_nodes.extend(graph.nodes)

    # Combine all relationships from all graphs
    all_relationships = []
    for graph in graphs:
        all_relationships.extend(graph.relationships)

    return KnowledgeGraph(
        schema_version=schema_version,
        nodes=all_nodes,
        relationships=all_relationships,
    )
