"""Graph merging module for combining multiple Knowledge Graphs.

This module provides functionality to merge multiple KnowledgeGraph instances
into a single unified graph. Used when extracting from chunked documents.

Key features:
- Re-numbers IDs to prevent collisions across chunks
- Deduplicates nodes based on (type, name) pairs
- Updates relationship references to point to deduplicated nodes
"""

from src.schema import KnowledgeGraph, Node, Relationship


def merge_graphs(graphs: list[KnowledgeGraph]) -> KnowledgeGraph:
    """Merge multiple Knowledge Graphs into a single unified graph.

    Combines nodes and relationships from all input graphs with:
    1. ID collision prevention: Each chunk's IDs are prefixed with chunk number
    2. Deduplication: Nodes with same (type, name) are merged
    3. Reference updates: Relationships point to deduplicated node IDs

    The schema_version is taken from the first graph in the list.

    Args:
        graphs: List of KnowledgeGraph instances to merge.
            Must contain at least one graph.

    Returns:
        A new KnowledgeGraph containing deduplicated nodes and
        all relationships with updated references.

    Raises:
        ValueError: If graphs list is empty.

    Example:
        >>> graph1 = KnowledgeGraph(
        ...     schema_version="1.0.0",
        ...     nodes=[Node(id="company_1", type="Company", name="Corp A")],
        ...     relationships=[]
        ... )
        >>> graph2 = KnowledgeGraph(
        ...     schema_version="1.0.0",
        ...     nodes=[Node(id="company_1", type="Company", name="Corp A")],
        ...     relationships=[]
        ... )
        >>> merged = merge_graphs([graph1, graph2])
        >>> len(merged.nodes)  # Deduplicated
        1
    """
    if not graphs:
        raise ValueError("Cannot merge empty list of graphs")

    if len(graphs) == 1:
        return graphs[0]

    # Use schema version from first graph
    schema_version = graphs[0].schema_version

    # Track unique nodes by (type, name) -> canonical ID
    # This deduplicates nodes that appear in multiple chunks
    node_key_to_id: dict[tuple[str, str], str] = {}

    # Track all unique nodes
    unique_nodes: list[Node] = []

    # Track ID mappings: (chunk_index, old_id) -> new_id
    id_mapping: dict[tuple[int, str], str] = {}

    # Counters for generating new unique IDs
    type_counters: dict[str, int] = {
        "Company": 0,
        "RiskFactor": 0,
        "DollarAmount": 0,
    }

    # First pass: collect and deduplicate nodes
    for chunk_idx, graph in enumerate(graphs):
        for node in graph.nodes:
            node_key = (node.type, node.name)

            if node_key in node_key_to_id:
                # Node already exists - map this chunk's ID to existing ID
                existing_id = node_key_to_id[node_key]
                id_mapping[(chunk_idx, node.id)] = existing_id
            else:
                # New unique node - generate new ID
                type_counters[node.type] = type_counters.get(node.type, 0) + 1
                type_prefix = _get_type_prefix(node.type)
                new_id = f"{type_prefix}_{type_counters[node.type]}"

                # Store mapping
                node_key_to_id[node_key] = new_id
                id_mapping[(chunk_idx, node.id)] = new_id

                # Create node with new ID
                unique_nodes.append(Node(
                    id=new_id,
                    type=node.type,
                    name=node.name,
                ))

    # Second pass: collect relationships with updated IDs
    all_relationships: list[Relationship] = []
    seen_relationships: set[tuple[str, str, str]] = set()

    for chunk_idx, graph in enumerate(graphs):
        for rel in graph.relationships:
            # Map source and target to new IDs
            new_source = id_mapping.get((chunk_idx, rel.source), rel.source)
            new_target = id_mapping.get((chunk_idx, rel.target), rel.target)

            # Deduplicate relationships
            rel_key = (new_source, new_target, rel.relation)
            if rel_key in seen_relationships:
                continue
            seen_relationships.add(rel_key)

            all_relationships.append(Relationship(
                source=new_source,
                target=new_target,
                relation=rel.relation,
                confidence=rel.confidence,
            ))

    return KnowledgeGraph(
        schema_version=schema_version,
        nodes=unique_nodes,
        relationships=all_relationships,
    )


def _get_type_prefix(node_type: str) -> str:
    """Get the ID prefix for a node type.

    Args:
        node_type: The node type (Company, RiskFactor, DollarAmount).

    Returns:
        The ID prefix to use for that type.
    """
    prefixes = {
        "Company": "company",
        "RiskFactor": "risk",
        "DollarAmount": "amount",
    }
    return prefixes.get(node_type, node_type.lower())
