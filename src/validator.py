"""Knowledge Graph validation module.

This module provides validation logic to ensure the integrity
of a KnowledgeGraph instance beyond Pydantic schema validation.
"""

from src.schema import KnowledgeGraph


def validate_knowledge_graph(graph: KnowledgeGraph) -> None:
    """Validate the integrity of a KnowledgeGraph.

     Performs the following validations:
       1. KnowledgeGraph must contain at least one node.
       2. All node IDs must be unique.
       3. Every relationship source and target must reference an existing node ID.

    This function does not modify the KnowledgeGraph.

    Args:
        graph: The KnowledgeGraph instance to validate.

    Returns:
        None. Validation passes silently if all checks succeed.

    Raises:
        ValueError: If duplicate node IDs are found.
        ValueError: If a relationship references a non-existent node ID.
    """
    if not graph.nodes:
        raise ValueError("KnowledgeGraph must contain at least one node")

    _validate_unique_node_ids(graph)
    _validate_relationship_references(graph)
    _validate_relationship_type_constraints(graph)


def _validate_unique_node_ids(graph: KnowledgeGraph) -> None:
    """Check that all node IDs in the graph are unique.

    Args:
        graph: The KnowledgeGraph instance to validate.

    Raises:
        ValueError: If duplicate node IDs are found.
    """
    seen_ids: set[str] = set()
    duplicates: list[str] = []

    for node in graph.nodes:
        if node.id in seen_ids:
            duplicates.append(node.id)
        seen_ids.add(node.id)

    if duplicates:
        raise ValueError(f"Duplicate node IDs found: {duplicates}")


def _validate_relationship_references(graph: KnowledgeGraph) -> None:
    """Check that all relationship sources and targets reference existing nodes.

    Args:
        graph: The KnowledgeGraph instance to validate.

    Raises:
        ValueError: If a relationship references a non-existent node ID.
    """
    node_ids: set[str] = {node.id for node in graph.nodes}
    invalid_references: list[str] = []

    for rel in graph.relationships:
        if rel.source not in node_ids:
            invalid_references.append(f"source '{rel.source}' in relationship {rel}")
        if rel.target not in node_ids:
            invalid_references.append(f"target '{rel.target}' in relationship {rel}")

    if invalid_references:
        raise ValueError(f"Invalid node references in relationships: {invalid_references}")


def _validate_relationship_type_constraints(graph: KnowledgeGraph) -> None:
    """Check that relationship types have valid source/target node types.

    Args:
        graph: The KnowledgeGraph instance to validate.

    Raises:
        ValueError: If a HAS_RISK relationship target is not a RiskFactor.
    """
    node_map: dict[str, object] = {node.id: node for node in graph.nodes}

    for rel in graph.relationships:
        if rel.relation == "HAS_RISK" and node_map[rel.target].type != "RiskFactor":
            raise ValueError(
                f"HAS_RISK relationship target must be a RiskFactor, "
                f"but '{rel.target}' is a {node_map[rel.target].type}"
            )

