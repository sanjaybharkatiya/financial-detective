"""Knowledge Graph validation module.

This module provides validation logic to ensure the integrity
of a KnowledgeGraph instance beyond Pydantic schema validation.

Includes auto-repair for common LLM extraction errors.
"""

from src.schema import KnowledgeGraph, Relationship


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
    # Relationship type constraints are now auto-fixed, not validated


def validate_and_repair_graph(graph: KnowledgeGraph) -> KnowledgeGraph:
    """Validate and auto-repair a KnowledgeGraph.
    
    Fixes common LLM extraction errors:
    - Removes relationships with invalid node type constraints
    - Removes relationships with missing node references
    
    Args:
        graph: The KnowledgeGraph instance to validate and repair.
        
    Returns:
        A new KnowledgeGraph with invalid relationships removed.
    """
    if not graph.nodes:
        raise ValueError("KnowledgeGraph must contain at least one node")
    
    # Build node map
    node_map = {node.id: node for node in graph.nodes}
    node_ids = set(node_map.keys())
    
    # Filter valid relationships
    valid_relationships: list[Relationship] = []
    removed_count = 0
    
    for rel in graph.relationships:
        # Check if source and target exist
        if rel.source not in node_ids or rel.target not in node_ids:
            removed_count += 1
            continue
        
        source_type = node_map[rel.source].type
        target_type = node_map[rel.target].type
        
        # Validate relationship type constraints
        is_valid = True
        
        if rel.relation == "HAS_RISK":
            # Target must be RiskFactor
            if target_type != "RiskFactor":
                is_valid = False
        elif rel.relation == "REPORTS_AMOUNT":
            # Target must be DollarAmount
            if target_type != "DollarAmount":
                is_valid = False
        elif rel.relation == "OWNS":
            # Both must be Company
            if source_type != "Company" or target_type != "Company":
                is_valid = False
        
        if is_valid:
            valid_relationships.append(rel)
        else:
            removed_count += 1
    
    if removed_count > 0:
        print(f"      ⚠️  Removed {removed_count} invalid relationships")
    
    return KnowledgeGraph(
        schema_version=graph.schema_version,
        nodes=graph.nodes,
        relationships=valid_relationships,
    )


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

