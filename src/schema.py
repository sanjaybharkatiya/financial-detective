"""Knowledge Graph schema definitions using Pydantic v2.

This module defines the strict schema for the Knowledge Graph, including
Node and Relationship models, as well as the root KnowledgeGraph container.

All models enforce strict validation with no optional or extra fields allowed.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class Node(BaseModel):
    """Represents a node in the Knowledge Graph.

    Attributes:
        id: Unique identifier for the node.
        type: The category of the node. Must be one of:
            "Company", "RiskFactor", or "DollarAmount".
        name: Human-readable name or label for the node.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["Company", "RiskFactor", "DollarAmount"]
    name: str


class Relationship(BaseModel):
    """Represents a directed relationship between two nodes.

    Attributes:
        source: The id of the source node.
        target: The id of the target node.
        relation: The type of relationship. Must be one of:
            "OWNS", "HAS_RISK", or "REPORTS_AMOUNT".
        confidence: Optional confidence score (0.0 to 1.0) for the relationship.
            Higher values indicate stronger evidence in the source text.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    relation: Literal["OWNS", "HAS_RISK", "REPORTS_AMOUNT"]
    confidence: float | None = None


class KnowledgeGraph(BaseModel):
    """Root container for the Knowledge Graph.

    Attributes:
        schema_version: Version string for the schema (e.g., "1.0.0").
        nodes: List of all nodes in the graph.
        relationships: List of all relationships connecting nodes.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    nodes: list[Node]
    relationships: list[Relationship]

