"""Knowledge Graph visualization module.

This module provides functionality to render a KnowledgeGraph
as a directed graph image using NetworkX and matplotlib.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from src.schema import KnowledgeGraph

# Node color mapping by type
NODE_COLORS: dict[str, str] = {
    "Company": "blue",
    "RiskFactor": "red",
    "DollarAmount": "green",
}

# Default output path for the rendered graph
OUTPUT_PATH: Path = Path("visuals/graph.png")


def render_graph(graph: KnowledgeGraph, output_path: Path = OUTPUT_PATH) -> None:
    """Render a KnowledgeGraph as a directed graph image.

    Builds a directed NetworkX graph from the KnowledgeGraph, applies
    node colors based on type, and saves the visualization to disk.

    Args:
        graph: A validated KnowledgeGraph instance to visualize.
        output_path: Path where the image will be saved.
            Defaults to visuals/graph.png.

    Returns:
        None. The graph image is saved to the specified output path.

    Raises:
        OSError: If the output directory cannot be created or file cannot be written.
    """
    G = _build_networkx_graph(graph)
    _render_and_save(G, graph, output_path)


def _build_networkx_graph(graph: KnowledgeGraph) -> nx.DiGraph:
    """Build a directed NetworkX graph from a KnowledgeGraph.

    Args:
        graph: The KnowledgeGraph instance to convert.

    Returns:
        A NetworkX DiGraph with nodes and edges populated.
    """
    G = nx.DiGraph()

    for node in graph.nodes:
        G.add_node(
            node.id,
            label=node.name,
            type=node.type,
        )

    for rel in graph.relationships:
        G.add_edge(
            rel.source,
            rel.target,
            label=rel.relation,
        )

    return G


def _render_and_save(
    G: nx.DiGraph,
    graph: KnowledgeGraph,
    output_path: Path,
) -> None:
    """Render the NetworkX graph and save to an image file.

    Args:
        G: The NetworkX DiGraph to render.
        graph: The original KnowledgeGraph (used for color mapping).
        output_path: Path where the image will be saved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    pos = nx.spring_layout(G, seed=42)

    node_colors = [NODE_COLORS.get(node.type, "gray") for node in graph.nodes]

    node_labels = {node.id: node.name for node in graph.nodes}

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=2000,
        alpha=0.9,
        ax=ax,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        font_size=8,
        font_weight="bold",
        ax=ax,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="gray",
        arrows=True,
        arrowsize=20,
        ax=ax,
    )

    edge_labels = {(rel.source, rel.target): rel.relation for rel in graph.relationships}
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=7,
        ax=ax,
    )

    ax.set_title("Financial Knowledge Graph", fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

