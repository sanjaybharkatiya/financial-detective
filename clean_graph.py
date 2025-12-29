"""Clean the knowledge graph by removing meaningless nodes."""

from pathlib import Path
import json
import re
from src.schema import KnowledgeGraph
from src.visualizer_mermaid import render_mermaid_html, render_mermaid


def is_meaningful_node(node) -> bool:
    """Check if node has meaningful content."""
    name = node.name.strip()
    context = (node.context or "").strip()
    
    # Remove nodes with empty or very short names
    if len(name) < 3:
        return False
    
    # Remove nodes that are just numbers (with commas, dots, spaces)
    clean_name = name.replace(",", "").replace(".", "").replace(" ", "").replace("-", "")
    if clean_name.isdigit():
        return False
    
    # Remove nodes like 'H 10', 'J 5,74,956', 'C 1,458' (single letter + space + numbers)
    if re.match(r"^[A-Z]\s+[\d,\.]+$", name):
        return False
    
    # Remove nodes that are just units without context
    if re.match(r"^[\d,\.~]+\s*(GW|GWh|MMTPA|MTPA|TPD|TPA|MW|MWh|acres?|lacs?\s*TPA|\+)h?$", name, re.IGNORECASE):
        if not context or len(context) < 10:
            return False
    
    # Keep dollar amounts that have currency symbols or good context
    if node.type == "DollarAmount":
        has_currency = any(c in name for c in ["$", "₹", "Rs", "USD", "INR", "crore", "billion", "million", "lakh"])
        has_context = context and len(context) > 5
        if not has_currency and not has_context:
            return False
    
    return True


def main():
    # Load existing graph
    graph_data = json.loads(Path("data/graph_output.json").read_text())
    graph = KnowledgeGraph.model_validate(graph_data)

    print(f"Before: {len(graph.nodes)} nodes, {len(graph.relationships)} relationships")

    # Filter meaningful nodes
    meaningful_nodes = [n for n in graph.nodes if is_meaningful_node(n)]
    removed_count = len(graph.nodes) - len(meaningful_nodes)
    print(f"Removed {removed_count} meaningless nodes")

    # Get IDs of remaining nodes
    valid_ids = set()
    for n in meaningful_nodes:
        valid_ids.add(n.id)

    # Filter relationships to only keep those with valid nodes
    valid_rels = [r for r in graph.relationships if r.source in valid_ids and r.target in valid_ids]
    removed_rels = len(graph.relationships) - len(valid_rels)
    print(f"Removed {removed_rels} relationships referencing removed nodes")

    # Now remove orphan nodes (nodes with no relationships)
    referenced_ids = set()
    for rel in valid_rels:
        referenced_ids.add(rel.source)
        referenced_ids.add(rel.target)

    connected_nodes = [n for n in meaningful_nodes if n.id in referenced_ids]
    orphan_count = len(meaningful_nodes) - len(connected_nodes)
    print(f"Removed {orphan_count} additional orphan nodes")

    print(f"After: {len(connected_nodes)} nodes, {len(valid_rels)} relationships")

    # Create cleaned graph
    cleaned = KnowledgeGraph(
        schema_version=graph.schema_version,
        nodes=connected_nodes,
        relationships=valid_rels,
    )

    # Save cleaned graph
    Path("data/graph_output.json").write_text(cleaned.model_dump_json(indent=2))

    # Regenerate visualizations
    render_mermaid(cleaned, Path("visuals/graph.mmd"))
    render_mermaid_html(cleaned, Path("visuals/graph.html"))

    print(f"Pages: {(len(connected_nodes) + 49) // 50}")
    print("Done!")


if __name__ == "__main__":
    main()

