"""Mermaid diagram generation module.

This module provides functionality to render a KnowledgeGraph
as a Mermaid flowchart diagram for lightweight visualization.
"""

from pathlib import Path

from src.schema import KnowledgeGraph, Node

# Default output path for the Mermaid diagram
OUTPUT_PATH: Path = Path("visuals/graph.mmd")


def _get_node_shape(node: Node) -> str:
    """Get the Mermaid shape syntax for a node based on its type.

    Args:
        node: The node to get shape syntax for.

    Returns:
        A string with the node ID and label in appropriate Mermaid shape syntax.
        - Company: rectangle ["label"]
        - RiskFactor: rounded ("label")
        - DollarAmount: parallelogram [/"label"/]
    """
    # Escape quotes in node name for Mermaid compatibility
    escaped_name = node.name.replace('"', "'")

    if node.type == "Company":
        return f'{node.id}["{escaped_name}"]'
    elif node.type == "RiskFactor":
        return f'{node.id}("{escaped_name}")'
    elif node.type == "DollarAmount":
        return f'{node.id}[/"{escaped_name}"/]'
    else:
        # Fallback to rectangle
        return f'{node.id}["{escaped_name}"]'


def render_mermaid(graph: KnowledgeGraph, output_path: Path = OUTPUT_PATH) -> None:
    """Render a KnowledgeGraph as a Mermaid flowchart diagram.

    Generates a Mermaid flowchart TD (top-down) diagram from the
    KnowledgeGraph and saves it to a .mmd file.

    Node shapes:
    - Company: rectangle
    - RiskFactor: rounded
    - DollarAmount: parallelogram

    Args:
        graph: A validated KnowledgeGraph instance to visualize.
        output_path: Path where the Mermaid file will be saved.
            Defaults to visuals/graph.mmd.

    Returns:
        None. The Mermaid diagram is saved to the specified output path.

    Raises:
        OSError: If the output directory cannot be created or file cannot be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    # Mermaid flowchart header
    lines.append("flowchart TD")
    lines.append("")

    # Add node definitions
    lines.append("    %% Node definitions")
    for node in graph.nodes:
        node_syntax = _get_node_shape(node)
        lines.append(f"    {node_syntax}")

    lines.append("")

    # Add relationship edges
    lines.append("    %% Relationships")
    for rel in graph.relationships:
        lines.append(f"    {rel.source} -->|{rel.relation}| {rel.target}")

    # Write to file
    content = "\n".join(lines) + "\n"
    output_path.write_text(content, encoding="utf-8")


def _generate_mermaid_content(graph: KnowledgeGraph) -> str:
    """Generate Mermaid diagram content from a KnowledgeGraph.

    Args:
        graph: A validated KnowledgeGraph instance.

    Returns:
        Mermaid diagram content as a string.
    """
    lines: list[str] = []

    # Mermaid flowchart header
    lines.append("flowchart TD")
    lines.append("")

    # Add node definitions
    lines.append("    %% Node definitions")
    for node in graph.nodes:
        node_syntax = _get_node_shape(node)
        lines.append(f"    {node_syntax}")

    lines.append("")

    # Add relationship edges
    lines.append("    %% Relationships")
    for rel in graph.relationships:
        lines.append(f"    {rel.source} -->|{rel.relation}| {rel.target}")

    return "\n".join(lines)


def render_mermaid_html(graph: KnowledgeGraph, output_path: Path) -> None:
    """Render a KnowledgeGraph as an HTML page with embedded Mermaid diagram.

    Generates a self-contained HTML file that uses Mermaid.js from CDN
    to render the Knowledge Graph diagram in a browser.

    Args:
        graph: A validated KnowledgeGraph instance to visualize.
        output_path: Path where the HTML file will be saved.

    Returns:
        None. The HTML file is saved to the specified output path.

    Raises:
        OSError: If the output directory cannot be created or file cannot be written.

    Example:
        >>> render_mermaid_html(graph, Path("visuals/graph.html"))
        # Opens in browser to view the diagram
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate Mermaid diagram content
    mermaid_content = _generate_mermaid_content(graph)

    # HTML template with Mermaid.js CDN
    html_template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Financial Detective - Knowledge Graph</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: #f5f5f5;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        header {{
            background-color: #1a1a2e;
            color: white;
            padding: 1rem 2rem;
            text-align: center;
        }}
        header h1 {{
            font-size: 1.5rem;
            font-weight: 600;
        }}
        header p {{
            font-size: 0.875rem;
            opacity: 0.8;
            margin-top: 0.25rem;
        }}
        main {{
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
        }}
        .mermaid {{
            background-color: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            max-width: 100%;
            overflow: auto;
        }}
        footer {{
            background-color: #1a1a2e;
            color: white;
            padding: 0.75rem 2rem;
            text-align: center;
            font-size: 0.75rem;
            opacity: 0.8;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Financial Detective</h1>
        <p>Knowledge Graph Visualization</p>
    </header>
    <main>
        <pre class="mermaid">
{mermaid_content}
        </pre>
    </main>
    <footer>
        Generated by Financial Detective | Schema Version: {graph.schema_version}
    </footer>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            flowchart: {{
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis'
            }}
        }});
    </script>
</body>
</html>
'''

    output_path.write_text(html_template, encoding="utf-8")

