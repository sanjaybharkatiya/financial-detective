"""Mermaid diagram generation module.

This module provides functionality to render a KnowledgeGraph
as a Mermaid flowchart diagram for lightweight visualization.
"""

from pathlib import Path

from src.schema import KnowledgeGraph, Node

# Default output path for the Mermaid diagram
OUTPUT_PATH: Path = Path("visuals/graph.mmd")


def _escape_mermaid_label(text: str) -> str:
    """Escape special characters in text for Mermaid compatibility.
    
    Args:
        text: Raw text that may contain special characters.
        
    Returns:
        Escaped text safe for use in Mermaid labels.
    """
    # Replace characters that break Mermaid syntax
    text = text.replace('"', "'")
    text = text.replace('`', "'")
    text = text.replace('#', "")
    text = text.replace('&', "and")
    text = text.replace('<', "")
    text = text.replace('>', "")
    text = text.replace('[', "(")
    text = text.replace(']', ")")
    text = text.replace('{', "(")
    text = text.replace('}', ")")
    text = text.replace('|', "-")
    # Truncate very long labels
    if len(text) > 60:
        text = text[:57] + "..."
    return text


def _get_node_shape(node: Node) -> str:
    """Get the Mermaid shape syntax for a node based on its type.

    Args:
        node: The node to get shape syntax for.

    Returns:
        A string with the node ID and label in appropriate Mermaid shape syntax.
        - Company: rectangle ["label"]
        - RiskFactor: rounded ("label")
        - DollarAmount: parallelogram [/"label"/]
        
    If context is available, it's included in the label for clarity.
    """
    # Escape special characters in node name
    escaped_name = _escape_mermaid_label(node.name)
    
    # Include context if available to show what the value represents
    if node.context:
        escaped_context = _escape_mermaid_label(node.context)
        # For amounts, show context first (e.g., "Revenue: $38.7B")
        if node.type == "DollarAmount":
            label = f"{escaped_context}: {escaped_name}"
        else:
            # For others, show name with context in parentheses
            label = f"{escaped_name} ({escaped_context})"
    else:
        label = escaped_name
    
    # Final truncation for very long combined labels
    if len(label) > 80:
        label = label[:77] + "..."

    if node.type == "Company":
        return f'{node.id}["{label}"]'
    elif node.type == "RiskFactor":
        return f'{node.id}("{label}")'
    elif node.type == "DollarAmount":
        return f'{node.id}[/"{label}"/]'
    else:
        # Fallback to rectangle
        return f'{node.id}["{label}"]'


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


def _group_connected_nodes(graph: KnowledgeGraph, page_size: int = 50) -> list[list[str]]:
    """Group connected nodes together to minimize cross-page orphans.
    
    Uses BFS-based clustering to keep related nodes on the same page.
    Small clusters are consolidated into larger pages.
    
    Args:
        graph: The KnowledgeGraph to group.
        page_size: Maximum nodes per page.
        
    Returns:
        List of lists, each containing node IDs for a page.
    """
    from collections import defaultdict, deque
    
    # Build adjacency list
    adjacency: dict[str, set[str]] = defaultdict(set)
    for rel in graph.relationships:
        adjacency[rel.source].add(rel.target)
        adjacency[rel.target].add(rel.source)
    
    # Find connected components using BFS
    all_node_ids = {n.id for n in graph.nodes}
    assigned: set[str] = set()
    clusters: list[list[str]] = []
    
    # Sort nodes by connection count for seeding
    sorted_nodes = sorted(all_node_ids, key=lambda nid: len(adjacency.get(nid, set())), reverse=True)
    
    for seed_id in sorted_nodes:
        if seed_id in assigned:
            continue
        
        # BFS to find all connected nodes
        cluster: list[str] = []
        queue: deque[str] = deque([seed_id])
        
        while queue:
            node_id = queue.popleft()
            if node_id in assigned:
                continue
            
            cluster.append(node_id)
            assigned.add(node_id)
            
            for neighbor in adjacency.get(node_id, set()):
                if neighbor not in assigned:
                    queue.append(neighbor)
        
        if cluster:
            clusters.append(cluster)
    
    # Sort clusters by size (largest first)
    clusters.sort(key=len, reverse=True)
    
    # Now pack clusters into pages
    pages: list[list[str]] = []
    current_page: list[str] = []
    
    for cluster in clusters:
        if len(cluster) > page_size:
            # Large cluster: split into multiple pages
            if current_page:
                pages.append(current_page)
                current_page = []
            
            for i in range(0, len(cluster), page_size):
                pages.append(cluster[i:i + page_size])
        elif len(current_page) + len(cluster) <= page_size:
            # Cluster fits in current page
            current_page.extend(cluster)
        else:
            # Start new page
            if current_page:
                pages.append(current_page)
            current_page = cluster.copy()
    
    if current_page:
        pages.append(current_page)
    
    return pages


def _generate_paginated_mermaid(graph: KnowledgeGraph, page_size: int = 50) -> list[str]:
    """Generate paginated Mermaid diagrams for large graphs.
    
    Uses smart grouping to keep connected nodes on the same page.
    Shows all relationships between nodes on the page, plus limited
    cross-page connections with ghost nodes to ensure no node appears orphaned.
    
    Args:
        graph: The KnowledgeGraph to paginate.
        page_size: Maximum nodes per page.
        
    Returns:
        List of Mermaid diagram strings, one per page.
    """
    from collections import defaultdict
    
    # Group connected nodes together
    page_groups = _group_connected_nodes(graph, page_size)
    
    # Build node map for quick lookup
    node_map = {n.id: n for n in graph.nodes}
    
    # Build adjacency for checking orphans
    adjacency: dict[str, set[str]] = defaultdict(set)
    for rel in graph.relationships:
        adjacency[rel.source].add(rel.target)
        adjacency[rel.target].add(rel.source)
    
    pages = []
    for page_node_ids in page_groups:
        page_id_set = set(page_node_ids)
        
        # Get relationships where both source and target are in this page
        page_rels = [
            r for r in graph.relationships
            if r.source in page_id_set and r.target in page_id_set
        ]
        
        # Find nodes that would appear orphaned (have connections but none on this page)
        orphan_nodes = []
        for node_id in page_node_ids:
            connections = adjacency.get(node_id, set())
            same_page = connections & page_id_set
            if connections and not same_page:
                orphan_nodes.append(node_id)
        
        # For orphaned nodes, add ONE ghost connection to show they're connected
        ghost_nodes: set[str] = set()
        ghost_rels = []
        max_ghosts = min(15, len(orphan_nodes))  # Limit ghost nodes per page
        
        for node_id in orphan_nodes[:max_ghosts]:
            # Get first connection to a node not on this page
            for rel in graph.relationships:
                if rel.source == node_id and rel.target not in page_id_set:
                    ghost_nodes.add(rel.target)
                    ghost_rels.append(rel)
                    break
                elif rel.target == node_id and rel.source not in page_id_set:
                    ghost_nodes.add(rel.source)
                    ghost_rels.append(rel)
                    break
        
        lines = ["flowchart TD", ""]
        
        # Add style for ghost nodes
        if ghost_nodes:
            lines.append("    %% Style for nodes from other pages")
            lines.append("    classDef ghost fill:#e8e8e8,stroke:#999,stroke-dasharray: 5 5")
        
        lines.append("")
        lines.append("    %% Node definitions")
        for node_id in page_node_ids:
            if node_id in node_map:
                lines.append(f"    {_get_node_shape(node_map[node_id])}")
        
        # Add ghost nodes
        for ghost_id in ghost_nodes:
            if ghost_id in node_map:
                ghost_node = node_map[ghost_id]
                name = _escape_mermaid_label(ghost_node.name)
                if len(name) > 25:
                    name = name[:22] + "..."
                lines.append(f'    {ghost_id}["{name} (other page)"]:::ghost')
        
        lines.append("")
        lines.append("    %% Relationships")
        for rel in page_rels:
            lines.append(f"    {rel.source} -->|{rel.relation}| {rel.target}")
        
        # Add ghost relationships
        if ghost_rels:
            lines.append("")
            lines.append("    %% Cross-page connections")
            for rel in ghost_rels:
                lines.append(f"    {rel.source} -.->|{rel.relation}| {rel.target}")
        
        # Add note if there are more orphans than shown
        if len(orphan_nodes) > max_ghosts:
            remaining = len(orphan_nodes) - max_ghosts
            lines.append("")
            lines.append(f"    %% Note: {remaining} more nodes have cross-page connections")
        
        pages.append("\n".join(lines))
    
    return pages


def render_mermaid_html(graph: KnowledgeGraph, output_path: Path) -> None:
    """Render a KnowledgeGraph as an HTML page with embedded Mermaid diagram.

    Creates a full horizontal scrollable view with zoom and pan controls.
    Works for both small and large graphs.

    Args:
        graph: A validated KnowledgeGraph instance to visualize.
        output_path: Path where the HTML file will be saved.

    Returns:
        None. The HTML file is saved to the specified output path.

    Raises:
        OSError: If the output directory cannot be created or file cannot be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    node_count = len(graph.nodes)
    rel_count = len(graph.relationships)
    
    # Use the full horizontal scrollable view for all graphs
    _render_fullgraph_html(graph, output_path)
    return

    # For smaller graphs, render full diagram
    mermaid_content = _generate_mermaid_content(graph)

    # Count stats for display
    node_count = len(graph.nodes)
    rel_count = len(graph.relationships)
    
    # HTML template - scrollable full-page layout for large graphs
    html_template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Financial Detective - Knowledge Graph ({node_count} nodes)</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        html {{
            scroll-behavior: smooth;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
        }}
        
        /* Fixed header */
        header {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        header h1 {{
            font-size: 1.5rem;
            font-weight: 700;
        }}
        .stats {{
            display: flex;
            gap: 1.5rem;
            font-size: 0.9rem;
        }}
        .stat {{
            background: rgba(255,255,255,0.15);
            padding: 0.5rem 1rem;
            border-radius: 6px;
        }}
        .stat-value {{
            font-weight: 700;
            font-size: 1.1rem;
        }}
        
        /* Controls bar */
        .controls {{
            position: fixed;
            top: 70px;
            left: 0;
            right: 0;
            z-index: 999;
            background: white;
            padding: 0.75rem 2rem;
            display: flex;
            gap: 1rem;
            align-items: center;
            border-bottom: 1px solid #ddd;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .controls button {{
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s;
        }}
        .controls button:hover {{
            transform: translateY(-1px);
        }}
        .btn-primary {{
            background: #2563eb;
            color: white;
        }}
        .btn-secondary {{
            background: #e5e7eb;
            color: #374151;
        }}
        .zoom-info {{
            margin-left: auto;
            color: #666;
            font-size: 0.85rem;
        }}
        
        /* Main diagram area - scrollable */
        main {{
            margin-top: 130px;
            padding: 2rem;
            min-height: calc(100vh - 130px);
        }}
        .diagram-wrapper {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            padding: 2rem;
            overflow: auto;
        }}
        .mermaid {{
            min-width: max-content;
        }}
        .mermaid svg {{
            max-width: none !important;
            height: auto !important;
        }}
        
        /* Node styling */
        .node rect {{
            stroke-width: 2px !important;
        }}
        .node .label {{
            font-size: 14px !important;
        }}
        .edgeLabel {{
            font-size: 12px !important;
            font-weight: 600 !important;
            background: white !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
        }}
        
        /* Legend */
        .legend {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            font-size: 0.85rem;
            z-index: 100;
        }}
        .legend h3 {{
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
            color: #666;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin: 0.3rem 0;
        }}
        .legend-shape {{
            width: 20px;
            height: 14px;
            border: 2px solid #333;
        }}
        .shape-rect {{ border-radius: 2px; }}
        .shape-rounded {{ border-radius: 7px; }}
        .shape-parallelogram {{ 
            transform: skewX(-10deg);
            width: 24px;
        }}
        
        /* Scroll indicator */
        .scroll-hint {{
            text-align: center;
            padding: 1rem;
            color: #666;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <header>
        <h1>📊 Financial Detective</h1>
        <div class="stats">
            <div class="stat">
                <span class="stat-value">{node_count}</span> Nodes
            </div>
            <div class="stat">
                <span class="stat-value">{rel_count}</span> Relationships
            </div>
        </div>
    </header>
    
    <div class="controls">
        <button class="btn-primary" onclick="zoomIn()">🔍 Zoom In</button>
        <button class="btn-primary" onclick="zoomOut()">🔍 Zoom Out</button>
        <button class="btn-secondary" onclick="resetZoom()">↺ Reset</button>
        <button class="btn-secondary" onclick="scrollToTop()">⬆ Top</button>
        <button class="btn-secondary" onclick="scrollToBottom()">⬇ Bottom</button>
        <span class="zoom-info">Scroll to navigate • Use zoom buttons to resize</span>
    </div>
    
    <main>
        <div class="scroll-hint">⬇ Scroll down to explore the full graph ⬇</div>
        <div class="diagram-wrapper" id="diagram">
            <pre class="mermaid" id="mermaid-diagram">
{mermaid_content}
            </pre>
        </div>
        <div class="scroll-hint">⬆ Scroll up to return to the top ⬆</div>
    </main>
    
    <div class="legend">
        <h3>Legend</h3>
        <div class="legend-item">
            <div class="legend-shape shape-rect"></div>
            <span>Company</span>
        </div>
        <div class="legend-item">
            <div class="legend-shape shape-rounded"></div>
            <span>Risk Factor</span>
        </div>
        <div class="legend-item">
            <div class="legend-shape shape-parallelogram"></div>
            <span>Dollar Amount</span>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        let currentZoom = 1;
        
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            maxTextSize: 500000,
            maxEdges: 5000,
            flowchart: {{
                useMaxWidth: false,
                htmlLabels: true,
                curve: 'basis',
                nodeSpacing: 50,
                rankSpacing: 70,
                padding: 15
            }},
            themeVariables: {{
                fontSize: '14px',
                fontFamily: '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif'
            }}
        }});
        
        function zoomIn() {{
            currentZoom = Math.min(currentZoom + 0.2, 3);
            applyZoom();
        }}
        
        function zoomOut() {{
            currentZoom = Math.max(currentZoom - 0.2, 0.3);
            applyZoom();
        }}
        
        function resetZoom() {{
            currentZoom = 1;
            applyZoom();
        }}
        
        function applyZoom() {{
            const svg = document.querySelector('.mermaid svg');
            if (svg) {{
                svg.style.transform = `scale(${{currentZoom}})`;
                svg.style.transformOrigin = 'top left';
            }}
        }}
        
        function scrollToTop() {{
            window.scrollTo({{ top: 0, behavior: 'smooth' }});
        }}
        
        function scrollToBottom() {{
            window.scrollTo({{ top: document.body.scrollHeight, behavior: 'smooth' }});
        }}
    </script>
</body>
</html>
'''

    output_path.write_text(html_template, encoding="utf-8")


def _render_fullgraph_html(graph: KnowledgeGraph, output_path: Path) -> None:
    """Render the full graph as a horizontal scrollable HTML with zoom controls.
    
    Uses a modern dark theme with enhanced zoom/pan controls, auto-fit on load,
    and keyboard navigation support.
    
    Args:
        graph: The full KnowledgeGraph.
        output_path: Path where the HTML file will be saved.
    """
    node_count = len(graph.nodes)
    rel_count = len(graph.relationships)
    
    # Generate the full Mermaid content with horizontal (LR) layout
    mermaid_content = _generate_mermaid_content_horizontal(graph)
    
    html_template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Financial Detective - Knowledge Graph ({node_count} nodes)</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html, body {{
            height: 100%;
            width: 100%;
            overflow: hidden;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            display: flex;
            flex-direction: column;
        }}
        
        /* Compact header */
        header {{
            background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
            color: white;
            padding: 0.5rem 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.5);
            flex-shrink: 0;
            border-bottom: 1px solid #30363d;
        }}
        header h1 {{ 
            font-size: 1rem; 
            font-weight: 600;
            color: #58a6ff;
        }}
        .stats {{
            display: flex;
            gap: 0.75rem;
            font-size: 0.75rem;
        }}
        .stat {{
            background: rgba(88,166,255,0.15);
            padding: 0.25rem 0.6rem;
            border-radius: 4px;
            border: 1px solid rgba(88,166,255,0.3);
        }}
        .stat-value {{ font-weight: 700; color: #58a6ff; }}
        
        /* Controls bar - more compact */
        .controls {{
            background: #161b22;
            padding: 0.4rem 1rem;
            display: flex;
            gap: 0.5rem;
            align-items: center;
            border-bottom: 1px solid #30363d;
            flex-shrink: 0;
            flex-wrap: wrap;
        }}
        .controls button {{
            padding: 0.35rem 0.7rem;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.75rem;
            font-weight: 500;
            transition: all 0.15s;
        }}
        .controls button:hover {{
            transform: translateY(-1px);
        }}
        .btn-zoom {{
            background: #238636;
            color: white;
        }}
        .btn-zoom:hover {{
            background: #2ea043;
        }}
        .btn-secondary {{
            background: #21262d;
            color: #c9d1d9;
            border: 1px solid #30363d;
        }}
        .btn-secondary:hover {{
            background: #30363d;
        }}
        .btn-fit {{
            background: #1f6feb;
            color: white;
        }}
        .btn-fit:hover {{
            background: #388bfd;
        }}
        .zoom-display {{
            background: #0d1117;
            color: #58a6ff;
            padding: 0.3rem 0.6rem;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.75rem;
            min-width: 55px;
            text-align: center;
            border: 1px solid #30363d;
        }}
        .separator {{
            width: 1px;
            height: 20px;
            background: #30363d;
            margin: 0 0.25rem;
        }}
        .hint {{
            margin-left: auto;
            color: #8b949e;
            font-size: 0.7rem;
        }}
        .zoom-slider {{
            width: 100px;
            height: 4px;
            -webkit-appearance: none;
            background: #30363d;
            border-radius: 2px;
            outline: none;
        }}
        .zoom-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 14px;
            height: 14px;
            background: #58a6ff;
            border-radius: 50%;
            cursor: pointer;
        }}
        
        /* Main diagram area - full viewport */
        .diagram-wrapper {{
            flex: 1;
            overflow: auto;
            background: #0d1117;
            cursor: grab;
            position: relative;
        }}
        .diagram-wrapper:active {{
            cursor: grabbing;
        }}
        .diagram-container {{
            display: inline-block;
            padding: 1rem;
            min-width: 100%;
            min-height: 100%;
            transform-origin: 0 0;
            transition: transform 0.05s ease-out;
        }}
        .mermaid {{
            background: #161b22;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 0 30px rgba(0,0,0,0.5);
            display: inline-block;
            border: 1px solid #30363d;
        }}
        .mermaid svg {{
            max-width: none !important;
            height: auto !important;
        }}
        /* Style the SVG nodes for dark theme */
        .mermaid .node rect, .mermaid .node polygon {{
            fill: #21262d !important;
            stroke: #58a6ff !important;
        }}
        .mermaid .node .label {{
            color: #c9d1d9 !important;
        }}
        .mermaid .edgePath path {{
            stroke: #8b949e !important;
        }}
        .mermaid .edgeLabel {{
            background-color: #161b22 !important;
            color: #8b949e !important;
        }}
        
        /* Floating Legend - collapsible */
        .legend {{
            position: fixed;
            bottom: 15px;
            right: 15px;
            background: #161b22;
            padding: 0.6rem;
            border-radius: 6px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            font-size: 0.7rem;
            z-index: 100;
            border: 1px solid #30363d;
            max-width: 140px;
        }}
        .legend h3 {{ 
            margin-bottom: 0.4rem; 
            color: #58a6ff; 
            font-size: 0.75rem;
            font-weight: 600;
            cursor: pointer;
        }}
        .legend-item {{ 
            display: flex; 
            align-items: center; 
            gap: 0.4rem; 
            margin: 0.2rem 0;
            color: #8b949e;
        }}
        .legend-shape {{ 
            width: 14px; 
            height: 10px; 
            border: 1.5px solid #58a6ff; 
            background: #21262d;
        }}
        .shape-rect {{ border-radius: 2px; }}
        .shape-rounded {{ border-radius: 4px; }}
        .shape-parallelogram {{ transform: skewX(-10deg); width: 18px; }}
        
        /* Loading */
        .loading {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #8b949e;
            font-size: 1rem;
            gap: 1rem;
        }}
        .loading-spinner {{
            width: 40px;
            height: 40px;
            border: 3px solid #30363d;
            border-top-color: #58a6ff;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }}
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        
        /* Quick zoom panel */
        .quick-zoom {{
            position: fixed;
            left: 15px;
            bottom: 15px;
            background: #161b22;
            padding: 0.5rem;
            border-radius: 6px;
            border: 1px solid #30363d;
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
            z-index: 100;
        }}
        .quick-zoom button {{
            width: 32px;
            height: 32px;
            border: none;
            background: #21262d;
            color: #c9d1d9;
            border-radius: 4px;
            cursor: pointer;
            font-size: 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .quick-zoom button:hover {{
            background: #30363d;
        }}
    </style>
</head>
<body>
    <header>
        <h1>📊 Financial Detective - Knowledge Graph</h1>
        <div class="stats">
            <div class="stat"><span class="stat-value">{node_count}</span> Nodes</div>
            <div class="stat"><span class="stat-value">{rel_count}</span> Relationships</div>
        </div>
    </header>
    
    <div class="controls">
        <button class="btn-zoom" onclick="zoomIn()">+ Zoom In</button>
        <button class="btn-zoom" onclick="zoomOut()">− Zoom Out</button>
        <input type="range" class="zoom-slider" id="zoomSlider" min="5" max="200" value="100" oninput="setZoomFromSlider(this.value)">
        <span class="zoom-display" id="zoomLevel">100%</span>
        <div class="separator"></div>
        <button class="btn-fit" onclick="fitToScreen()">⊡ Fit to Screen</button>
        <button class="btn-secondary" onclick="resetZoom()">↺ Reset 100%</button>
        <div class="separator"></div>
        <button class="btn-secondary" onclick="setZoom(0.1)">10%</button>
        <button class="btn-secondary" onclick="setZoom(0.25)">25%</button>
        <button class="btn-secondary" onclick="setZoom(0.5)">50%</button>
        <span class="hint">Drag to pan • Ctrl+Scroll to zoom • Arrow keys to navigate</span>
    </div>
    
    <div class="quick-zoom">
        <button onclick="zoomIn()" title="Zoom In">+</button>
        <button onclick="zoomOut()" title="Zoom Out">−</button>
        <button onclick="fitToScreen()" title="Fit to Screen">⊡</button>
        <button onclick="resetZoom()" title="Reset to 100%">↺</button>
    </div>
    
    <div class="diagram-wrapper" id="wrapper">
        <div class="diagram-container" id="container">
            <div class="loading" id="loading">
                <div class="loading-spinner"></div>
                <div>Rendering {node_count} nodes...</div>
            </div>
            <pre class="mermaid" id="diagram" style="display:none;">
{mermaid_content}
            </pre>
        </div>
    </div>
    
    <div class="legend">
        <h3>Node Types</h3>
        <div class="legend-item"><div class="legend-shape shape-rect"></div><span>Company</span></div>
        <div class="legend-item"><div class="legend-shape shape-rounded"></div><span>Risk Factor</span></div>
        <div class="legend-item"><div class="legend-shape shape-parallelogram"></div><span>Dollar Amount</span></div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        let zoom = 1;
        const minZoom = 0.05;
        const maxZoom = 2;
        const zoomStep = 0.05;
        
        const wrapper = document.getElementById('wrapper');
        const container = document.getElementById('container');
        const loading = document.getElementById('loading');
        const diagram = document.getElementById('diagram');
        const zoomSlider = document.getElementById('zoomSlider');
        
        // Initialize Mermaid with dark theme
        mermaid.initialize({{
            startOnLoad: false,
            theme: 'dark',
            maxTextSize: 2000000,
            maxEdges: 20000,
            flowchart: {{
                useMaxWidth: false,
                htmlLabels: true,
                curve: 'basis',
                nodeSpacing: 20,
                rankSpacing: 40,
                padding: 15
            }},
            themeVariables: {{
                fontSize: '10px',
                primaryColor: '#21262d',
                primaryTextColor: '#c9d1d9',
                primaryBorderColor: '#58a6ff',
                lineColor: '#8b949e',
                secondaryColor: '#30363d',
                tertiaryColor: '#161b22'
            }}
        }});
        
        // Render the diagram
        async function renderDiagram() {{
            try {{
                const content = diagram.textContent.trim();
                const {{ svg }} = await mermaid.render('rendered-graph', content);
                diagram.innerHTML = svg;
                diagram.style.display = 'block';
                loading.style.display = 'none';
                
                // Auto-fit to screen after render with slight delay
                setTimeout(() => {{
                    fitToScreen();
                    wrapper.scrollTo(0, 0);
                }}, 200);
            }} catch (e) {{
                loading.innerHTML = '<div style="color:#f85149;">Error rendering: ' + e.message + '</div><div style="margin-top:1rem;font-size:0.8rem;color:#8b949e;">Try refreshing the page or use a different browser.</div>';
            }}
        }}
        
        // Update zoom display and slider
        function updateZoom() {{
            container.style.transform = `scale(${{zoom}})`;
            const percent = Math.round(zoom * 100);
            document.getElementById('zoomLevel').textContent = percent + '%';
            zoomSlider.value = percent;
        }}
        
        // Set zoom directly
        function setZoom(newZoom) {{
            zoom = Math.max(minZoom, Math.min(maxZoom, newZoom));
            updateZoom();
        }}
        
        // Set zoom from slider
        function setZoomFromSlider(value) {{
            setZoom(value / 100);
        }}
        
        function zoomIn() {{
            const step = zoom < 0.2 ? 0.02 : (zoom < 0.5 ? 0.05 : 0.1);
            setZoom(zoom + step);
        }}
        
        function zoomOut() {{
            const step = zoom < 0.2 ? 0.02 : (zoom < 0.5 ? 0.05 : 0.1);
            setZoom(zoom - step);
        }}
        
        function resetZoom() {{
            zoom = 1;
            updateZoom();
            wrapper.scrollTo(0, 0);
        }}
        
        function fitToScreen() {{
            const svg = diagram.querySelector('svg');
            if (!svg) return;
            
            const bbox = svg.getBBox ? svg.getBBox() : null;
            const svgWidth = bbox ? bbox.width : (svg.viewBox?.baseVal?.width || svg.clientWidth);
            const svgHeight = bbox ? bbox.height : (svg.viewBox?.baseVal?.height || svg.clientHeight);
            
            const wrapperWidth = wrapper.clientWidth - 40;
            const wrapperHeight = wrapper.clientHeight - 40;
            
            const scaleX = wrapperWidth / svgWidth;
            const scaleY = wrapperHeight / svgHeight;
            
            zoom = Math.max(minZoom, Math.min(scaleX, scaleY, 1));
            
            updateZoom();
            
            setTimeout(() => {{
                const containerWidth = container.scrollWidth * zoom;
                const containerHeight = container.scrollHeight * zoom;
                const scrollX = Math.max(0, (containerWidth - wrapperWidth) / 2);
                const scrollY = Math.max(0, (containerHeight - wrapperHeight) / 2);
                wrapper.scrollTo(scrollX, scrollY);
            }}, 50);
        }}
        
        // Scroll navigation
        function scrollTo(direction) {{
            const amount = 300;
            switch(direction) {{
                case 'left': wrapper.scrollBy(-amount, 0); break;
                case 'right': wrapper.scrollBy(amount, 0); break;
                case 'top': wrapper.scrollBy(0, -amount); break;
                case 'bottom': wrapper.scrollBy(0, amount); break;
            }}
        }}
        
        // Mouse wheel zoom (with Ctrl or Meta key)
        wrapper.addEventListener('wheel', (e) => {{
            if (e.ctrlKey || e.metaKey) {{
                e.preventDefault();
                const delta = e.deltaY > 0 ? -1 : 1;
                const step = zoom < 0.2 ? 0.01 : (zoom < 0.5 ? 0.02 : 0.05);
                setZoom(zoom + (delta * step));
            }}
        }}, {{ passive: false }});
        
        // Drag to pan
        let isDragging = false;
        let startX, startY, scrollLeft, scrollTop;
        
        wrapper.addEventListener('mousedown', (e) => {{
            isDragging = true;
            startX = e.pageX - wrapper.offsetLeft;
            startY = e.pageY - wrapper.offsetTop;
            scrollLeft = wrapper.scrollLeft;
            scrollTop = wrapper.scrollTop;
            wrapper.style.cursor = 'grabbing';
        }});
        
        wrapper.addEventListener('mouseleave', () => {{
            isDragging = false;
            wrapper.style.cursor = 'grab';
        }});
        
        wrapper.addEventListener('mouseup', () => {{
            isDragging = false;
            wrapper.style.cursor = 'grab';
        }});
        
        wrapper.addEventListener('mousemove', (e) => {{
            if (!isDragging) return;
            e.preventDefault();
            const x = e.pageX - wrapper.offsetLeft;
            const y = e.pageY - wrapper.offsetTop;
            wrapper.scrollLeft = scrollLeft - (x - startX);
            wrapper.scrollTop = scrollTop - (y - startY);
        }});
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {{
            if (e.target.tagName === 'INPUT') return;
            
            const scrollAmount = e.shiftKey ? 200 : 50;
            
            switch(e.key) {{
                case '+': case '=': zoomIn(); e.preventDefault(); break;
                case '-': case '_': zoomOut(); e.preventDefault(); break;
                case '0': resetZoom(); e.preventDefault(); break;
                case 'f': case 'F': fitToScreen(); e.preventDefault(); break;
                case 'ArrowLeft': wrapper.scrollBy(-scrollAmount, 0); e.preventDefault(); break;
                case 'ArrowRight': wrapper.scrollBy(scrollAmount, 0); e.preventDefault(); break;
                case 'ArrowUp': wrapper.scrollBy(0, -scrollAmount); e.preventDefault(); break;
                case 'ArrowDown': wrapper.scrollBy(0, scrollAmount); e.preventDefault(); break;
                case 'Home': wrapper.scrollTo(0, 0); e.preventDefault(); break;
                case 'End': wrapper.scrollTo(wrapper.scrollWidth, wrapper.scrollHeight); e.preventDefault(); break;
            }}
        }});
        
        // Touch support for mobile
        let touchStartX, touchStartY;
        wrapper.addEventListener('touchstart', (e) => {{
            if (e.touches.length === 1) {{
                touchStartX = e.touches[0].pageX;
                touchStartY = e.touches[0].pageY;
            }}
        }}, {{ passive: true }});
        
        wrapper.addEventListener('touchmove', (e) => {{
            if (e.touches.length === 1) {{
                const dx = touchStartX - e.touches[0].pageX;
                const dy = touchStartY - e.touches[0].pageY;
                wrapper.scrollBy(dx, dy);
                touchStartX = e.touches[0].pageX;
                touchStartY = e.touches[0].pageY;
            }}
        }}, {{ passive: true }});
        
        // Start rendering
        renderDiagram();
    </script>
</body>
</html>
'''
    
    output_path.write_text(html_template, encoding="utf-8")


def _generate_mermaid_content_horizontal(graph: KnowledgeGraph) -> str:
    """Generate Mermaid diagram content with horizontal (LR) layout.

    Args:
        graph: A validated KnowledgeGraph instance.

    Returns:
        Mermaid diagram content as a string with left-to-right layout.
    """
    lines: list[str] = []

    # Mermaid flowchart header - Left to Right for horizontal scrolling
    lines.append("flowchart LR")
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

