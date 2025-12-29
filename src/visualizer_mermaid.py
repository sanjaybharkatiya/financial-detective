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

    For large graphs (>100 nodes), creates a paginated view.
    For smaller graphs, renders the full diagram.

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
    
    # For large graphs, use pagination
    if node_count > 100:
        pages = _generate_paginated_mermaid(graph, page_size=50)
        _render_paginated_html(graph, pages, output_path)
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


def _render_paginated_html(graph: KnowledgeGraph, pages: list[str], output_path: Path) -> None:
    """Render a paginated HTML view for large graphs.
    
    Args:
        graph: The full KnowledgeGraph for stats.
        pages: List of Mermaid diagram strings, one per page.
        output_path: Path where the HTML file will be saved.
    """
    import json as json_module
    
    node_count = len(graph.nodes)
    rel_count = len(graph.relationships)
    total_pages = len(pages)
    
    # Escape pages for JavaScript
    pages_json = json_module.dumps(pages)
    
    html_template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Financial Detective - Knowledge Graph ({node_count} nodes)</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f0f2f5;
            min-height: 100vh;
        }}
        
        header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 1000;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        header h1 {{ font-size: 1.4rem; }}
        .stats {{
            display: flex;
            gap: 1rem;
            font-size: 0.9rem;
        }}
        .stat {{
            background: rgba(255,255,255,0.15);
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
        }}
        .stat-value {{ font-weight: 700; }}
        
        .pagination {{
            background: white;
            padding: 1rem 2rem;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 1rem;
            border-bottom: 1px solid #ddd;
            position: sticky;
            top: 60px;
            z-index: 999;
        }}
        .pagination button {{
            padding: 0.6rem 1.2rem;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s;
        }}
        .pagination button:hover:not(:disabled) {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .pagination button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        .btn-nav {{
            background: #2563eb;
            color: white;
        }}
        .page-info {{
            font-weight: 600;
            color: #374151;
            min-width: 120px;
            text-align: center;
        }}
        .page-select {{
            padding: 0.5rem;
            border-radius: 6px;
            border: 1px solid #ddd;
            font-size: 0.9rem;
        }}
        
        main {{
            padding: 2rem;
            max-width: 100%;
            overflow-x: auto;
        }}
        .diagram-container {{
            background: white;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            min-height: 500px;
            overflow: auto;
        }}
        .mermaid {{
            min-width: max-content;
        }}
        .mermaid svg {{
            max-width: none !important;
        }}
        
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
        .legend h3 {{ margin-bottom: 0.5rem; color: #666; font-size: 0.9rem; }}
        .legend-item {{ display: flex; align-items: center; gap: 0.5rem; margin: 0.3rem 0; }}
        .legend-shape {{ width: 20px; height: 14px; border: 2px solid #333; }}
        .shape-rect {{ border-radius: 2px; }}
        .shape-rounded {{ border-radius: 7px; }}
        .shape-parallelogram {{ transform: skewX(-10deg); width: 24px; }}
        
        .loading {{
            text-align: center;
            padding: 3rem;
            color: #666;
            font-size: 1.2rem;
        }}
    </style>
</head>
<body>
    <header>
        <h1>📊 Financial Detective</h1>
        <div class="stats">
            <div class="stat"><span class="stat-value">{node_count}</span> Nodes</div>
            <div class="stat"><span class="stat-value">{rel_count}</span> Relationships</div>
            <div class="stat"><span class="stat-value">{total_pages}</span> Pages</div>
        </div>
    </header>
    
    <div class="pagination">
        <button class="btn-nav" onclick="firstPage()">⏮ First</button>
        <button class="btn-nav" onclick="prevPage()">◀ Previous</button>
        <span class="page-info">Page <span id="currentPage">1</span> of {total_pages}</span>
        <select class="page-select" id="pageSelect" onchange="goToPage(this.value)">
            {"".join(f'<option value="{i}">{i+1}</option>' for i in range(total_pages))}
        </select>
        <button class="btn-nav" onclick="nextPage()">Next ▶</button>
        <button class="btn-nav" onclick="lastPage()">Last ⏭</button>
    </div>
    
    <main>
        <div class="diagram-container">
            <div id="diagram-content" class="loading">Loading diagram...</div>
        </div>
    </main>
    
    <div class="legend">
        <h3>Legend</h3>
        <div class="legend-item"><div class="legend-shape shape-rect"></div><span>Company</span></div>
        <div class="legend-item"><div class="legend-shape shape-rounded"></div><span>Risk Factor</span></div>
        <div class="legend-item"><div class="legend-shape shape-parallelogram"></div><span>Dollar Amount</span></div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        const pages = {pages_json};
        let currentPageIndex = 0;
        const totalPages = {total_pages};
        
        mermaid.initialize({{
            startOnLoad: false,
            theme: 'default',
            maxTextSize: 100000,
            flowchart: {{
                useMaxWidth: false,
                htmlLabels: true,
                curve: 'basis',
                nodeSpacing: 50,
                rankSpacing: 70
            }}
        }});
        
        async function renderPage(index) {{
            const container = document.getElementById('diagram-content');
            container.innerHTML = '<div class="loading">Rendering page ' + (index + 1) + '...</div>';
            
            try {{
                const {{ svg }} = await mermaid.render('mermaid-' + index, pages[index]);
                container.innerHTML = svg;
            }} catch (e) {{
                container.innerHTML = '<div class="loading">Error rendering diagram: ' + e.message + '</div>';
            }}
            
            document.getElementById('currentPage').textContent = index + 1;
            document.getElementById('pageSelect').value = index;
        }}
        
        function nextPage() {{
            if (currentPageIndex < totalPages - 1) {{
                currentPageIndex++;
                renderPage(currentPageIndex);
            }}
        }}
        
        function prevPage() {{
            if (currentPageIndex > 0) {{
                currentPageIndex--;
                renderPage(currentPageIndex);
            }}
        }}
        
        function firstPage() {{
            currentPageIndex = 0;
            renderPage(currentPageIndex);
        }}
        
        function lastPage() {{
            currentPageIndex = totalPages - 1;
            renderPage(currentPageIndex);
        }}
        
        function goToPage(index) {{
            currentPageIndex = parseInt(index);
            renderPage(currentPageIndex);
        }}
        
        // Initial render
        renderPage(0);
    </script>
</body>
</html>
'''
    
    output_path.write_text(html_template, encoding="utf-8")

