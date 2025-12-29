"""Main orchestration module for the Financial Detective pipeline.

This module coordinates the end-to-end workflow:
1. Load raw financial text
2. Extract Knowledge Graph via LLM
3. Validate graph integrity
4. Save graph to JSON
5. Render graph visualization

Features iterative extraction with live progress updates:
- Saves intermediate JSON after each chunk
- Updates Mermaid/HTML visualizations in real-time
- Allows viewing results while processing continues
"""

import sys
import webbrowser
from pathlib import Path

from src.config import load_config
from src.extractor import extract_knowledge_graph
from src.input_loader import load_raw_text
from src.schema import KnowledgeGraph
from src.validator import validate_and_repair_graph

# Try to import visualizer - may fail on Python 3.14 due to networkx compatibility
try:
    from src.visualizer import render_graph
    VISUALIZER_AVAILABLE = True
except (ImportError, AttributeError):
    VISUALIZER_AVAILABLE = False
    render_graph = None  # type: ignore

# Mermaid visualizer is always available (no external dependencies)
from src.visualizer_mermaid import render_mermaid, render_mermaid_html

# Output paths
OUTPUT_JSON_PATH: Path = Path("data/graph_output.json")
MERMAID_PATH: Path = Path("visuals/graph.mmd")
HTML_PATH: Path = Path("visuals/graph.html")


def save_graph_json(graph: KnowledgeGraph, output_path: Path) -> None:
    """Save a KnowledgeGraph to a JSON file.

    Args:
        graph: The KnowledgeGraph instance to save.
        output_path: Path where the JSON file will be written.

    Raises:
        OSError: If the file cannot be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        graph.model_dump_json(indent=2),
        encoding="utf-8",
    )


def save_intermediate_results(graph: KnowledgeGraph, chunk_idx: int, total_chunks: int) -> None:
    """Save intermediate extraction results after each chunk.

    This callback is invoked after each chunk is processed, allowing
    users to view results while extraction continues. Updates:
    - JSON output file
    - Mermaid diagram
    - HTML visualization

    Args:
        graph: The merged KnowledgeGraph so far.
        chunk_idx: Current chunk index (1-based).
        total_chunks: Total number of chunks.
    """
    # Save JSON
    save_graph_json(graph, OUTPUT_JSON_PATH)

    # Save Mermaid
    MERMAID_PATH.parent.mkdir(parents=True, exist_ok=True)
    render_mermaid(graph, MERMAID_PATH)

    # Save HTML
    render_mermaid_html(graph, HTML_PATH)

    # Print progress summary
    print(f"      📁 Saved: {len(graph.nodes)} nodes, {len(graph.relationships)} relationships (chunk {chunk_idx}/{total_chunks})")


def main() -> int:
    """Run the Financial Detective pipeline.

    Orchestrates the full workflow: load text, extract graph,
    validate, save JSON, and render visualization.

    Features iterative extraction with live progress:
    - After each chunk, saves JSON + Mermaid + HTML
    - Refresh browser to see graph building up
    - Continues processing even if some chunks fail

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    try:
        print("[1/5] Loading raw financial text...")
        raw_text = load_raw_text()
        print(f"      Loaded {len(raw_text)} characters")

        print("[2/5] Extracting Knowledge Graph via LLM...")
        config = load_config()
        if config.llm_provider == "openai":
            print(f"      🤖 Provider: OpenAI | Model: {config.openai_model}")
        elif config.llm_provider == "gemini":
            print(f"      🤖 Provider: Google Gemini | Model: {config.gemini_model}")
        elif config.llm_provider == "ollama":
            print(f"      🤖 Provider: Ollama (local) | Model: {config.ollama_model}")
        print("      ℹ️  Results are saved after each chunk - refresh browser to view progress")

        # Open browser early so user can watch progress
        HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not HTML_PATH.exists():
            # Create empty placeholder
            HTML_PATH.write_text("<html><body><h1>Extraction in progress...</h1></body></html>")

        html_absolute_path = HTML_PATH.resolve().as_uri()
        webbrowser.open(html_absolute_path)
        print(f"      📊 Browser opened - refresh to see live updates")

        # Extract with progress callback
        graph = extract_knowledge_graph(raw_text, on_chunk_complete=save_intermediate_results)
        print(f"      ✓ Extraction complete: {len(graph.nodes)} nodes, {len(graph.relationships)} relationships")

        print("[3/5] Validating and repairing Knowledge Graph...")
        graph = validate_and_repair_graph(graph)
        print(f"      Validation passed: {len(graph.nodes)} nodes, {len(graph.relationships)} relationships")

        print(f"[4/5] Saving final graph to {OUTPUT_JSON_PATH}...")
        save_graph_json(graph, OUTPUT_JSON_PATH)
        print("      Graph saved successfully")

        print("[5/5] Rendering final visualizations...")
        if VISUALIZER_AVAILABLE:
            render_graph(graph)
            print("      PNG saved to visuals/graph.png")
        else:
            # Provide a friendly, informative message about NetworkX compatibility
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            print(f"      PNG skipped: NetworkX is not yet compatible with Python {python_version}")
            if sys.version_info >= (3, 14):
                print("      ℹ️  To enable PNG graph rendering, use Python 3.11–3.13")
                print("         Mermaid diagrams remain fully available as an alternative")

        render_mermaid(graph, MERMAID_PATH)
        print(f"      Mermaid saved to {MERMAID_PATH}")

        render_mermaid_html(graph, HTML_PATH)
        print(f"      HTML saved to {HTML_PATH}")

        print("\n✓ Pipeline completed successfully")
        print("      📊 Refresh browser to see final visualization")

        return 0

    except FileNotFoundError as e:
        print(f"\n✗ File not found: {e}", file=sys.stderr)
        return 1

    except ValueError as e:
        print(f"\n✗ Validation error: {e}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
