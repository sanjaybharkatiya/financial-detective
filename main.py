"""Main orchestration module for the Financial Detective pipeline.

This module coordinates the end-to-end workflow:
1. Load raw financial text
2. Extract Knowledge Graph via LLM
3. Validate graph integrity
4. Save graph to JSON
5. Render graph visualization
"""

import sys
from pathlib import Path

from src.extractor import extract_knowledge_graph
from src.input_loader import load_raw_text
from src.schema import KnowledgeGraph
from src.validator import validate_knowledge_graph

# Try to import visualizer - may fail on Python 3.14 due to networkx compatibility
try:
    from src.visualizer import render_graph
    VISUALIZER_AVAILABLE = True
except (ImportError, AttributeError):
    VISUALIZER_AVAILABLE = False
    render_graph = None  # type: ignore

# Mermaid visualizer is always available (no external dependencies)
from src.visualizer_mermaid import render_mermaid

# Output path for the extracted Knowledge Graph JSON
OUTPUT_JSON_PATH: Path = Path("data/graph_output.json")


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


def main() -> int:
    """Run the Financial Detective pipeline.

    Orchestrates the full workflow: load text, extract graph,
    validate, save JSON, and render visualization.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    try:
        print("[1/5] Loading raw financial text...")
        raw_text = load_raw_text()
        print(f"      Loaded {len(raw_text)} characters")

        print("[2/5] Extracting Knowledge Graph via LLM...")
        graph = extract_knowledge_graph(raw_text)
        print(f"      Extracted {len(graph.nodes)} nodes and {len(graph.relationships)} relationships")

        print("[3/5] Validating Knowledge Graph...")
        validate_knowledge_graph(graph)
        print("      Validation passed")

        print(f"[4/5] Saving graph to {OUTPUT_JSON_PATH}...")
        save_graph_json(graph, OUTPUT_JSON_PATH)
        print("      Graph saved successfully")

        print("[5/5] Rendering graph visualizations...")
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

        render_mermaid(graph)
        print("      Mermaid saved to visuals/graph.mmd")

        print("\n✓ Pipeline completed successfully")
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
