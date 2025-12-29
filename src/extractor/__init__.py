"""LLM-based Knowledge Graph extraction package.

This package provides the main entry point for extracting structured
Knowledge Graph data from raw financial text using an LLM.
The actual LLM provider is determined by configuration.

When chunking is enabled and text exceeds the configured chunk size,
the text is split into overlapping chunks, each chunk is processed
independently, and the resulting graphs are merged.

Features:
- Graceful failure handling: failed chunks are skipped, not fatal
- Detailed logging: shows exactly what succeeded and failed
- Progress callbacks: save intermediate results while processing
"""

from typing import Callable

from src.chunker import estimate_tokens, split_text
from src.config import load_config
from src.extractor.factory import create_extractor
from src.graph_merger import merge_graphs
from src.schema import KnowledgeGraph


# Type alias for progress callback
ProgressCallback = Callable[[KnowledgeGraph, int, int], None]


def extract_knowledge_graph(
    text: str,
    on_chunk_complete: ProgressCallback | None = None,
) -> KnowledgeGraph:
    """Extract a Knowledge Graph from raw financial text.

    Creates an LLM extractor based on application configuration and
    uses it to extract entities and relationships from the input text.
    The LLM provider (OpenAI, Ollama, etc.) is determined by the
    LLM_PROVIDER environment variable.

    When chunking is enabled (CHUNK_ENABLED=true) and the text exceeds
    the configured chunk size (CHUNK_SIZE_TOKENS), the text is split
    into overlapping chunks. Each chunk is processed independently by
    the LLM, and the resulting graphs are merged into a single graph.

    Failed chunks are logged and skipped - extraction continues with
    remaining chunks. Only fails if ALL chunks fail.

    Args:
        text: Raw financial text to extract entities and relationships from.
        on_chunk_complete: Optional callback invoked after each chunk is processed.
            Receives the merged graph so far, current chunk index, and total chunks.
            Use this to save intermediate results or update UI.

    Returns:
        A validated KnowledgeGraph instance containing the extracted
        nodes and relationships.

    Raises:
        ValueError: If the LLM provider is not configured correctly,
            API key is missing, or ALL chunks fail to extract.
    """
    config = load_config()
    extractor = create_extractor()

    # Check if chunking is enabled and text exceeds chunk size
    if config.chunk_enabled and estimate_tokens(text) > config.chunk_size_tokens:
        # Validate chunk configuration
        if config.chunk_overlap_tokens >= config.chunk_size_tokens:
            raise ValueError(
                "CHUNK_OVERLAP_TOKENS must be less than CHUNK_SIZE_TOKENS"
            )

        # Split text into chunks
        chunks = split_text(
            text=text,
            chunk_size=config.chunk_size_tokens,
            overlap=config.chunk_overlap_tokens,
        )

        total_chunks = len(chunks)
        print(f"      Splitting into {total_chunks} chunks (size={config.chunk_size_tokens} tokens, overlap={config.chunk_overlap_tokens} tokens)")

        # Track success/failure stats
        graphs: list[KnowledgeGraph] = []
        failed_chunks: list[int] = []
        total_nodes = 0
        total_relationships = 0

        # Extract from each chunk, merging incrementally
        for i, chunk in enumerate(chunks, start=1):
            print(f"      [{i}/{total_chunks}] Processing...", end=" ", flush=True)
            try:
                graph = extractor.extract(chunk)
                nodes_count = len(graph.nodes)
                rels_count = len(graph.relationships)
                total_nodes += nodes_count
                total_relationships += rels_count
                print(f"✓ {nodes_count} nodes, {rels_count} rels")
                graphs.append(graph)

                # Merge and invoke callback with current progress
                if on_chunk_complete is not None and graphs:
                    merged_so_far = merge_graphs(graphs)
                    on_chunk_complete(merged_so_far, i, total_chunks)

            except Exception as e:
                failed_chunks.append(i)
                error_msg = str(e)[:100]
                print(f"✗ FAILED: {error_msg}")
                # Continue processing other chunks

        # Summary
        success_count = len(graphs)
        fail_count = len(failed_chunks)
        print(f"      ─────────────────────────────────────")
        print(f"      Summary: {success_count}/{total_chunks} chunks succeeded")
        if failed_chunks:
            print(f"      Failed chunks: {failed_chunks}")
        print(f"      Total extracted: {total_nodes} nodes, {total_relationships} relationships")

        if not graphs:
            raise ValueError(
                f"All {total_chunks} chunks failed to extract. "
                "Check LLM connection and prompt compatibility."
            )

        # Merge all graphs into one
        return merge_graphs(graphs)

    # Single extraction (chunking disabled or text fits in one chunk)
    return extractor.extract(text)
