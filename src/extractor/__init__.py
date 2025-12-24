"""LLM-based Knowledge Graph extraction package.

This package provides the main entry point for extracting structured
Knowledge Graph data from raw financial text using an LLM.
The actual LLM provider is determined by configuration.

When chunking is enabled and text exceeds the configured chunk size,
the text is split into overlapping chunks, each chunk is processed
independently, and the resulting graphs are merged.
"""

from src.chunker import estimate_tokens, split_text
from src.config import load_config
from src.extractor.factory import create_extractor
from src.graph_merger import merge_graphs
from src.schema import KnowledgeGraph


def extract_knowledge_graph(text: str) -> KnowledgeGraph:
    """Extract a Knowledge Graph from raw financial text.

    Creates an LLM extractor based on application configuration and
    uses it to extract entities and relationships from the input text.
    The LLM provider (OpenAI, Ollama, etc.) is determined by the
    LLM_PROVIDER environment variable.

    When chunking is enabled (CHUNK_ENABLED=true) and the text exceeds
    the configured chunk size (CHUNK_SIZE_TOKENS), the text is split
    into overlapping chunks. Each chunk is processed independently by
    the LLM, and the resulting graphs are merged into a single graph.

    If any chunk extraction fails, the entire operation fails immediately
    (fail-fast behavior).

    Args:
        text: Raw financial text to extract entities and relationships from.

    Returns:
        A validated KnowledgeGraph instance containing the extracted
        nodes and relationships.

    Raises:
        ValueError: If the LLM provider is not configured correctly,
            API key is missing, or the LLM response is invalid.
        json.JSONDecodeError: If the LLM response is not valid JSON.
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

        # Extract from each chunk (fail-fast: any error propagates immediately)
        graphs: list[KnowledgeGraph] = []
        for chunk in chunks:
            graph = extractor.extract(chunk)
            graphs.append(graph)

        # Merge all graphs into one
        return merge_graphs(graphs)

    # Single extraction (chunking disabled or text fits in one chunk)
    return extractor.extract(text)
