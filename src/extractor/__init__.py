"""LLM-based Knowledge Graph extraction package.

This package provides the main entry point for extracting structured
Knowledge Graph data from raw financial text using an LLM.
The actual LLM provider is determined by configuration.
"""

from src.extractor.factory import create_extractor
from src.schema import KnowledgeGraph


def extract_knowledge_graph(text: str) -> KnowledgeGraph:
    """Extract a Knowledge Graph from raw financial text.

    Creates an LLM extractor based on application configuration and
    uses it to extract entities and relationships from the input text.
    The LLM provider (OpenAI, Ollama, etc.) is determined by the
    LLM_PROVIDER environment variable.

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
    extractor = create_extractor()
    return extractor.extract(text)
