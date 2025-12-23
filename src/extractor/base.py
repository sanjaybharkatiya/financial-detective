"""Abstract base class for LLM extractors.

This module defines the contract that all LLM extractor implementations
must follow. Each extractor is responsible for transforming raw text
into a validated KnowledgeGraph using an LLM provider.
"""

from abc import ABC, abstractmethod

from src.schema import KnowledgeGraph


class LLMExtractor(ABC):
    """Abstract base class for LLM-based Knowledge Graph extractors.

    All LLM extractor implementations must inherit from this class and
    implement the extract method. This ensures a consistent interface
    across different LLM providers (OpenAI, Ollama, etc.).

    The extractor is responsible for:
    1. Sending the input text to an LLM with appropriate prompts
    2. Parsing the LLM response as JSON
    3. Validating the JSON against the KnowledgeGraph schema
    4. Returning a validated KnowledgeGraph instance

    Implementations should NOT perform any post-processing, cleanup,
    or regex-based extraction. All extraction logic must be delegated
    to the LLM.
    """

    @abstractmethod
    def extract(self, text: str) -> KnowledgeGraph:
        """Extract a Knowledge Graph from raw text using an LLM.

        This method must be implemented by all concrete extractor classes.
        The implementation should send the text to an LLM, parse the
        JSON response, and return a validated KnowledgeGraph.

        Args:
            text: Raw text to extract entities and relationships from.
                Should be unprocessed financial document content.

        Returns:
            A validated KnowledgeGraph instance containing the extracted
            nodes and relationships.

        Raises:
            ValueError: If the LLM response cannot be parsed or validated.
            ConnectionError: If the LLM provider is unreachable.
        """
        pass

