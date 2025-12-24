"""Text chunking module for large document processing.

This module provides functionality to split large documents into smaller chunks
that fit within LLM context windows, while preserving semantic boundaries and
maintaining overlap for context continuity.
"""

import re


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in text.

    Uses a simple approximation: 1 token ≈ 4 characters.
    This is suitable for quick estimation without external tokenizer dependencies.

    Args:
        text: The text to estimate token count for.

    Returns:
        Estimated number of tokens.

    Example:
        >>> estimate_tokens("Hello, world!")
        3
    """
    return len(text) // 4


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks at semantic boundaries.

    Attempts to split on paragraph boundaries first, then sentence boundaries
    if paragraphs are too large. Preserves overlap between chunks to maintain
    context continuity across boundaries.

    Args:
        text: The text to split into chunks.
        chunk_size: Target size of each chunk in tokens.
        overlap: Number of tokens to overlap between consecutive chunks.

    Returns:
        List of text chunks, each approximately chunk_size tokens.
        Returns empty list if text is empty.
        Returns single-item list with original text if it fits in one chunk.

    Raises:
        ValueError: If chunk_size <= 0 or overlap < 0 or overlap >= chunk_size.

    Example:
        >>> text = "Paragraph 1.\\n\\nParagraph 2.\\n\\nParagraph 3."
        >>> chunks = split_text(text, chunk_size=100, overlap=10)
        >>> len(chunks) >= 1
        True
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    if not text.strip():
        return []

    estimated_tokens = estimate_tokens(text)
    if estimated_tokens <= chunk_size:
        return [text]

    # Convert token counts to character counts for splitting
    chunk_chars = chunk_size * 4
    overlap_chars = overlap * 4

    chunks: list[str] = []
    current_pos = 0

    while current_pos < len(text):
        # Calculate end position for this chunk
        end_pos = min(current_pos + chunk_chars, len(text))

        # Extract chunk
        chunk = text[current_pos:end_pos]

        # If not the last chunk, try to split at a natural boundary
        if end_pos < len(text):
            chunk = _split_at_boundary(chunk)

        chunks.append(chunk.strip())

        # Move position forward, accounting for overlap
        # Overlap is applied by moving back from the end of current chunk
        if end_pos < len(text):
            current_pos = end_pos - overlap_chars
            # Ensure we make progress (avoid infinite loop)
            if current_pos <= chunks[-1].find(chunk[:100]):
                current_pos = end_pos
        else:
            break

    return chunks


def _split_at_boundary(text: str) -> str:
    """Split text at the last natural boundary (paragraph or sentence).

    Attempts to find the last paragraph boundary, then sentence boundary,
    then word boundary. Falls back to the full text if no boundary found.

    Args:
        text: Text to find boundary in.

    Returns:
        Text truncated at the last natural boundary.
    """
    # Try to split at last paragraph boundary (double newline)
    paragraph_matches = list(re.finditer(r'\n\n+', text))
    if paragraph_matches:
        last_para = paragraph_matches[-1]
        return text[:last_para.end()].rstrip()

    # Try to split at last sentence boundary
    sentence_matches = list(re.finditer(r'[.!?]\s+', text))
    if sentence_matches:
        # Use second-to-last sentence to leave some context
        if len(sentence_matches) >= 2:
            last_sentence = sentence_matches[-2]
        else:
            last_sentence = sentence_matches[-1]
        return text[:last_sentence.end()].rstrip()

    # Try to split at last word boundary
    word_matches = list(re.finditer(r'\s+', text))
    if word_matches and len(word_matches) > 10:
        # Use a word boundary in the last 20% of text
        target_match = word_matches[int(len(word_matches) * 0.8)]
        return text[:target_match.end()].rstrip()

    # No good boundary found, return full text
    return text
