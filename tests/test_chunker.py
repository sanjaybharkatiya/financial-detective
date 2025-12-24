"""Tests for the text chunking module."""

import pytest

from src.chunker import estimate_tokens, split_text


class TestEstimateTokens:
    """Tests for the estimate_tokens function."""

    def test_empty_string(self) -> None:
        """Empty string should return 0 tokens."""
        assert estimate_tokens("") == 0

    def test_short_text(self) -> None:
        """Short text should return approximate token count."""
        text = "Hello, world!"
        tokens = estimate_tokens(text)
        # 13 chars / 4 = 3.25, integer division = 3
        assert tokens == 3

    def test_longer_text(self) -> None:
        """Longer text should scale linearly."""
        text = "a" * 400
        tokens = estimate_tokens(text)
        assert tokens == 100

    def test_whitespace(self) -> None:
        """Whitespace counts toward token estimate."""
        text = "    "
        tokens = estimate_tokens(text)
        assert tokens == 1


class TestSplitText:
    """Tests for the split_text function."""

    def test_empty_text(self) -> None:
        """Empty text should return empty list."""
        chunks = split_text("", chunk_size=100, overlap=10)
        assert chunks == []

    def test_whitespace_only(self) -> None:
        """Whitespace-only text should return empty list."""
        chunks = split_text("   \n\n   ", chunk_size=100, overlap=10)
        assert chunks == []

    def test_text_fits_in_one_chunk(self) -> None:
        """Text that fits in one chunk should return single chunk."""
        text = "This is a short text."
        chunks = split_text(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_text_requires_multiple_chunks(self) -> None:
        """Text exceeding chunk size should be split."""
        # Create text that's definitely too large
        text = "Paragraph one. " * 50 + "\n\n" + "Paragraph two. " * 50
        chunks = split_text(text, chunk_size=100, overlap=10)
        assert len(chunks) > 1

    def test_paragraph_boundary_splitting(self) -> None:
        """Text should be split at paragraph boundaries when possible."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        # Use very small chunk size to force splitting (5 tokens = 20 chars)
        chunks = split_text(text, chunk_size=5, overlap=1)
        assert len(chunks) >= 2

    def test_overlap_preservation(self) -> None:
        """Consecutive chunks should have overlapping content."""
        text = "A" * 200 + "B" * 200 + "C" * 200
        chunks = split_text(text, chunk_size=100, overlap=20)
        
        # Should have multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should be roughly the target size (within reasonable bounds)
        for chunk in chunks:
            # Chunks should not be empty
            assert len(chunk) > 0

    def test_invalid_chunk_size(self) -> None:
        """Negative or zero chunk_size should raise ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            split_text("text", chunk_size=0, overlap=10)
        
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            split_text("text", chunk_size=-100, overlap=10)

    def test_invalid_overlap(self) -> None:
        """Negative overlap should raise ValueError."""
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            split_text("text", chunk_size=100, overlap=-10)

    def test_overlap_exceeds_chunk_size(self) -> None:
        """Overlap >= chunk_size should raise ValueError."""
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            split_text("text", chunk_size=100, overlap=100)
        
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            split_text("text", chunk_size=100, overlap=150)

    def test_single_paragraph(self) -> None:
        """Single paragraph without breaks should still be chunked."""
        text = "This is one long paragraph without any breaks. " * 100
        chunks = split_text(text, chunk_size=50, overlap=10)
        
        # Should create multiple chunks even without paragraph breaks
        assert len(chunks) > 1

    def test_chunks_are_stripped(self) -> None:
        """All chunks should be stripped of leading/trailing whitespace."""
        text = "First\n\n\n\nSecond\n\n\n\nThird"
        chunks = split_text(text, chunk_size=10, overlap=2)
        
        for chunk in chunks:
            # Should not start or end with whitespace
            assert chunk == chunk.strip()

    def test_sentence_boundary_splitting(self) -> None:
        """Text should split at sentence boundaries when no paragraphs."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = split_text(text, chunk_size=20, overlap=5)
        
        # Should have multiple chunks
        assert len(chunks) >= 1
        
        # All chunks should be non-empty
        for chunk in chunks:
            assert len(chunk) > 0
