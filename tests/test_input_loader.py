"""Unit tests for the input_loader module.

Tests raw text file loading:
- Successful file loading
- UTF-8 encoding
- File not found handling
- Empty file handling
"""

from pathlib import Path

import pytest

from src.input_loader import load_raw_text


class TestLoadRawText:
    """Tests for load_raw_text function."""

    def test_loads_file_content(self, tmp_path: Path) -> None:
        """load_raw_text should load file content as string."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!", encoding="utf-8")

        result = load_raw_text(test_file)

        assert result == "Hello, World!"

    def test_preserves_line_breaks(self, tmp_path: Path) -> None:
        """load_raw_text should preserve line breaks."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3", encoding="utf-8")

        result = load_raw_text(test_file)

        assert result == "Line 1\nLine 2\nLine 3"
        assert result.count("\n") == 2

    def test_preserves_formatting(self, tmp_path: Path) -> None:
        """load_raw_text should preserve all formatting."""
        test_file = tmp_path / "test.txt"
        content = "  Indented\n\t\tTabbed\n\nDouble newline"
        test_file.write_text(content, encoding="utf-8")

        result = load_raw_text(test_file)

        assert result == content

    def test_handles_utf8_characters(self, tmp_path: Path) -> None:
        """load_raw_text should handle UTF-8 encoded characters."""
        test_file = tmp_path / "test.txt"
        content = "Currency: ₹1,00,000 and €100 and £50 and $100"
        test_file.write_text(content, encoding="utf-8")

        result = load_raw_text(test_file)

        assert "₹" in result
        assert "€" in result
        assert "£" in result

    def test_handles_unicode_emoji(self, tmp_path: Path) -> None:
        """load_raw_text should handle Unicode emoji."""
        test_file = tmp_path / "test.txt"
        content = "Status: ✓ Complete 🚀"
        test_file.write_text(content, encoding="utf-8")

        result = load_raw_text(test_file)

        assert "✓" in result
        assert "🚀" in result

    def test_file_not_found_raises_error(self) -> None:
        """load_raw_text should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_raw_text(Path("/nonexistent/path/file.txt"))

    def test_empty_file_returns_empty_string(self, tmp_path: Path) -> None:
        """load_raw_text should return empty string for empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("", encoding="utf-8")

        result = load_raw_text(test_file)

        assert result == ""

    def test_large_file(self, tmp_path: Path) -> None:
        """load_raw_text should handle large files."""
        test_file = tmp_path / "large.txt"
        large_content = "This is a line.\n" * 10000
        test_file.write_text(large_content, encoding="utf-8")

        result = load_raw_text(test_file)

        assert len(result) == len(large_content)
        assert result.count("\n") == 10000

    def test_financial_report_content(self, tmp_path: Path) -> None:
        """load_raw_text should handle typical financial report content."""
        test_file = tmp_path / "report.txt"
        content = """ANNUAL REPORT 2024

Company: Acme Corporation
Revenue: $1,234,567,890
Net Income: $123,456,789

Risk Factors:
- Market volatility
- Regulatory changes
- Currency fluctuations

Management Discussion:
The company reported strong growth...
"""
        test_file.write_text(content, encoding="utf-8")

        result = load_raw_text(test_file)

        assert "Acme Corporation" in result
        assert "$1,234,567,890" in result
        assert "Risk Factors" in result

    def test_path_object_accepted(self, tmp_path: Path) -> None:
        """load_raw_text should accept Path objects."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Content", encoding="utf-8")

        # Should not raise - Path object is valid
        result = load_raw_text(test_file)

        assert result == "Content"

    def test_nested_directory_path(self, tmp_path: Path) -> None:
        """load_raw_text should work with nested directory paths."""
        nested_dir = tmp_path / "deep" / "nested" / "dir"
        nested_dir.mkdir(parents=True)
        test_file = nested_dir / "test.txt"
        test_file.write_text("Nested content", encoding="utf-8")

        result = load_raw_text(test_file)

        assert result == "Nested content"

    def test_special_chars_in_content(self, tmp_path: Path) -> None:
        """load_raw_text should handle special characters."""
        test_file = tmp_path / "test.txt"
        content = "Special chars: <>&\"'[]{}|\\/"
        test_file.write_text(content, encoding="utf-8")

        result = load_raw_text(test_file)

        assert result == content

    def test_windows_line_endings(self, tmp_path: Path) -> None:
        """load_raw_text should handle Windows-style line endings."""
        test_file = tmp_path / "test.txt"
        # Write with Windows line endings
        test_file.write_bytes(b"Line 1\r\nLine 2\r\nLine 3")

        result = load_raw_text(test_file)

        # Python's read should handle \r\n
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

