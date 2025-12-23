"""Raw input loading module.

This module handles loading raw financial text from disk.
No parsing, cleaning, or normalization is performed.
"""

from pathlib import Path

# Default path to the raw financial report
RAW_REPORT_PATH: Path = Path("data/raw_report.txt")


def load_raw_text(file_path: Path = RAW_REPORT_PATH) -> str:
    """Load raw financial text from a file.

    Reads the entire file content as a single string, preserving
    all formatting and line breaks. No cleaning or parsing is applied.

    Args:
        file_path: Path to the raw text file. Defaults to data/raw_report.txt.

    Returns:
        The full text content of the file as a single string.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        PermissionError: If the file cannot be read due to permissions.
    """
    return file_path.read_text(encoding="utf-8")

