import os
from pathlib import Path
from typing import List

def get_file_extension(filename: str) -> str:
    """Extract file extension from filename"""
    return Path(filename).suffix.lower()

def validate_file_size(content: bytes, max_size: int) -> bool:
    """Check if file size is within limits"""
    return len(content) <= max_size

def safe_filename(filename: str) -> str:
    """Generate a safe filename by removing unsafe characters"""
    import re
    # Remove unsafe characters
    filename = re.sub(r'[^\w\s.-]', '', filename)
    # Replace spaces with underscores
    filename = re.sub(r'\s+', '_', filename)
    return filename

def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if it doesn't"""
    Path(path).mkdir(parents=True, exist_ok=True)
