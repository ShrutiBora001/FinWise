
"""
Preprocessing helpers for text and document records.
"""
import re
import hashlib


def clean_text(text: str) -> str:
    """Basic text cleaning: strip whitespace, remove HTML tags, normalize spaces."""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def deduplicate_records(records):
    """Remove duplicates based on hash of text content."""
    seen = set()
    unique = []
    for r in records:
        h = hashlib.md5(r["text"].encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(r)
    return unique

