"""Fuzzy text comparison, used to verify generated speech with Whisper."""

from __future__ import annotations

import difflib
import re
import unicodedata


def normalize_words(text: str) -> list[str]:
    """Lower-case words without punctuation or accents."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"\w+", text)


def word_similarity(a: str, b: str) -> float:
    """Similarity of two texts on the word level, 0.0 … 1.0."""
    wa, wb = normalize_words(a), normalize_words(b)
    if not wa and not wb:
        return 1.0
    return difflib.SequenceMatcher(None, wa, wb).ratio()
