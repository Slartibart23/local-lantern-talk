"""Streaming sentence splitter.

Speech text arrives token by token. :class:`SentenceSplitter` emits complete
sentences as early as possible so TTS can start while the LLM keeps generating.
"""

from __future__ import annotations

import re

# Abbreviations (lower case, without the final period) that do not end a sentence.
# Covers the four built-in voice languages: English, German, Spanish, French.
ABBREVIATIONS: frozenset[str] = frozenset(
    {
        # English
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
        "approx", "no", "fig", "inc", "ltd", "co", "mt", "jan", "feb", "mar", "apr",
        "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
        # German
        "z.b", "u.a", "d.h", "bzw", "ca", "evtl", "ggf", "inkl", "nr", "str", "usw",
        "vgl", "hr", "fr", "bspw", "s", "u.s.w", "z.t", "o.ä",
        # Spanish
        "sra", "srta", "dra", "ud", "uds", "pág",
        # French
        "mme", "mlle", "m", "env", "p.ex",
    }
)

_TERMINATORS = ".!?…。！？"
_CLOSERS = "\"'”’»)]"
# Minimum characters for a sentence to be sent on its own; shorter pieces
# ("Oh." "Yes!") are merged with the following sentence for smoother speech.
MIN_SENTENCE_CHARS = 12
# Force a split when no sentence end appears for this long.
MAX_SENTENCE_CHARS = 220


def _ends_sentence(buffer: str, idx: int) -> bool:
    """Decide whether the terminator at ``buffer[idx]`` ends a sentence."""
    ch = buffer[idx]
    if ch != ".":
        return True
    # Ellipsis written as "..." — treat as a sentence end only at the last dot.
    if idx + 1 < len(buffer) and buffer[idx + 1] == ".":
        return False
    m = re.search(r"([\w.]+)$", buffer[:idx])
    word = m.group(1).lower() if m else ""
    if word in ABBREVIATIONS:
        return False
    # Single letters ("A. Smith") and ordinals ("am 3. Mai") are not sentence ends.
    if len(word) == 1 and word.isalpha():
        return False
    if word.isdigit() and len(word) <= 2:
        return False
    return True


class SentenceSplitter:
    """Collects streamed text and returns finished sentences."""

    def __init__(self, min_chars: int = MIN_SENTENCE_CHARS, max_chars: int = MAX_SENTENCE_CHARS) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buffer = ""
        self._carry = ""  # short sentence waiting to be merged with the next one

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        out: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            sentence, self._buffer = self._buffer[:cut].strip(), self._buffer[cut:].lstrip()
            self._push(sentence, out)
        return out

    def finish(self) -> list[str]:
        out: list[str] = []
        rest = self._buffer.strip()
        self._buffer = ""
        combined = f"{self._carry} {rest}".strip()
        self._carry = ""
        if combined:
            out.append(combined)
        return out

    def _push(self, sentence: str, out: list[str]) -> None:
        if not sentence:
            return
        combined = f"{self._carry} {sentence}".strip()
        if len(combined) < self.min_chars:
            self._carry = combined
        else:
            self._carry = ""
            out.append(combined)

    def _find_cut(self) -> int | None:
        buf = self._buffer
        for idx, ch in enumerate(buf):
            if ch == "\n":
                return idx + 1
            if ch in _TERMINATORS:
                end = idx + 1
                while end < len(buf) and (buf[end] in _TERMINATORS or buf[end] in _CLOSERS):
                    end += 1
                if end >= len(buf):
                    return None  # need the next character to be sure
                if buf[end].isspace() and _ends_sentence(buf, idx):
                    return end
        if len(buf) > self.max_chars:
            return self._soft_cut(buf)
        return None

    def _soft_cut(self, buf: str) -> int:
        """Split an overlong sentence at the last comma-like pause, else at a space."""
        window = buf[: self.max_chars]
        for pattern in (r"[,;:–—]\s", r"\s"):
            matches = list(re.finditer(pattern, window))
            if matches and matches[-1].end() > self.max_chars // 3:
                return matches[-1].end()
        return self.max_chars
