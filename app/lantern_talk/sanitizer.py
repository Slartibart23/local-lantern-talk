"""Speech sanitizer — guarantees that actions are never spoken (spec §8).

LLM output is split into *segments*:

* ``speech`` — text that may be sent to TTS (after :func:`clean_for_speech`),
* ``action`` — roleplay actions such as ``*smiles*``, ``_nods_`` or ``[sighs]``,
* ``ooc``    — out-of-character text such as ``((brb))`` or ``[OOC: ...]``,
* ``code``   — fenced code blocks.

Only ``speech`` segments are ever spoken. :class:`ActionFilter` works on a token
stream: when a marker opens, the text is held back until the marker closes or
the message ends. A marker that is still open when the message ends is reported
as an action, so a half-open action is never spoken.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

SegmentKind = Literal["speech", "action", "ooc", "code"]


@dataclass(frozen=True)
class Segment:
    """A piece of LLM output. ``text`` has its markers removed."""

    kind: SegmentKind
    text: str

    @property
    def spoken(self) -> bool:
        return self.kind == "speech"


class _Pending(Exception):
    """More input is needed to decide what the construct at the current position means."""


_LINE_PREFIX = re.compile(r"[ \t]*(?:#{1,6}[ \t]+|[-*+•][ \t]+|\d{1,3}[.)][ \t]+|>[ \t]?)")
_OOC_LINE = re.compile(r"[ \t]*OOC[ \t]*:", re.IGNORECASE)
_OOC_PREFIX = re.compile(r"^\s*ooc\b\s*:?\s*", re.IGNORECASE)
_OOC_START = re.compile(r"\s*ooc\b", re.IGNORECASE)
_LINE_START_LOOKAHEAD = 6


def _is_word(ch: str) -> bool:
    return bool(ch) and (ch.isalnum() or ch == "_")


class _Parser:
    """A single pass over ``text``.

    Everything collected before a :class:`_Pending` is raised is final, because
    each handler raises before it emits anything. ``committed`` is the index up to
    which the input has been consumed.
    """

    def __init__(self, text: str, final: bool, prev_char: str) -> None:
        self.text = text
        self.final = final
        self.prev_char = prev_char
        self.segments: list[Segment] = []
        self._speech: list[str] = []

    # -- output --------------------------------------------------------------
    def _flush_speech(self) -> None:
        if self._speech:
            self.segments.append(Segment("speech", "".join(self._speech)))
            self._speech = []

    def _emit(self, kind: SegmentKind, text: str) -> None:
        self._flush_speech()
        text = text.strip()
        if text:
            self.segments.append(Segment(kind, text))

    # -- input helpers ---------------------------------------------------------
    def _char(self, i: int) -> str:
        if 0 <= i < len(self.text):
            return self.text[i]
        return self.prev_char if i == -1 else ""

    def _need(self, i: int) -> None:
        """Require ``text[i]`` to exist unless the stream has ended."""
        if i >= len(self.text) and not self.final:
            raise _Pending

    def _find(self, marker: str, start: int) -> int:
        """Index of ``marker`` at or after ``start``; -1 only when the stream has ended."""
        pos = self.text.find(marker, start)
        if pos < 0 and not self.final:
            raise _Pending
        return pos

    # -- main loop -------------------------------------------------------------
    def run(self) -> tuple[list[Segment], int]:
        i = 0
        line_checked = -1
        text = self.text
        while i < len(text):
            start = i
            try:
                if self._char(i - 1) == "\n" and line_checked != i:
                    line_checked = i
                    i = self._line_start(i)
                    continue
                ch = text[i]
                if ch == "*":
                    i = self._star(i)
                elif ch == "_":
                    i = self._underscore(i)
                elif ch == "(":
                    i = self._paren(i)
                elif ch == "[":
                    i = self._bracket(i)
                elif ch == "`":
                    i += 1  # inline code marker: dropped, content stays
                else:
                    self._speech.append(ch)
                    i += 1
            except _Pending:
                self._flush_speech()
                return self.segments, start
        self._flush_speech()
        return self.segments, len(text)

    # -- handlers ----------------------------------------------------------------
    def _line_start(self, i: int) -> int:
        text = self.text
        rest = text[i: i + _LINE_START_LOOKAHEAD]
        if not self.final and len(rest) < _LINE_START_LOOKAHEAD and "\n" not in rest:
            raise _Pending
        if text.startswith("```", i):
            end = self._find("```", i + 3)
            body = text[i + 3: end if end >= 0 else len(text)]
            body = body.split("\n", 1)[1] if "\n" in body else ""  # drop the language tag
            self._emit("code", body)
            if end < 0:
                return len(text)
            end += 3
            return end + 1 if self._char(end) == "\n" else end
        m = _OOC_LINE.match(text, i)
        if m:
            end = self._find("\n", m.end())
            self._emit("ooc", text[m.end(): end if end >= 0 else len(text)])
            return len(text) if end < 0 else end + 1
        m = _LINE_PREFIX.match(text, i)
        if m:
            return m.end()  # header / bullet / numbering / quote marker: drop it
        return i

    def _star(self, i: int) -> int:
        self._need(i + 1)
        nxt = self._char(i + 1)
        if nxt == "*":
            return i + 2  # **bold**: drop the marker, keep the words
        if nxt == "" or nxt.isspace():
            return i + 1  # a lone "*" (e.g. "2 * 3") is not a marker
        end = self._find("*", i + 1)
        if end < 0:
            self._emit("action", self.text[i + 1:])
            return len(self.text)
        self._emit("action", self.text[i + 1: end])
        return end + 1

    def _underscore(self, i: int) -> int:
        text = self.text
        if _is_word(self._char(i - 1)):
            self._speech.append(" ")  # snake_case: read the underscore as a space
            return i + 1
        self._need(i + 1)
        nxt = self._char(i + 1)
        if nxt == "_":
            return i + 2  # __bold__
        if nxt == "" or nxt.isspace():
            return i + 1
        # The closing "_" must not be followed by a word character.
        j = i + 1
        while True:
            end = self._find("_", j)
            if end < 0:
                self._emit("action", text[i + 1:])
                return len(text)
            self._need(end + 1)
            if not _is_word(self._char(end + 1)):
                self._emit("action", text[i + 1: end])
                return end + 1
            j = end + 1

    def _paren(self, i: int) -> int:
        text = self.text
        self._need(i + 1)
        if self._char(i + 1) == "(":
            end = self._find("))", i + 2)
            self._emit("ooc", _OOC_PREFIX.sub("", text[i + 2: end if end >= 0 else len(text)]))
            return len(text) if end < 0 else end + 2
        head = text[i + 1: i + 5].lower()
        if not self.final and len(head) < 4 and "ooc ".startswith(head.lstrip() or "o"):
            raise _Pending
        if _OOC_START.match(text, i + 1):
            end = self._find(")", i + 1)
            body = text[i + 1: end if end >= 0 else len(text)]
            self._emit("ooc", _OOC_PREFIX.sub("", body))
            return len(text) if end < 0 else end + 1
        return i + 1  # ordinary parenthesis: drop it, keep the content

    def _bracket(self, i: int) -> int:
        text = self.text
        self._need(i + 1)
        if self._char(i + 1) == "[":
            end = self._find("]]", i + 2)
            self._emit("ooc", _OOC_PREFIX.sub("", text[i + 2: end if end >= 0 else len(text)]))
            return len(text) if end < 0 else end + 2
        end = self._find("]", i + 1)
        inner = text[i + 1: end if end >= 0 else len(text)]
        if end >= 0:
            self._need(end + 1)
            if self._char(end + 1) == "(":
                close = self._find(")", end + 2)
                self._speech.append(inner)  # markdown link: speak the link text only
                return len(text) if close < 0 else close + 1
        kind: SegmentKind = "ooc" if _OOC_START.match(inner) else "action"
        self._emit(kind, _OOC_PREFIX.sub("", inner) if kind == "ooc" else inner)
        return len(text) if end < 0 else end + 1


def split_segments(text: str) -> list[Segment]:
    """Split a complete message into segments (non-streaming)."""
    segments, _ = _Parser(text, final=True, prev_char="\n").run()
    return _merge(segments)


def _merge(segments: list[Segment]) -> list[Segment]:
    merged: list[Segment] = []
    for seg in segments:
        if merged and merged[-1].kind == seg.kind == "speech":
            merged[-1] = Segment("speech", merged[-1].text + seg.text)
        else:
            merged.append(seg)
    return merged


class ActionFilter:
    """Streaming version of :func:`split_segments`.

    ``feed`` returns the segments that are final so far; ``finish`` flushes the
    rest. Concatenating all returned segments equals ``split_segments`` on the
    full text (up to merging of adjacent speech pieces).
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._prev_char = "\n"

    def feed(self, chunk: str) -> list[Segment]:
        self._buffer += chunk
        return self._run(final=False)

    def finish(self) -> list[Segment]:
        return self._run(final=True)

    def _run(self, final: bool) -> list[Segment]:
        if not self._buffer:
            return []
        segments, consumed = _Parser(self._buffer, final, self._prev_char).run()
        if consumed:
            self._prev_char = self._buffer[consumed - 1]
            self._buffer = self._buffer[consumed:]
        return segments


# -- final cleanup before TTS ------------------------------------------------------

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_TAG = re.compile(r"</?[a-zA-Z][^<>]{0,40}>")
# Punctuation and symbols a TTS model can pronounce or use for prosody.
_ALLOWED_PUNCT = set(".,!?;:'\"-–—…%&€$£@/+=°«»„“”‘’¿¡")
_SENTENCE_PUNCT_RUN = re.compile(r"([.,!?;:…])(?:\s*[.,;:])+")


def clean_for_speech(text: str) -> str:
    """Remove everything from a ``speech`` segment that must not or cannot be spoken.

    Strips emojis, leftover markdown characters, URLs, HTML tags and any other
    symbol that is not pronounceable punctuation. Whitespace is normalized.
    """
    text = _URL.sub("", text)
    text = _HTML_TAG.sub("", text)
    out: list[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in "LN":
            out.append(ch)
        elif cat[0] == "M":
            # Combining marks belong to a letter; after an emoji they are
            # variation selectors / keycaps and are dropped with it.
            if out and unicodedata.category(out[-1])[0] in "LM":
                out.append(ch)
        elif ch.isspace():
            out.append(" ")
        elif ch in _ALLOWED_PUNCT:
            out.append(ch)
        # everything else (emoji, *, #, ~, |, <, >, {, }, \, ^ ...) is dropped
    cleaned = re.sub(r"\s+", " ", "".join(out)).strip()
    cleaned = re.sub(r"\s+([.,!?;:…])", r"\1", cleaned)
    cleaned = _SENTENCE_PUNCT_RUN.sub(r"\1", cleaned)
    if not any(unicodedata.category(c)[0] in "LN" for c in cleaned):
        return ""  # nothing pronounceable left
    return cleaned


def speech_stream(segments: list[Segment]) -> str:
    """Concatenate the spoken parts of ``segments``.

    Streamed speech segments are fragments of one text, so they are joined
    directly; a removed (non-spoken) segment leaves a single space behind.
    """
    return "".join(s.text if s.spoken else " " for s in segments)


def spoken_text(text: str) -> str:
    """Convenience: the exact text that would be spoken for a complete message."""
    return clean_for_speech(speech_stream(split_segments(text)))
