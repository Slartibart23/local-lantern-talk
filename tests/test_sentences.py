"""Tests for the streaming sentence splitter."""

from __future__ import annotations

import pytest

from lantern_talk.sentences import SentenceSplitter


def split_stream(text: str, chunk: int = 1, **kwargs: int) -> list[str]:
    s = SentenceSplitter(**kwargs)
    out: list[str] = []
    for i in range(0, len(text), chunk):
        out += s.feed(text[i: i + chunk])
    return out + s.finish()


@pytest.mark.parametrize("chunk", [1, 3, 1000])
def test_basic_split(chunk: int) -> None:
    text = "Hello there, my friend. How are you today? I am fine, thank you!"
    assert split_stream(text, chunk) == [
        "Hello there, my friend.",
        "How are you today?",
        "I am fine, thank you!",
    ]


def test_abbreviations_do_not_split() -> None:
    text = "Dr. Smith arrived at noon. Das ist z.B. ein Test, d.h. nur ein Beispiel."
    assert split_stream(text) == [
        "Dr. Smith arrived at noon.",
        "Das ist z.B. ein Test, d.h. nur ein Beispiel.",
    ]


def test_german_ordinal_does_not_split() -> None:
    assert split_stream("Wir treffen uns am 3. Mai im Park. Bis dann!") == [
        "Wir treffen uns am 3. Mai im Park.",
        "Bis dann!",
    ]


def test_decimal_numbers_do_not_split() -> None:
    assert split_stream("The value is 3.14 exactly. Next sentence here.") == [
        "The value is 3.14 exactly.",
        "Next sentence here.",
    ]


def test_short_sentences_are_merged() -> None:
    assert split_stream("Oh. Yes! That is a great idea indeed.") == [
        "Oh. Yes! That is a great idea indeed.",
    ]


def test_closing_quote_stays_with_sentence() -> None:
    assert split_stream('She said "Run now!" and we ran away fast.') == [
        'She said "Run now!"',
        "and we ran away fast.",
    ]


def test_ellipsis() -> None:
    assert split_stream("Well... I am not entirely sure about that. Maybe later.") == [
        "Well... I am not entirely sure about that.",
        "Maybe later.",
    ]


def test_newline_splits() -> None:
    assert split_stream("First line without end\nSecond line also here") == [
        "First line without end",
        "Second line also here",
    ]


def test_overlong_sentence_is_cut_at_comma() -> None:
    text = "word " * 30 + ", and then " + "more " * 30
    out = split_stream(text, max_chars=200)
    assert len(out) >= 2
    assert all(len(s) <= 200 for s in out[:-1])


def test_nothing_lost() -> None:
    text = "A. B. C. Dr. X! Ok? Yes. 1. 2. 3. Done…"
    joined = " ".join(split_stream(text))
    assert joined.split() == text.split()
