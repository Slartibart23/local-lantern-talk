"""Tests for matching built-in voices to the spoken language (spec §7.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lantern_talk.tts import load_builtin_voices, match_builtin_voice

BUILTINS = load_builtin_voices(Path(__file__).parents[1] / "config" / "builtin_voices.yaml")


def test_eight_builtin_voices() -> None:
    assert {(v.language, v.gender) for v in BUILTINS.values()} == {
        (lang, g) for lang in ("de", "en", "es", "fr") for g in ("male", "female")
    }


@pytest.mark.parametrize(
    "voice, language, expected",
    [
        ("builtin:en_female", "de", "builtin:de_female"),
        ("builtin:en_male", "fr", "builtin:fr_male"),
        ("builtin:de_male", "es", "builtin:es_male"),
        ("builtin:de_female", "de", "builtin:de_female"),   # already matching
        ("builtin:en_female", "ja", "builtin:en_female"),   # no Japanese built-in voice
        ("builtin:en_female", None, "builtin:en_female"),   # language unknown
        (r"C:\voices\me.mp3", "de", r"C:\voices\me.mp3"),   # custom voices never change
    ],
)
def test_match(voice: str, language: str | None, expected: str) -> None:
    assert match_builtin_voice(voice, language, BUILTINS) == expected
