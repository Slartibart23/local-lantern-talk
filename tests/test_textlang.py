from __future__ import annotations

import pytest

from lantern_talk.textlang import guess_language


@pytest.mark.parametrize(
    "text, expected",
    [
        ("*waves* Hallo! Wer bist du? Antworte kurz.", "de"),
        ("Erzähl mir etwas über die Sterne.", "de"),
        ("Was kannst du alles?", "de"),
        ("Hallo!", "de"),
        ("Hello! Who are you and what can you do?", "en"),
        ("Tell me a joke about cats.", "en"),
        ("¿Qué tal? Cuéntame algo de la historia de España.", "es"),
        ("Hola, ¿cómo estás?", "es"),
        ("Bonjour ! Comment ça va aujourd'hui ?", "fr"),
        ("Est-ce que tu peux m'expliquer la photosynthèse ?", "fr"),
        ("Merci beaucoup !", "fr"),
    ],
)
def test_guess(text: str, expected: str) -> None:
    assert guess_language(text) == expected


@pytest.mark.parametrize("text", ["ok", "42", "", "😊", "Lantern", "Pizza"])
def test_ambiguous_returns_none(text: str) -> None:
    assert guess_language(text) is None
