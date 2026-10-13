"""Tests for the speech sanitizer: actions must never be spoken (spec §8)."""

from __future__ import annotations

import random

import pytest

from lantern_talk.sanitizer import ActionFilter, Segment, clean_for_speech, speech_stream, split_segments, spoken_text


def stream(text: str, chunk_sizes: list[int] | None = None) -> list[Segment]:
    """Feed ``text`` through ActionFilter in chunks and collect all segments."""
    f = ActionFilter()
    out: list[Segment] = []
    i = 0
    sizes = chunk_sizes or [1]
    k = 0
    while i < len(text):
        n = sizes[k % len(sizes)]
        out += f.feed(text[i: i + n])
        i += n
        k += 1
    out += f.finish()
    return out


def spoken_from_stream(text: str, chunk_sizes: list[int] | None = None) -> str:
    return clean_for_speech(speech_stream(stream(text, chunk_sizes)))


def kinds(segments: list[Segment]) -> list[tuple[str, str]]:
    return [(s.kind, s.text.strip()) for s in segments if s.text.strip()]


# -- actions ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("*smiles and leans back* Welcome, traveler!", "Welcome, traveler!"),
        ("Welcome! *wipes the counter* What will it be?", "Welcome! What will it be?"),
        ("_nods slowly_ Very well.", "Very well."),
        ("[sighs] Fine, I'll do it.", "Fine, I'll do it."),
        ("Hello there. *waves", "Hello there."),  # unclosed action at the end
        ("*grins* *winks* Got you.", "Got you."),
        ("Aye. *pours an ale, sliding it over* There you go.", "Aye. There you go."),
    ],
)
def test_actions_are_removed(text: str, expected: str) -> None:
    assert spoken_text(text) == expected


def test_action_segments_are_reported() -> None:
    segs = split_segments("*smiles* Hi! [laughter] Bye.")
    assert ("action", "smiles") in kinds(segs)
    assert ("action", "laughter") in kinds(segs)
    assert all(s.kind in {"speech", "action"} for s in segs)


def test_unclosed_action_at_end_is_an_action() -> None:
    segs = split_segments("Sure. *starts walking towards the door")
    assert kinds(segs)[-1] == ("action", "starts walking towards the door")


# -- OOC --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello. ((brb, getting coffee))", "Hello."),
        ("[OOC: this scene is fun] Onward!", "Onward!"),
        ("(OOC: quick note) Onward!", "Onward!"),
        ("(ooc - note) Onward!", "Onward!"),
        ("Onward!\nOOC: I need to go soon.", "Onward!"),
        ("Onward! [[meta comment]]", "Onward!"),
    ],
)
def test_ooc_is_removed(text: str, expected: str) -> None:
    assert spoken_text(text) == expected


def test_ooc_kind() -> None:
    assert kinds(split_segments("[OOC: note] Hi there."))[0] == ("ooc", "note")
    assert kinds(split_segments("Hi there. ((OOC: note))"))[-1] == ("ooc", "note")


def test_emphasis_in_asterisks_counts_as_action() -> None:
    # Spec §8: text between single asterisks is always treated as an action.
    assert spoken_text("The baker uses *magic* for his bread.") == "The baker uses for his bread."


# -- markdown -----------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("# Title\nSome text.", "Title Some text."),
        ("- first point\n- second point", "first point second point"),
        ("1. Eggs\n2. Milk", "Eggs Milk"),
        ("This is **very** important.", "This is very important."),
        ("This is __very__ important.", "This is very important."),
        ("See [the docs](https://example.com) for more.", "See the docs for more."),
        ("Use the `print` function.", "Use the print function."),
        ("> quoted line", "quoted line"),
        ("Here:\n```python\nprint('hi')\n```\nDone.", "Here: Done."),
        ("Visit https://example.com today.", "Visit today."),
        ("2 * 3 = 6", "2 3 = 6"),
        ("my_variable_name is set", "my variable name is set"),
        ("(this is an aside) and more", "this is an aside and more"),
    ],
)
def test_markdown_is_removed(text: str, expected: str) -> None:
    assert spoken_text(text) == expected


def test_code_block_kind() -> None:
    segs = split_segments("```\ncode here\n```")
    assert kinds(segs) == [("code", "code here")]


# -- emojis / symbols ------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("Great job! 🎉🎉", "Great job!"),
        ("I ❤️ this 😊 place.", "I this place."),
        ("~~~ Hello ~~~", "Hello"),
        ("Price: 5 € or $6.", "Price: 5 € or $6."),
        ("Ça va? ¿Qué tal? Grüß dich!", "Ça va? ¿Qué tal? Grüß dich!"),
        ("🎉", ""),
        ("...", ""),
    ],
)
def test_clean_for_speech(text: str, expected: str) -> None:
    assert clean_for_speech(text) == expected


# -- streaming ---------------------------------------------------------------------------

STREAM_CASES = [
    "*smiles and leans back* Welcome, traveler! What brings you here?",
    "Aye. *pours an ale, sliding it over* There you go. _winks_ Enjoy.",
    "Hello. ((brb)) [OOC: note] (OOC: x) Back again.",
    "# Header\n- item one\n- item **two**\n1. first\nDone with [link](http://x.y).",
    "Sure. *starts walking towards the door",
    "Look_at_this and _that_ thing.",
    "Text\n```py\nx = 1\n```\nAfter code.",
    "Start (aside) end.\nOOC: bye\nReal line.",
]


@pytest.mark.parametrize("text", STREAM_CASES)
@pytest.mark.parametrize("sizes", [[1], [2], [3, 1, 5], [7], [1000]])
def test_streaming_matches_batch(text: str, sizes: list[int]) -> None:
    assert spoken_from_stream(text, sizes) == spoken_text(text)


@pytest.mark.parametrize("text", STREAM_CASES)
def test_streaming_random_chunks(text: str) -> None:
    rng = random.Random(42)
    for _ in range(25):
        sizes = [rng.randint(1, 6) for _ in range(20)]
        assert spoken_from_stream(text, sizes) == spoken_text(text)


def test_half_open_action_is_never_emitted_as_speech_mid_stream() -> None:
    f = ActionFilter()
    emitted: list[Segment] = []
    for ch in "Hi there. *slowly draws her sword and":
        emitted += f.feed(ch)
    spoken = speech_stream(emitted)
    assert "draws" not in spoken and "sword" not in spoken
    emitted += f.finish()
    assert all("sword" not in s.text for s in emitted if s.spoken)
    assert any(s.kind == "action" and "sword" in s.text for s in emitted)


def test_action_words_never_spoken_property() -> None:
    """Fuzz: random dialogue with random actions — no action word ever reaches speech."""
    rng = random.Random(7)
    words = ["hello", "there", "friend", "how", "are", "you", "today", "well"]
    action_words = ["SMILE", "NOD", "GRIN", "WAVE", "SIGH"]
    for _ in range(200):
        parts: list[str] = []
        for _ in range(rng.randint(1, 6)):
            r = rng.random()
            aw = " ".join(rng.choice(action_words) for _ in range(rng.randint(1, 3)))
            if r < 0.2:
                parts.append(f"*{aw}*")
            elif r < 0.3:
                parts.append(f"_{aw}_")
            elif r < 0.4:
                parts.append(f"[{aw}]")
            elif r < 0.45:
                parts.append(f"(({aw}))")
            else:
                parts.append(" ".join(rng.choice(words) for _ in range(rng.randint(1, 5))) + rng.choice([".", "!", "?", ","]))
        if rng.random() < 0.3:
            parts.append("*" + rng.choice(action_words))  # unclosed at the end
        text = " ".join(parts)
        sizes = [rng.randint(1, 8) for _ in range(10)]
        spoken = spoken_from_stream(text, sizes)
        for aw in action_words:
            assert aw not in spoken, (text, spoken)
