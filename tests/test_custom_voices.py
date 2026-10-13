"""Custom voice preparation: Whisper transcript vs. user-provided reference text."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lantern_talk import tts as tts_module
from lantern_talk.tts import VoiceLibrary

ROOT = Path(__file__).parents[1]


def owned_by(stem: str, name: str) -> bool:
    """Cache file of voice ``stem``: "<stem>-<16 hex>..." (a fingerprint may start with a digit)."""
    import re

    return re.match(re.escape(stem) + "-[0-9a-f]{16}[-.]", name) is not None


class FakePrompt:
    def __init__(self, ref_text: str) -> None:
        self.ref_text = ref_text

    def save(self, path: str) -> None:
        Path(path).write_text(self.ref_text, encoding="utf-8")


class FakeTTS:
    def __init__(self) -> None:
        self.prompt_texts: list[str] = []

    def make_prompt(self, audio: np.ndarray, sample_rate: int, ref_text: str) -> FakePrompt:
        self.prompt_texts.append(ref_text)
        return FakePrompt(ref_text)


@pytest.fixture()
def lib(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    monkeypatch.setattr(tts_module, "_load_prompt", lambda p: FakePrompt(Path(p).read_text(encoding="utf-8")))
    voices = tmp_path / "voices"
    voices.mkdir()
    for name in ("clip.wav", "clip-2.wav"):
        sf.write(voices / name, np.zeros(24_000 * 4, dtype=np.float32), 24_000)
    calls: list[int] = []

    def transcribe(audio: np.ndarray, sr: int, lang: str | None) -> str:
        calls.append(len(audio))
        return "whisper heard this"

    fake = FakeTTS()
    library = VoiceLibrary(fake, ROOT / "config" / "builtin_voices.yaml", tmp_path / "builtin",
                           voices / ".cache", transcribe)
    return library, fake, calls, voices


def test_whisper_transcript_is_used_and_cached(lib) -> None:  # noqa: ANN001
    library, fake, calls, voices = lib
    clip = voices / "clip.wav"
    assert library.reference_text(clip) is None
    library.prompt_for(str(clip))
    assert library.reference_text(clip) == ("whisper heard this", "whisper")
    assert fake.prompt_texts == ["whisper heard this"]
    # A new session reuses the cached prompt: no second transcription.
    library2 = VoiceLibrary(fake, ROOT / "config" / "builtin_voices.yaml",
                            library.builtin_dir, library.custom_cache_dir, library.transcribe)
    library2.prompt_for(str(clip))
    assert len(calls) == 1


def test_user_text_overrides_whisper(lib) -> None:  # noqa: ANN001
    library, fake, calls, voices = lib
    clip = voices / "clip.wav"
    library.prompt_for(str(clip))
    library.set_reference_text(clip, "  Das ist der echte Text.  ")
    assert (voices / "clip.txt").read_text(encoding="utf-8") == "Das ist der echte Text.\n"
    assert library.reference_text(clip) == ("Das ist der echte Text.", "user")
    prompt = library.prompt_for(str(clip))
    assert prompt.ref_text == "Das ist der echte Text."
    assert len(calls) == 1  # Whisper not needed again
    pts = [p.name for p in library.custom_cache_dir.iterdir() if owned_by("clip", p.name) and p.suffix == ".pt"]
    assert len(pts) == 1  # the prompt built from the Whisper text was removed


def test_empty_user_text_falls_back_to_whisper(lib) -> None:  # noqa: ANN001
    library, fake, calls, voices = lib
    clip = voices / "clip.wav"
    library.set_reference_text(clip, "Mein Text")
    library.prompt_for(str(clip))
    library.set_reference_text(clip, "   ")
    assert not (voices / "clip.txt").exists()
    assert library.prompt_for(str(clip)).ref_text == "whisper heard this"


def test_cleanup_never_touches_similarly_named_voice(lib) -> None:  # noqa: ANN001
    library, fake, calls, voices = lib
    library.prompt_for(str(voices / "clip-2.wav"))
    before = {p.name for p in library.custom_cache_dir.iterdir() if owned_by("clip-2", p.name)}
    library.set_reference_text(voices / "clip.wav", "Anderer Text")
    library.prompt_for(str(voices / "clip.wav"))
    after = {p.name for p in library.custom_cache_dir.iterdir() if owned_by("clip-2", p.name)}
    assert before and before == after
