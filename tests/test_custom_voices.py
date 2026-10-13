"""Custom voice preparation: transcript, caching, ffmpeg preparation and quality report."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lantern_talk import tts as tts_module
from lantern_talk.paths import Paths
from lantern_talk.tts import VoiceLibrary, file_fingerprint

ROOT = Path(__file__).parents[1]
FFMPEG = Paths(ROOT).ffmpeg


class FakePrompt:
    def __init__(self, ref_text: str, n_samples: int = 0) -> None:
        self.ref_text = ref_text
        self.n_samples = n_samples

    def save(self, path: str) -> None:
        Path(path).write_text(self.ref_text, encoding="utf-8")


class FakeTTS:
    def __init__(self) -> None:
        self.prompts: list[FakePrompt] = []

    def make_prompt(self, audio: np.ndarray, sample_rate: int, ref_text: str) -> FakePrompt:
        self.prompts.append(FakePrompt(ref_text, len(audio)))
        return self.prompts[-1]


def tone_clip(path: Path, seconds: float, lead_silence: float = 0.0) -> None:
    sr = 24_000
    t = np.arange(int(sr * seconds)) / sr
    voice = (0.3 * (np.sin(2 * np.pi * 3 * t) > 0) * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    sf.write(path, np.concatenate([np.zeros(int(sr * lead_silence), np.float32), voice]), sr)


def make_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ffmpeg: Path | None = None,
                 transcript: str = "Hallo, das ist meine Stimme."):  # noqa: ANN201
    monkeypatch.setattr(tts_module, "_load_prompt", lambda p: FakePrompt(Path(p).read_text(encoding="utf-8")))
    voices = tmp_path / "voices"
    voices.mkdir()
    calls: list[int] = []

    def transcribe(audio: np.ndarray, sr: int, lang: str | None) -> str:
        calls.append(len(audio))
        return transcript

    fake = FakeTTS()
    lib = VoiceLibrary(fake, ROOT / "config" / "builtin_voices.yaml", tmp_path / "builtin",
                       voices / ".cache", transcribe, ffmpeg=ffmpeg)
    return lib, fake, calls, voices


def test_whisper_transcript_is_used_and_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lib, fake, calls, voices = make_library(tmp_path, monkeypatch)
    clip = voices / "anna.wav"
    tone_clip(clip, 4)
    assert lib.reference_text(clip) is None
    lib.prompt_for(str(clip))
    assert lib.reference_text(clip) == ("Hallo, das ist meine Stimme.", "whisper")
    assert lib.custom_language(clip) == "de"
    # A new session reuses the cached prompt: no second transcription.
    lib2 = VoiceLibrary(fake, ROOT / "config" / "builtin_voices.yaml", lib.builtin_dir,
                        lib.custom_cache_dir, lib.transcribe)
    lib2.prompt_for(str(clip))
    assert len(calls) == 1


def test_user_text_overrides_whisper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lib, fake, calls, voices = make_library(tmp_path, monkeypatch)
    clip = voices / "anna.wav"
    tone_clip(clip, 4)
    lib.prompt_for(str(clip))
    lib.set_reference_text(clip, "  Das ist der echte Text.  ")
    assert (voices / "anna.txt").read_text(encoding="utf-8") == "Das ist der echte Text.\n"
    assert lib.reference_text(clip) == ("Das ist der echte Text.", "user")
    assert lib.prompt_for(str(clip)).ref_text == "Das ist der echte Text."
    assert len(calls) == 1  # Whisper not needed again
    assert len(list(lib.custom_cache_dir.glob("*.pt"))) == 1  # the old prompt was removed


def test_empty_user_text_falls_back_to_whisper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lib, fake, calls, voices = make_library(tmp_path, monkeypatch)
    clip = voices / "anna.wav"
    tone_clip(clip, 4)
    lib.set_reference_text(clip, "Mein Text")
    lib.prompt_for(str(clip))
    lib.set_reference_text(clip, "   ")
    assert not (voices / "anna.txt").exists()
    assert lib.prompt_for(str(clip)).ref_text == "Hallo, das ist meine Stimme."


def test_rename_keeps_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lib, fake, calls, voices = make_library(tmp_path, monkeypatch)
    clip = voices / "anna.wav"
    tone_clip(clip, 4)
    lib.prompt_for(str(clip))
    renamed = voices / "Anna (ruhig).wav"
    os.replace(clip, renamed)  # renaming keeps the file's time stamp
    assert file_fingerprint(renamed) == file_fingerprint(clip.with_name(renamed.name))
    lib.prompt_for(str(renamed))
    assert len(calls) == 1 and len(fake.prompts) == 1


def test_cleanup_removes_only_orphans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lib, fake, calls, voices = make_library(tmp_path, monkeypatch)
    keep, gone = voices / "keep.wav", voices / "gone.wav"
    tone_clip(keep, 4)
    tone_clip(gone, 5)
    lib.prompt_for(str(keep))
    lib.prompt_for(str(gone))
    gone_fp = file_fingerprint(gone)
    gone.unlink()
    lib.cleanup_cache(voices)
    names = [f.name for f in lib.custom_cache_dir.iterdir()]
    assert names and all(n.startswith(file_fingerprint(keep)) for n in names)
    assert not any(n.startswith(gone_fp) for n in names)


@pytest.mark.skipif(not FFMPEG.exists(), reason="ffmpeg not installed")
def test_prepared_clip_is_used_and_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lib, fake, calls, voices = make_library(tmp_path, monkeypatch, ffmpeg=FFMPEG)
    clip = voices / "padded.wav"
    tone_clip(clip, 4, lead_silence=2.0)
    lib.prompt_for(str(clip))
    assert fake.prompts[0].n_samples < 24_000 * 4.6  # leading silence trimmed by ffmpeg
    assert list(lib.custom_cache_dir.glob("*.prepared.wav"))
    report = lib.clip_report(clip)
    assert report is not None and 3.4 < report["duration_s"] < 4.6
    assert list(lib.custom_cache_dir.glob("*.meta.json"))


def test_builtin_voices_come_from_the_shipped_clips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lib, fake, calls, voices = make_library(tmp_path, monkeypatch)
    fake.design = lambda *a: pytest.fail("shipped clips must not be designed again")  # type: ignore[attr-defined]
    for voice_id in lib.builtins:
        assert (lib.shipped_dir / f"{voice_id}.flac").exists()
    lib.prompt_for("builtin:de_female")
    assert (lib.builtin_dir / "de_female.pt").exists() and (lib.builtin_dir / "de_female.wav").exists()
    assert fake.prompts[-1].ref_text == lib.builtins["de_female"].reference_text
    assert calls == []  # no Whisper check needed for shipped clips


def test_chosen_clip_language_wins_and_guides_whisper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lib, fake, calls, voices = make_library(tmp_path, monkeypatch)
    seen: list[str | None] = []
    lib.transcribe = lambda audio, sr, lang: (seen.append(lang), "Hallo, das ist meine Stimme.")[1]
    clip = voices / "anna.wav"
    tone_clip(clip, 4)
    lib.prompt_for(str(clip))
    assert seen == [None] and lib.custom_language(clip) == "de"  # detected from the transcript
    lib.set_language(clip, "fr")
    assert lib.custom_language(clip) == "fr" and lib.user_language(clip) == "fr"
    assert (voices / "anna.lang").read_text(encoding="utf-8").strip() == "fr"
    lib.prompt_for(str(clip))
    assert seen == [None, "fr"]  # transcribed again, now with the chosen language
    lib.set_language(clip, "")
    assert not (voices / "anna.lang").exists()
    lib.prompt_for(str(clip))
    assert seen == [None, "fr", None] and lib.custom_language(clip) == "de"  # detected again
