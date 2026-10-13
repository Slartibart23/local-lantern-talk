"""Reference clip preparation (ffmpeg) and quality analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lantern_talk.audio_prep import PREP_SAMPLE_RATE, PrepError, analyze, decode, prepare
from lantern_talk.paths import Paths

ROOT = Path(__file__).parents[1]
FFMPEG = Paths(ROOT).ffmpeg
needs_ffmpeg = pytest.mark.skipif(not FFMPEG.exists(), reason="ffmpeg not installed (run install.bat)")


def speechlike(seconds: float, sr: int = 44_100, level: float = 0.3, noise: float = 0.0) -> np.ndarray:
    """Amplitude-modulated tones (syllable-like bursts) plus optional noise."""
    t = np.arange(int(sr * seconds)) / sr
    envelope = (np.sin(2 * np.pi * 3 * t) > 0).astype(np.float32)  # 3 bursts per second
    voice = level * envelope * (np.sin(2 * np.pi * 180 * t) + 0.5 * np.sin(2 * np.pi * 720 * t)) / 1.5
    rng = np.random.default_rng(0)
    return (voice + noise * rng.standard_normal(len(t))).astype(np.float32)


def with_silence(audio: np.ndarray, sr: int, lead: float, trail: float) -> np.ndarray:
    return np.concatenate([np.zeros(int(sr * lead), np.float32), audio, np.zeros(int(sr * trail), np.float32)])


@needs_ffmpeg
def test_prepare_trims_silence_resamples_and_normalizes(tmp_path: Path) -> None:
    src = tmp_path / "clip.wav"
    sf.write(src, with_silence(speechlike(4.0, level=0.05), 44_100, 1.5, 2.0), 44_100)  # quiet, padded
    out = prepare(src, tmp_path / "prepared.wav", FFMPEG)
    data, sr = sf.read(tmp_path / "prepared.wav", dtype="float32")
    assert sr == PREP_SAMPLE_RATE and data.ndim == 1
    assert 3.4 < len(out) / PREP_SAMPLE_RATE < 4.6  # 3.5 s of silence removed
    assert np.sqrt(np.mean(out**2)) > 0.05  # louder than the quiet source


@needs_ffmpeg
def test_decode_reads_mp3(tmp_path: Path) -> None:
    import subprocess

    wav, mp3 = tmp_path / "a.wav", tmp_path / "a.mp3"
    sf.write(wav, speechlike(2.0), 44_100)
    subprocess.run([str(FFMPEG), "-loglevel", "error", "-i", str(wav), str(mp3)], check=True)
    assert abs(len(decode(mp3, FFMPEG)) / PREP_SAMPLE_RATE - 2.0) < 0.1


@needs_ffmpeg
def test_unreadable_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"this is not audio")
    with pytest.raises(PrepError):
        prepare(bad, tmp_path / "out.wav", FFMPEG)


def test_analyze_clean_clip() -> None:
    clip = speechlike(6.0, sr=PREP_SAMPLE_RATE)
    report = analyze(clip, clip)
    assert report.warnings == [] and report.snr_db > 30 and not report.clipped


def test_analyze_noisy_short_and_clipped() -> None:
    noisy = speechlike(2.0, sr=PREP_SAMPLE_RATE, noise=0.08)
    loud = np.clip(speechlike(2.0, sr=PREP_SAMPLE_RATE, level=3.0), -1, 1)
    report = analyze(loud, noisy)
    text = " ".join(report.warnings)
    assert "Only 2.0 s" in text and "background noise" in text and "distorted" in text


def test_analyze_long_clip() -> None:
    clip = speechlike(25.0, sr=PREP_SAMPLE_RATE)
    assert "first 15 s" in analyze(clip, clip).warnings[0]
