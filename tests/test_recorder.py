"""Session recording: MP3 appending, offsets, transcript round trip (spec §12)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from lantern_talk.recorder import (
    SESSION_SAMPLE_RATE, CaptureBuffer, SessionItem, SessionRecorder, encode_mp3, estimate_duration,
    format_offset, list_sessions, markdown_segments, mp3_duration, normalize_loudness, parse_offset,
    parse_transcript, segments_markdown,
)

ROOT = Path(__file__).parents[1]
FFMPEG = ROOT / "runtime" / "ffmpeg" / "bin" / "ffmpeg.exe"
needs_ffmpeg = pytest.mark.skipif(not FFMPEG.exists(), reason="portable ffmpeg not installed")


def tone(seconds: float, sr: int = SESSION_SAMPLE_RATE, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_offsets_format_and_parse() -> None:
    assert format_offset(0) == "00:00.0"
    assert format_offset(83.25) == "01:23.2"
    assert format_offset(4000) == "66:40.0"
    assert parse_offset("01:23.2") == pytest.approx(83.2)


def test_normalize_brings_quiet_and_loud_speech_together() -> None:
    quiet = normalize_loudness(tone(1, amp=0.02), SESSION_SAMPLE_RATE)
    loud = normalize_loudness(tone(1, amp=0.9), SESSION_SAMPLE_RATE)
    rms = lambda a: 20 * np.log10(np.sqrt(np.mean(a**2)))  # noqa: E731
    assert abs(rms(quiet) - rms(loud)) < 1.0
    assert np.max(np.abs(loud)) <= 10 ** (-1 / 20) + 1e-6
    silence = np.zeros(1000, dtype=np.float32)
    assert np.array_equal(normalize_loudness(silence, SESSION_SAMPLE_RATE), silence)


@needs_ffmpeg
def test_concatenated_mp3_pieces_have_exact_durations() -> None:
    a = encode_mp3(tone(1.0), FFMPEG, 64)
    b = encode_mp3(tone(2.5), FFMPEG, 64)
    assert not a.startswith(b"ID3")
    da, db = mp3_duration(a), mp3_duration(b)
    assert 1.0 <= da < 1.2 and 2.5 <= db < 2.7  # encoder delay / padding add a few frames
    assert mp3_duration(a + b) == pytest.approx(da + db)


@needs_ffmpeg
def test_recorder_writes_audio_and_transcript(tmp_path: Path) -> None:
    recorded: list[tuple[int, str, float, float | None]] = []
    rec = SessionRecorder(tmp_path, FFMPEG, lambda *a: recorded.append(a), gap_ms=lambda: 500)
    name = rec.new_name("lena")
    rec.start(name, "Lena — test", ["Character: Lena (assistant)"])
    rec.add(SessionItem(1, "assistant", "Lena", "Hallo! Was steht an?", ["greeting"]))
    rec.add(SessionItem(2, "user", "You", "Ein Ausflug.", ["spoken", "de"], tone(1.2, 48_000), 48_000))
    rec.add(SessionItem(3, "assistant", "Lena", segments_markdown(
        [{"kind": "action", "text": "lächelt"}, {"kind": "speech", "text": "Wie schön!"}], True),
        ["German female"], tone(2.0)))
    rec.add(SessionItem(4, "user", "You", "Getippt.", ["typed"]))
    rec.wait_idle()
    rec.stop()
    rec.close()

    by_id = {r[0]: r for r in recorded}
    assert by_id[1][2] == 0.0 and by_id[1][3] is None           # text only, no audio
    assert by_id[2][2] == 0.0 and by_id[2][3] >= 1.2            # first audio: no gap
    assert by_id[3][2] == pytest.approx(by_id[2][3] + 0.5, abs=0.001)  # gap before the reply
    total = mp3_duration((tmp_path / f"{name}.mp3").read_bytes())
    assert by_id[3][2] + by_id[3][3] == pytest.approx(total)
    assert by_id[4][2] == pytest.approx(total)                  # typed: offset = position at that time

    parsed = parse_transcript((tmp_path / f"{name}.md").read_text(encoding="utf-8"))
    assert parsed["title"] == "Lena — test"
    e = parsed["entries"]
    assert [x["role"] for x in e] == ["assistant", "user", "assistant", "user"]
    assert e[0]["greeting"] and e[0]["duration"] is None
    assert e[1]["source"] == "voice" and e[1]["text"] == "Ein Ausflug."
    assert e[2]["offset"] == pytest.approx(by_id[3][2], abs=0.06)
    assert e[2]["segments"] == [{"kind": "action", "text": "lächelt"}, {"kind": "speech", "text": "Wie schön!"}]
    assert e[3]["source"] == "text" and e[3]["duration"] is None

    sessions = list_sessions(tmp_path)
    assert sessions[0]["name"] == name and sessions[0]["character"] == "lena"
    assert sessions[0]["duration"] == pytest.approx(total, abs=0.1)
    assert estimate_duration(tmp_path / f"{name}.mp3") == pytest.approx(total, abs=0.1)


@needs_ffmpeg
def test_resume_appends_at_the_end(tmp_path: Path) -> None:
    rec = SessionRecorder(tmp_path, FFMPEG)
    name = rec.new_name("max")
    rec.start(name, "Max", [])
    rec.add(SessionItem(1, "user", "You", "Eins", ["spoken"], tone(1.0)))
    rec.wait_idle()
    first = rec.duration
    rec.stop()
    rec.close()
    recorded = []
    rec2 = SessionRecorder(tmp_path, FFMPEG, lambda *a: recorded.append(a))
    rec2.resume(name)
    rec2.add(SessionItem(2, "assistant", "Max", "Zwei", [], tone(1.0)))
    rec2.wait_idle()
    rec2.close()
    assert recorded[0][2] == pytest.approx(first + 0.5)
    entries = parse_transcript((tmp_path / f"{name}.md").read_text(encoding="utf-8"))["entries"]
    assert len(entries) == 2


def test_new_names_never_collide(tmp_path: Path) -> None:
    rec = SessionRecorder(tmp_path, FFMPEG)
    from datetime import datetime

    now = datetime(2026, 10, 13, 15, 38)
    first = rec.new_name("lena", now)
    assert first == "2026-10-13_1538_lena"
    (tmp_path / f"{first}.md").write_text("# x\n", encoding="utf-8")
    assert rec.new_name("lena", now) == "2026-10-13_1538_lena_2"
    rec.close()
    assert SessionRecorder.valid_name(first)
    assert not SessionRecorder.valid_name("../evil")
    assert not SessionRecorder.valid_name("a b")


def test_actions_follow_the_display_setting() -> None:
    segs = [{"kind": "speech", "text": "Hallo."}, {"kind": "action", "text": "winkt"},
            {"kind": "ooc", "text": "brb"}, {"kind": "thinking", "text": "secret"}]
    shown = segments_markdown(segs, True)
    assert "_[not spoken: winkt]_" in shown and "_[OOC, not spoken: brb]_" in shown
    assert "secret" not in shown
    assert segments_markdown(segs, False) == "Hallo."
    assert markdown_segments(shown) == segs[:3]


def test_capture_buffer_cuts_full_quality_audio() -> None:
    buf = CaptureBuffer(keep_s=2.0)
    sr = 48_000
    signal = np.arange(sr * 3, dtype=np.float32)  # 3 s, sample value = index
    for i in range(0, len(signal), 960):
        buf.feed(signal[i:i + 960], sr)
    # 16 kHz range of the last second -> 48 kHz samples 96000..144000
    cut = buf.slice_16k(32_000, 48_000)
    assert len(cut) == 48_000 and cut[0] == 96_000 and cut[-1] == 143_999
    assert len(buf.slice_16k(0, 16_000)) == 0  # older than keep_s: dropped
    buf.reset()
    assert len(buf.slice_16k(0, 100)) == 0


def test_list_sessions_sorted_by_last_change(tmp_path: Path) -> None:
    for i, name in enumerate(["2026-10-01_1000_max", "2026-10-02_1000_lena"]):
        (tmp_path / f"{name}.md").write_text(f"# {name}\n", encoding="utf-8")
        os.utime(tmp_path / f"{name}.md", (time.time() + i, time.time() + (1 - i) * 10))
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    names = [s["name"] for s in list_sessions(tmp_path)]
    assert names == ["2026-10-01_1000_max", "2026-10-02_1000_lena"]
