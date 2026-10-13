"""Session recording: MP3 appending, offsets, transcript round trip (spec §12)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from lantern_talk.recorder import (
    INDEX_DIR, SESSION_SAMPLE_RATE, CaptureBuffer, SessionItem, SessionRecorder, encode_mp3, estimate_duration,
    list_sessions, load_session, markdown_segments, mp3_duration, normalize_loudness, parse_offset, spoken_text,
)

ROOT = Path(__file__).parents[1]
FFMPEG = ROOT / "runtime" / "ffmpeg" / "bin" / "ffmpeg.exe"
needs_ffmpeg = pytest.mark.skipif(not FFMPEG.exists(), reason="portable ffmpeg not installed")


def tone(seconds: float, sr: int = SESSION_SAMPLE_RATE, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_parse_offset_of_old_transcripts() -> None:
    assert parse_offset("01:23.2") == pytest.approx(83.2)
    assert parse_offset("66:40.0") == 4000


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
    rec.start(name, "Lena — test")
    rec.add(SessionItem(1, "assistant", "Lena", "Hallo! Was steht an?", ["greeting"]))
    rec.add(SessionItem(2, "user", "You", "Ein Ausflug.", ["spoken", "de"], tone(1.2, 48_000), 48_000))
    segs = [{"kind": "action", "text": "lächelt"}, {"kind": "speech", "text": "Wie schön!"},
            {"kind": "thinking", "text": "geheim"}]
    rec.add(SessionItem(3, "assistant", "Lena", spoken_text(segs), ["German female"], tone(2.0), segments=segs))
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

    # the transcript reads like a screenplay: only who said what
    md = (tmp_path / f"{name}.md").read_text(encoding="utf-8")
    assert md == ("# Lena — test\n\nLena: Hallo! Was steht an?\n\nYou: Ein Ausflug.\n\n"
                  "Lena: Wie schön!\n\nYou: Getippt.\n\n")
    # offsets and actions live in the hidden index, for replay
    assert (tmp_path / INDEX_DIR / f"{name}.jsonl").exists()
    parsed = load_session(tmp_path, name)
    assert parsed["title"] == "Lena — test"
    e = parsed["entries"]
    assert [x["role"] for x in e] == ["assistant", "user", "assistant", "user"]
    assert e[0]["greeting"] and e[0]["duration"] is None
    assert e[1]["source"] == "voice" and e[1]["text"] == "Ein Ausflug."
    assert e[2]["offset"] == pytest.approx(by_id[3][2], abs=0.001)
    assert e[2]["segments"] == segs[:2]  # model reasoning is never kept
    assert e[3]["source"] == "text" and e[3]["duration"] is None

    sessions = list_sessions(tmp_path)
    assert sessions[0]["name"] == name and sessions[0]["character"] == "lena"
    assert sessions[0]["duration"] == pytest.approx(total, abs=0.1)
    assert estimate_duration(tmp_path / f"{name}.mp3") == pytest.approx(total, abs=0.1)


@needs_ffmpeg
def test_resume_appends_at_the_end(tmp_path: Path) -> None:
    rec = SessionRecorder(tmp_path, FFMPEG)
    name = rec.new_name("max")
    rec.start(name, "Max")
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
    assert len(load_session(tmp_path, name)["entries"]) == 2
    assert (tmp_path / f"{name}.md").read_text(encoding="utf-8") == "# Max\n\nYou: Eins\n\nMax: Zwei\n\n"


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


def test_transcript_contains_only_what_was_said() -> None:
    segs = [{"kind": "speech", "text": "Hallo. "}, {"kind": "action", "text": "winkt"},
            {"kind": "speech", "text": " Wie geht's?"}, {"kind": "ooc", "text": "brb"},
            {"kind": "thinking", "text": "secret"}]
    assert spoken_text(segs) == "Hallo. Wie geht's?"
    assert spoken_text([{"kind": "action", "text": "nickt"}]) == ""


def test_old_format_transcripts_can_still_be_opened(tmp_path: Path) -> None:
    (tmp_path / "2026-10-13_1200_lena.md").write_text(
        "# Lena — 13 Oct 2026, 12:00\n\n- Character: Lena (assistant)\n\n"
        "### [00:00.0] Lena · greeting · no audio\n\nHallo!\n\n"
        "### [00:01.5] You · spoken · de · 1.2 s\n\nHi.\n\n"
        "### [00:03.2] Lena · German female · 2.0 s\n\n_[not spoken: winkt]_ Schön!\n\n", encoding="utf-8")
    e = load_session(tmp_path, "2026-10-13_1200_lena")["entries"]
    assert [x["role"] for x in e] == ["assistant", "user", "assistant"]
    assert e[1]["offset"] == pytest.approx(1.5) and e[1]["duration"] == 1.2
    assert e[2]["segments"] == [{"kind": "action", "text": "winkt"}, {"kind": "speech", "text": "Schön!"}]
    assert markdown_segments("Nur Text.") == [{"kind": "speech", "text": "Nur Text."}]


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
    (tmp_path / INDEX_DIR).mkdir()
    (tmp_path / INDEX_DIR / "2026-09-01_1000_gone.jsonl").write_text("{}\n", encoding="utf-8")
    names = [s["name"] for s in list_sessions(tmp_path)]
    assert names == ["2026-10-01_1000_max", "2026-10-02_1000_lena"]
    assert not (tmp_path / INDEX_DIR / "2026-09-01_1000_gone.jsonl").exists()  # deleted session
