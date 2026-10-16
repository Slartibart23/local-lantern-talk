"""Session recording (spec §12): one MP3 of the whole dialogue plus a Markdown transcript.

Each utterance is encoded on its own (in a background thread) and appended to the
session MP3, so the file is always playable up to the last complete utterance, even
after a crash. Waiting time is not recorded; a short fixed gap separates the turns.
The Markdown transcript reads like a screenplay ("Elara: …"); the offset of every
utterance in the MP3 is kept in a hidden index file for replay.

MP3 frames of separately encoded pieces can simply be concatenated. No Xing/Info
header or ID3 tag is written, so the playback position of every piece is exactly the
sum of the frame durations before it.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .audio_io import resample

log = logging.getLogger(__name__)

SESSION_SAMPLE_RATE = 24_000      # OmniVoice's rate; microphone audio is resampled to it
TARGET_RMS_DBFS = -20.0           # common loudness of user and AI speech
PEAK_DBFS = -1.0
MAX_GAIN_DB = 20.0
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,120}$")


# ---------------------------------------------------------------------------- audio helpers

def normalize_loudness(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Bring speech to a common loudness (RMS of the voiced parts), without clipping."""
    audio = audio.astype(np.float32, copy=False)
    if not len(audio):
        return audio
    frame = max(1, sample_rate // 50)  # 20 ms
    n = len(audio) // frame
    if n:
        rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
        voiced = rms[rms > 10 ** (-50 / 20)]
        level = float(np.sqrt(np.mean(voiced**2))) if len(voiced) else 0.0
    else:
        level = float(np.sqrt(np.mean(audio**2)))
    if level < 1e-5:
        return audio  # silence: leave it alone
    gain_db = float(np.clip(TARGET_RMS_DBFS - 20 * np.log10(level), -MAX_GAIN_DB, MAX_GAIN_DB))
    out = audio * (10 ** (gain_db / 20))
    peak = float(np.max(np.abs(out)))
    limit = 10 ** (PEAK_DBFS / 20)
    if peak > limit:
        out *= limit / peak
    return out.astype(np.float32)


_BITRATES = {  # kbit/s by bitrate index, Layer III
    "mpeg1": [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0],
    "mpeg2": [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0],
}
_RATES = {3: [44100, 48000, 32000], 2: [22050, 24000, 16000], 0: [11025, 12000, 8000]}


def _frame_info(data: bytes | memoryview, i: int) -> tuple[int, int, int] | None:
    """(frame length in bytes, samples per frame, sample rate) of an MPEG Layer III frame at ``i``."""
    if i + 4 > len(data) or data[i] != 0xFF or (data[i + 1] & 0xE0) != 0xE0:
        return None
    version = (data[i + 1] >> 3) & 0x03   # 3 = MPEG-1, 2 = MPEG-2, 0 = MPEG-2.5
    layer = (data[i + 1] >> 1) & 0x03     # 1 = Layer III
    if version == 1 or layer != 1:
        return None
    br_index = (data[i + 2] >> 4) & 0x0F
    sr_index = (data[i + 2] >> 2) & 0x03
    if sr_index == 3:
        return None
    bitrate = _BITRATES["mpeg1" if version == 3 else "mpeg2"][br_index] * 1000
    if not bitrate:
        return None
    rate = _RATES[version][sr_index]
    padding = (data[i + 2] >> 1) & 0x01
    samples = 1152 if version == 3 else 576
    length = samples // 8 * bitrate // rate + padding
    return length, samples, rate


def mp3_duration(data: bytes | memoryview) -> float:
    """Exact playing time of concatenated MP3 frames (no tags expected)."""
    i, seconds = 0, 0.0
    while i < len(data):
        info = _frame_info(data, i)
        if info is None:
            i += 1  # resync (should not happen with our own files)
            continue
        length, samples, rate = info
        seconds += samples / rate
        i += length
    return seconds


def mp3_cut_position(data: bytes | memoryview, seconds: float) -> int:
    """Byte position of the first frame that starts at or after ``seconds``."""
    i, t = 0, 0.0
    while i < len(data):
        if t >= seconds - 1e-6:
            return i
        info = _frame_info(data, i)
        if info is None:
            i += 1
            continue
        length, samples, rate = info
        t += samples / rate
        i += length
    return len(data)


def estimate_duration(path: Path) -> float:
    """Quick duration of a constant-bitrate MP3 from its size."""
    try:
        with path.open("rb") as f:
            head = f.read(4)
        info = _frame_info(head, 0)
        if info is None:
            return 0.0
        length, samples, rate = info
        return path.stat().st_size / length * samples / rate
    except OSError:
        return 0.0


def encode_mp3(audio: np.ndarray, ffmpeg: Path, bitrate_kbps: int = 64,
               sample_rate: int = SESSION_SAMPLE_RATE) -> bytes:
    """Mono float32 -> constant-bitrate MP3 frames without tags."""
    cmd = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", str(sample_rate), "-ac", "1",
           "-i", "pipe:0", "-c:a", "libmp3lame", "-b:a", f"{bitrate_kbps}k", "-write_xing", "0",
           "-id3v2_version", "0", "-f", "mp3", "pipe:1"]
    out = subprocess.run(cmd, input=np.ascontiguousarray(audio, dtype=np.float32).tobytes(), capture_output=True,
                         timeout=120, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if out.returncode != 0 or not out.stdout:
        raise RuntimeError(f"MP3 encoding failed: {out.stderr.decode(errors='replace').strip()[:300]}")
    return out.stdout


# ---------------------------------------------------------------------------- microphone capture

class CaptureBuffer:
    """Keeps the last seconds of full-quality microphone audio (spec §12.1).

    The VAD works on a 16 kHz copy; an utterance found there is cut out of this
    buffer by mapping its 16 kHz sample positions to the capture rate.
    """

    def __init__(self, keep_s: float = 45.0) -> None:
        self.keep_s = keep_s
        self.reset()

    def reset(self) -> None:
        self._chunks: deque[np.ndarray] = deque()
        self._start = 0      # absolute index of the first kept sample
        self._total = 0      # samples received since reset
        self.sample_rate = 0

    def feed(self, samples: np.ndarray, sample_rate: int) -> None:
        if sample_rate != self.sample_rate:
            self.reset()
            self.sample_rate = sample_rate
        self._chunks.append(samples.astype(np.float32, copy=False))
        self._total += len(samples)
        limit = int(self.keep_s * sample_rate)
        while self._chunks and self._total - self._start - len(self._chunks[0]) >= limit:
            self._start += len(self._chunks.popleft())

    def slice_16k(self, start: int, end: int, rate_16k: int = 16_000) -> np.ndarray:
        """The capture-rate audio for the 16 kHz sample range ``[start, end)``."""
        if not self.sample_rate or end <= start:
            return np.zeros(0, dtype=np.float32)
        a = max(int(round(start * self.sample_rate / rate_16k)), self._start)
        b = min(int(round(end * self.sample_rate / rate_16k)), self._total)
        if b <= a:
            return np.zeros(0, dtype=np.float32)
        data = np.concatenate(list(self._chunks))
        return data[a - self._start: b - self._start].copy()


# ---------------------------------------------------------------------------- transcript

def parse_offset(text: str) -> float:
    minutes, sec = text.split(":")
    return int(minutes) * 60 + float(sec)


def spoken_text(segments: list[dict[str, str]]) -> str:
    """Only what was actually said: the transcript reads like a screenplay."""
    text = " ".join(seg["text"].strip() for seg in segments if seg["kind"] == "speech" and seg["text"].strip())
    return re.sub(r"[ \t]+", " ", text).strip()


def visible_segments(segments: list[dict[str, str]]) -> list[dict[str, str]]:
    """Segments for the session view (model reasoning is never kept)."""
    return [{"kind": s["kind"], "text": s["text"]} for s in segments if s["kind"] != "thinking"]


# Transcripts recorded before the screenplay format (13 Oct 2026) marked actions inline and
# had a "### [offset] speaker · tags" heading per utterance; they can still be opened.
_MARK_RE = re.compile(r"_\[(OOC, )?not spoken: (.*?)\]_|```\n(.*?)\n```\n_\[not spoken: code\]_", re.DOTALL)


def markdown_segments(text: str) -> list[dict[str, str]]:
    """Segments of an utterance in an old-format transcript."""
    out: list[dict[str, str]] = []
    pos = 0
    for m in _MARK_RE.finditer(text):
        before = text[pos:m.start()].strip()
        if before:
            out.append({"kind": "speech", "text": before})
        if m.group(3) is not None:
            out.append({"kind": "code", "text": m.group(3)})
        else:
            out.append({"kind": "ooc" if m.group(1) else "action", "text": m.group(2)})
        pos = m.end()
    rest = text[pos:].strip()
    if rest:
        out.append({"kind": "speech", "text": rest})
    return out


_ENTRY_RE = re.compile(r"^### \[(\d+:\d+(?:\.\d+)?)\] (.+?)(?: · (.*))?$")


def parse_transcript(md: str) -> dict[str, Any]:
    """Read an old-format session transcript (with offsets in the Markdown)."""
    title = ""
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    body: list[str] = []

    def flush() -> None:
        if current is not None:
            text = "\n".join(body).strip()
            if current["role"] == "user":
                current["text"] = text
            else:
                current["segments"] = markdown_segments(text)
            entries.append(current)

    for line in md.splitlines():
        if line.startswith("# ") and not title:
            title = line[2:].strip()
            continue
        m = _ENTRY_RE.match(line)
        if m:
            flush()
            body = []
            tags = [t.strip() for t in (m.group(3) or "").split(" · ") if t.strip()]
            duration = next((float(t[:-2]) for t in tags if re.fullmatch(r"\d+(\.\d+)? s", t)), None)
            speaker = m.group(2).strip()
            role = "user" if speaker == "You" else "assistant"
            current = {"role": role, "speaker": speaker, "offset": parse_offset(m.group(1)),
                       "duration": None if "no audio" in tags else duration, "tags": tags}
            if role == "user":
                current["source"] = "voice" if "spoken" in tags else "text"
            else:
                current["greeting"] = "greeting" in tags
            continue
        if current is not None:
            body.append(line)
    flush()
    return {"title": title, "entries": entries}


# ---------------------------------------------------------------------------- recorder

@dataclass
class SessionItem:
    msg_id: int
    role: str                          # "user" | "assistant"
    speaker: str                       # "You" or the character's name
    text: str                          # what was said (user: the message; assistant: speech only)
    tags: list[str] = field(default_factory=list)   # "spoken" / "typed", language, voice, "greeting"
    audio: np.ndarray | None = None
    sample_rate: int = SESSION_SAMPLE_RATE
    segments: list[dict[str, str]] | None = None    # assistant: everything shown in the chat


RecordedFn = Callable[[int, str, float, float | None], None]
INDEX_DIR = ".index"   # hidden: offsets and details of every utterance, for replay


class SessionRecorder:
    """Appends utterances to ``<name>.mp3`` and ``<name>.md`` in a background thread.

    The Markdown transcript holds only who said what ("Elara: …"). Offsets, durations
    and actions go to ``.index\\<name>.jsonl`` next to it, which the replay view reads.
    """

    def __init__(self, folder: Path, ffmpeg: Path, on_recorded: RecordedFn | None = None,
                 gap_ms: Callable[[], int] = lambda: 500, bitrate_kbps: Callable[[], int] = lambda: 64) -> None:
        self.folder = folder
        self.ffmpeg = ffmpeg
        self.on_recorded = on_recorded
        self.gap_ms = gap_ms
        self.bitrate_kbps = bitrate_kbps
        self.name: str | None = None
        self.duration = 0.0            # seconds of audio in the MP3 so far
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, name="session-recorder", daemon=True)
        self._thread.start()

    # -- naming ---------------------------------------------------------------------
    @staticmethod
    def valid_name(name: str) -> bool:
        return bool(_NAME_RE.match(name))

    def new_name(self, character_id: str, now: datetime | None = None) -> str:
        stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M")
        base = f"{stamp}_{character_id}"
        name, n = base, 2
        while (self.folder / f"{name}.md").exists() or (self.folder / f"{name}.mp3").exists():
            name, n = f"{base}_{n}", n + 1
        return name

    def files(self) -> dict[str, str] | None:
        if self.name is None:
            return None
        return {"name": self.name, "mp3": f"{self.name}.mp3", "md": f"{self.name}.md"}

    # -- lifecycle (called from the engine thread; the work happens in the recorder thread) --
    def start(self, name: str, title: str) -> None:
        self.name = name
        self._queue.put(("start", (name, title)))

    def resume(self, name: str) -> None:
        self.name = name
        self._queue.put(("resume", name))

    def stop(self) -> None:
        if self.name is not None:
            self.name = None
            self._queue.put(("stop", None))

    def add(self, item: SessionItem) -> None:
        if self.name is not None:
            self._queue.put(("item", item))

    def truncate(self, name: str, drop: int) -> None:
        """Remove the newest ``drop`` utterances of session ``name`` (MP3, transcript and index)."""
        if drop > 0 and self.valid_name(name):
            self._queue.put(("truncate", (name, drop)))

    def wait_idle(self, timeout: float = 30.0) -> None:
        """Block until everything queued so far is written (tests, shutdown)."""
        done = threading.Event()
        self._queue.put(("sync", done))
        done.wait(timeout)

    def close(self) -> None:
        self._queue.put(("quit", None))
        self._thread.join(timeout=30)

    # -- worker ---------------------------------------------------------------------
    def _worker(self) -> None:
        mp3: Path | None = None
        md: Path | None = None
        while True:
            cmd, arg = self._queue.get()
            try:
                if cmd == "quit":
                    return
                if cmd == "sync":
                    arg.set()
                elif cmd == "start":
                    name, title = arg
                    self.folder.mkdir(parents=True, exist_ok=True)
                    _make_index_dir(self.folder)
                    mp3, md = self.folder / f"{name}.mp3", self.folder / f"{name}.md"
                    self.duration = 0.0
                    _append(md, f"# {title}\n\n")
                    log.info("Recording session %s", name)
                elif cmd == "resume":
                    mp3, md = self.folder / f"{arg}.mp3", self.folder / f"{arg}.md"
                    self.duration = mp3_duration(mp3.read_bytes()) if mp3.exists() else 0.0
                    log.info("Continuing session %s at %.1f s", arg, self.duration)
                elif cmd == "stop":
                    mp3 = md = None
                elif cmd == "item" and mp3 is not None and md is not None:
                    self._write(arg, mp3, md)
                elif cmd == "truncate":
                    name, drop = arg
                    seconds = self._truncate(name, drop)
                    if mp3 is not None and mp3.stem == name:
                        self.duration = seconds  # the next utterance continues where the cut is
            except Exception:  # noqa: BLE001 - recording must never break the conversation
                log.exception("Session recording failed (%s)", cmd)

    def _write(self, item: SessionItem, mp3: Path, md: Path) -> None:
        offset: float = self.duration
        duration: float | None = None
        if item.audio is not None and len(item.audio):
            audio = resample(item.audio, item.sample_rate, SESSION_SAMPLE_RATE)
            audio = normalize_loudness(audio, SESSION_SAMPLE_RATE)
            gap = int(SESSION_SAMPLE_RATE * self.gap_ms() / 1000) if self.duration > 0 else 0
            if gap:
                audio = np.concatenate([np.zeros(gap, dtype=np.float32), audio])
            data = encode_mp3(audio, self.ffmpeg, self.bitrate_kbps())
            _append(mp3, data)  # audio first: every transcript entry points at complete audio
            offset = self.duration + gap / SESSION_SAMPLE_RATE
            self.duration += mp3_duration(data)
            duration = self.duration - offset
        speaker = item.speaker.replace("\n", " ").strip() or "?"
        entry: dict[str, Any] = {"role": item.role, "speaker": speaker, "offset": round(offset, 3),
                                 "duration": None if duration is None else round(duration, 3), "tags": item.tags}
        if item.role == "user":
            entry["text"] = item.text
        else:
            entry["segments"] = visible_segments(item.segments or [{"kind": "speech", "text": item.text}])
        _append(index_path(self.folder, mp3.stem), json.dumps(entry, ensure_ascii=False) + "\n")
        if item.text.strip():  # the transcript reads like a screenplay: "Elara: …"
            _append(md, f"{speaker}: {item.text.strip()}\n\n")
        if self.on_recorded is not None:
            self.on_recorded(item.msg_id, mp3.stem, offset, duration)


    def _truncate(self, name: str, drop: int) -> float:
        """Cut the last ``drop`` index entries; returns the new MP3 length in seconds."""
        index = index_path(self.folder, name)
        md = self.folder / f"{name}.md"
        mp3 = self.folder / f"{name}.mp3"
        entries = []
        if index.exists():
            for line in index.read_text(encoding="utf-8").splitlines():
                try:
                    entries.append(json.loads(line))
                except ValueError:
                    continue  # a line cut off by a crash
        kept = entries[:max(0, len(entries) - drop)]
        end = max((e["offset"] + e["duration"] for e in kept if e.get("duration")), default=0.0)
        seconds = 0.0
        if mp3.exists():
            data = mp3.read_bytes()
            cut = mp3_cut_position(data, end)
            with mp3.open("r+b") as f:
                f.truncate(cut)
            seconds = mp3_duration(data[:cut])
        index.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in kept), encoding="utf-8",
                         newline="")
        if md.exists():
            first = md.read_text(encoding="utf-8").split("\n", 1)[0]
            lines = [first if first.startswith("# ") else f"# {name}", ""]
            for e in kept:
                text = e.get("text", "") if e.get("role") == "user" else spoken_text(e.get("segments") or [])
                if text.strip():
                    lines += [f"{e.get('speaker', '?')}: {text.strip()}", ""]
            md.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
        log.info("Removed the last %d utterances of session %s", min(drop, len(entries)), name)
        return seconds


def _append(path: Path, data: bytes | str) -> None:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    with path.open("ab") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def index_path(folder: Path, name: str) -> Path:
    return folder / INDEX_DIR / f"{name}.jsonl"


def _make_index_dir(folder: Path) -> None:
    path = folder / INDEX_DIR
    if path.is_dir():
        return
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":  # keep the folder out of sight in Explorer
        import ctypes

        ctypes.windll.kernel32.SetFileAttributesW(str(path), 0x02)  # FILE_ATTRIBUTE_HIDDEN


# ---------------------------------------------------------------------------- reading

def load_session(folder: Path, name: str) -> dict[str, Any]:
    """Title and utterances (with offsets) of a recorded session, for the replay view."""
    md = folder / f"{name}.md"
    text = md.read_text(encoding="utf-8") if md.exists() else ""
    index = index_path(folder, name)
    if not index.exists():
        return parse_transcript(text)  # recorded before the screenplay format
    title = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), name)
    entries = []
    for line in index.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue  # a line cut off by a crash
        tags = entry.get("tags") or []
        if entry.get("role") == "user":
            entry["source"] = "voice" if "spoken" in tags else "text"
        else:
            entry["greeting"] = "greeting" in tags
        entries.append(entry)
    return {"title": title, "entries": entries}


# ---------------------------------------------------------------------------- listing

def list_sessions(folder: Path, limit: int = 30) -> list[dict[str, Any]]:
    """Recorded sessions, most recently changed first."""
    if not folder.is_dir():
        return []
    for index in (folder / INDEX_DIR).glob("*.jsonl"):  # sessions deleted in Explorer
        if not (folder / f"{index.stem}.md").exists() and not (folder / f"{index.stem}.mp3").exists():
            index.unlink(missing_ok=True)
    items = []
    for md in folder.glob("*.md"):
        if not SessionRecorder.valid_name(md.stem):
            continue
        try:
            stat = md.stat()
            with md.open(encoding="utf-8") as f:
                first = f.readline().strip()
        except OSError:
            continue
        mp3 = md.with_suffix(".mp3")
        items.append({
            "name": md.stem,
            "title": first[2:].strip() if first.startswith("# ") else md.stem,
            "character": re.sub(r"_\d+$", "", md.stem.split("_", 2)[2]) if md.stem.count("_") >= 2 else "",
            "updated": stat.st_mtime,
            "duration": estimate_duration(mp3) if mp3.exists() else 0.0,
        })
    items.sort(key=lambda s: s["updated"], reverse=True)
    return items[:limit]


def session_title(character_name: str, started: float | None = None) -> str:
    when = datetime.fromtimestamp(started or time.time())
    return f"{character_name} — {when:%d %b %Y, %H:%M}"
