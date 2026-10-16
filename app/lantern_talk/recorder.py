"""Session recording (spec §12): one MP3 of the whole dialogue plus a Markdown transcript.

Each utterance is encoded on its own (in a background thread) into a file of its own in
the session's folder, so single messages can be deleted or edited. When the session ends,
the files are put together into one MP3 of the whole dialogue; waiting time is not
recorded, a short fixed gap separates the turns. The Markdown transcript reads like a
screenplay ("Elara: …"); details of every utterance are kept in a hidden index file.

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
import shutil
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


RecordedFn = Callable[[int, str, int, str | None, float | None], None]  # msg id, session, seq, file, seconds
INDEX_DIR = ".index"   # hidden: details of every utterance, for replay
_PART_RE = re.compile(r"^\d{4}[\w\- ]{0,64}\.mp3$")


def part_name(seq: int, speaker: str) -> str:
    """File name of one utterance in a session's folder: "0003 Elara.mp3"."""
    label = re.sub(r"[^\w\- ]+", "", speaker).strip()[:40]
    return f"{seq:04d} {label}.mp3" if label else f"{seq:04d}.mp3"


def valid_part(name: str) -> bool:
    return bool(_PART_RE.match(name)) and Path(name).name == name


def parts_dir(folder: Path, name: str) -> Path:
    return folder / name


class SessionRecorder:
    """Records a chat in a background thread: every utterance as its own MP3 first.

    While a chat is open, each utterance is kept as ``<name>\\0001 You.mp3`` and so on, so a
    message can be deleted or replaced on its own. The MP3 of the whole dialogue
    (``<name>.mp3``) is put together from them when the session ends (End chat, another
    character, closing the app); End chat also removes the single files. The Markdown
    transcript (``<name>.md``) holds only who said what ("Elara: …") and is always current;
    details of every utterance go to ``.index\\<name>.jsonl``.
    """

    def __init__(self, folder: Path, ffmpeg: Path, on_recorded: RecordedFn | None = None,
                 gap_ms: Callable[[], int] = lambda: 500, bitrate_kbps: Callable[[], int] = lambda: 64,
                 on_built: Callable[[str], None] | None = None) -> None:
        self.folder = folder
        self.ffmpeg = ffmpeg
        self.on_recorded = on_recorded
        self.on_built = on_built        # the MP3 of a whole session is ready
        self.gap_ms = gap_ms
        self.bitrate_kbps = bitrate_kbps
        self.name: str | None = None
        self._current: str | None = None          # the session the worker records into
        self._entries: list[dict[str, Any]] = []   # of the session being recorded
        self._title = ""
        self._silence: dict[tuple[int, int], bytes] = {}
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
        while any(p.exists() for p in (self.folder / f"{name}.md", self.folder / f"{name}.mp3",
                                       parts_dir(self.folder, name))):
            name, n = f"{base}_{n}", n + 1
        return name

    def files(self) -> dict[str, str] | None:
        if self.name is None:
            return None
        return {"name": self.name, "mp3": f"{self.name}.mp3", "md": f"{self.name}.md", "parts": f"{self.name}\\"}

    # -- lifecycle (called from the engine thread; the work happens in the recorder thread) --
    def start(self, name: str, title: str) -> None:
        self.name = name
        self._queue.put(("start", (name, title)))

    def resume(self, name: str) -> None:
        self.name = name
        self._queue.put(("resume", name))

    def stop(self) -> None:
        """The session pauses (another character, closing the app): its MP3 is put together."""
        if self.name is not None:
            self.name = None
            self._queue.put(("stop", None))

    def finish(self, name: str) -> None:
        """End chat: the whole MP3 is put together and the single files are removed."""
        if self.name == name:
            self.name = None
        if self.valid_name(name):
            self._queue.put(("finish", name))

    def add(self, item: SessionItem) -> None:
        if self.name is not None:
            self._queue.put(("item", item))

    def truncate(self, name: str, drop: int) -> None:
        """Remove the newest ``drop`` utterances of session ``name``."""
        if drop > 0 and self.valid_name(name):
            self._queue.put(("truncate", (name, drop)))

    def replace(self, name: str, seq: int, item: SessionItem, keep_audio: bool = False) -> None:
        """Replace utterance ``seq`` (an edited message): new text, and new audio unless ``keep_audio``."""
        if self.valid_name(name):
            self._queue.put(("replace", (name, int(seq), item, keep_audio)))

    def wait_idle(self, timeout: float = 60.0) -> None:
        """Block until everything queued so far is written (tests, shutdown)."""
        done = threading.Event()
        self._queue.put(("sync", done))
        done.wait(timeout)

    def close(self) -> None:
        self.stop()
        self._queue.put(("quit", None))
        self._thread.join(timeout=120)

    def finish_pending(self, keep: set[str]) -> None:
        """At start: sessions whose single files are still there (the app was closed or crashed during a
        chat) get their whole MP3. Sessions in ``keep`` are chats that can still go on."""
        if not self.folder.is_dir():
            return
        for index in (self.folder / INDEX_DIR).glob("*.jsonl"):
            name = index.stem
            if self.valid_name(name) and parts_dir(self.folder, name).is_dir():
                mp3 = self.folder / f"{name}.mp3"
                if name not in keep:
                    self._queue.put(("finish", name))
                elif not mp3.exists():
                    self._queue.put(("build", name))

    # -- worker ---------------------------------------------------------------------
    def _worker(self) -> None:
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
                    self._current, self._title, self._entries = name, title, []
                    self._save(name, title, [])
                    log.info("Recording session %s", name)
                elif cmd == "resume":
                    self._current = arg
                    self._title, self._entries = self._load(arg)
                    log.info("Continuing session %s (%d utterances)", arg, len(self._entries))
                elif cmd == "stop":
                    name = self._current
                    self._current = None
                    if name:
                        self._build(name)
                elif cmd == "build":
                    self._build(arg)
                elif cmd == "finish":
                    self._build(arg, notify=False)
                    if self._current == arg:
                        self._current = None
                    shutil.rmtree(parts_dir(self.folder, arg), ignore_errors=True)
                    if self.on_built is not None:
                        self.on_built(arg)
                elif cmd == "item" and self._current:
                    self._write(arg)
                elif cmd == "truncate":
                    self._truncate(*arg)
                elif cmd == "replace":
                    self._replace(*arg)
            except Exception:  # noqa: BLE001 - recording must never break the conversation
                log.exception("Session recording failed (%s)", cmd)

    def _entries_of(self, name: str) -> tuple[str, list[dict[str, Any]]]:
        if name == self._current:
            return self._title, self._entries
        return self._load(name)

    def _load(self, name: str) -> tuple[str, list[dict[str, Any]]]:
        """Title and utterances of a session; recordings from before the single files are split up once."""
        md = self.folder / f"{name}.md"
        first = md.read_text(encoding="utf-8").split("\n", 1)[0] if md.exists() else ""
        title = first[2:].strip() if first.startswith("# ") else name
        entries = []
        index = index_path(self.folder, name)
        if index.exists():
            for line in index.read_text(encoding="utf-8").splitlines():
                try:
                    entries.append(json.loads(line))
                except ValueError:
                    continue  # a line cut off by a crash
        if any("seq" not in e for e in entries):
            entries = self._split_old(name, entries)
            self._save(name, title, entries)
        return title, entries

    def _split_old(self, name: str, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """A session recorded into one MP3 (before 17 Oct 2026): cut it into single files at the offsets."""
        mp3 = self.folder / f"{name}.mp3"
        data = mp3.read_bytes() if mp3.exists() else b""
        folder = parts_dir(self.folder, name)
        for seq, e in enumerate(entries, 1):
            e["seq"], e["part"] = seq, None
            if data and e.get("duration"):
                a = mp3_cut_position(data, e["offset"])
                b = mp3_cut_position(data, e["offset"] + e["duration"])
                if b > a:
                    folder.mkdir(parents=True, exist_ok=True)
                    e["part"] = part_name(seq, e.get("speaker", ""))
                    (folder / e["part"]).write_bytes(data[a:b])
                    e["duration"] = mp3_duration(data[a:b])
        return entries

    def _save(self, name: str, title: str, entries: list[dict[str, Any]]) -> None:
        """Index and transcript, rewritten as a whole (they are small)."""
        index_path(self.folder, name).write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8", newline="")
        lines = [f"# {title}", ""]
        for e in entries:
            text = e.get("text", "") if e.get("role") == "user" else spoken_text(e.get("segments") or [])
            if text.strip():  # the transcript reads like a screenplay: "Elara: …"
                lines += [f"{e.get('speaker', '?')}: {text.strip()}", ""]
        _write_atomic(self.folder / f"{name}.md", "\n".join(lines) + "\n")

    def _encode(self, item: SessionItem) -> bytes | None:
        if item.audio is None or not len(item.audio):
            return None
        audio = resample(item.audio, item.sample_rate, SESSION_SAMPLE_RATE)
        audio = normalize_loudness(audio, SESSION_SAMPLE_RATE)
        return encode_mp3(audio, self.ffmpeg, self.bitrate_kbps())

    @staticmethod
    def _entry(item: SessionItem) -> dict[str, Any]:
        entry: dict[str, Any] = {"role": item.role, "speaker": item.speaker.replace("\n", " ").strip() or "?",
                                 "tags": item.tags}
        if item.role == "user":
            entry["text"] = item.text
        else:
            entry["segments"] = visible_segments(item.segments or [{"kind": "speech", "text": item.text}])
        return entry

    def _write(self, item: SessionItem) -> None:
        name = self._current
        seq = max((e["seq"] for e in self._entries), default=0) + 1
        entry = {"seq": seq, **self._entry(item), "part": None, "duration": None}
        data = self._encode(item)
        if data is not None:
            folder = parts_dir(self.folder, name)
            folder.mkdir(parents=True, exist_ok=True)
            entry["part"] = part_name(seq, entry["speaker"])
            _write_atomic(folder / entry["part"], data)
            entry["duration"] = round(mp3_duration(data), 3)
        self._entries.append(entry)
        self._save(name, self._title, self._entries)
        if self.on_recorded is not None:
            self.on_recorded(item.msg_id, name, seq, entry["part"], entry["duration"])

    def _truncate(self, name: str, drop: int) -> None:
        title, entries = self._entries_of(name)
        cut = max(0, len(entries) - drop)
        for e in entries[cut:]:
            if e.get("part"):
                (parts_dir(self.folder, name) / e["part"]).unlink(missing_ok=True)
        del entries[cut:]
        self._save(name, title, entries)
        self._stale(name)
        log.info("Removed the last %d utterances of session %s", drop, name)

    def _replace(self, name: str, seq: int, item: SessionItem, keep_audio: bool) -> None:
        title, entries = self._entries_of(name)
        e = next((x for x in entries if x.get("seq") == seq), None)
        if e is None:
            log.warning("Utterance %d of session %s not found", seq, name)
            return
        new = self._entry(item)
        new["tags"] = sorted({*e.get("tags", []), "edited"}, key=lambda t: (t != "edited", t))
        e.update(new)
        if not keep_audio:
            data = self._encode(item)
            folder = parts_dir(self.folder, name)
            if e.get("part"):
                (folder / e["part"]).unlink(missing_ok=True)
            e["part"], e["duration"] = None, None
            if data is not None:
                folder.mkdir(parents=True, exist_ok=True)
                e["part"] = part_name(e["seq"], e["speaker"])
                _write_atomic(folder / e["part"], data)
                e["duration"] = round(mp3_duration(data), 3)
        self._save(name, title, entries)
        self._stale(name)
        if self.on_recorded is not None:
            self.on_recorded(item.msg_id, name, seq, e["part"], e["duration"])

    def _stale(self, name: str) -> None:
        """The whole MP3 no longer matches the single files; it is put together again at the end."""
        (self.folder / f"{name}.mp3").unlink(missing_ok=True)

    def _gap(self) -> bytes:
        key = (self.gap_ms(), self.bitrate_kbps())
        if key not in self._silence:
            samples = int(SESSION_SAMPLE_RATE * key[0] / 1000)
            self._silence[key] = encode_mp3(np.zeros(samples, np.float32), self.ffmpeg, key[1]) if samples else b""
        return self._silence[key]

    def _build(self, name: str, notify: bool = True) -> None:
        """Put the whole MP3 together from the single files, with a short pause between turns."""
        title, entries = self._entries_of(name)
        folder = parts_dir(self.folder, name)
        if not folder.is_dir():
            if notify and self.on_built is not None:
                self.on_built(name)  # nothing spoken, or already put together
            return
        mp3 = self.folder / f"{name}.mp3"
        tmp = mp3.with_suffix(".mp3.tmp")
        position = 0.0
        gap = self._gap()
        gap_s = mp3_duration(gap) if gap else 0.0
        with tmp.open("wb") as out:
            for e in entries:
                path = folder / e["part"] if e.get("part") else None
                if path is None or not path.exists():
                    e["offset"] = round(position, 3) if e.get("part") is None else None
                    continue
                data = path.read_bytes()
                if position > 0 and gap:
                    out.write(gap)
                    position += gap_s
                out.write(data)
                e["offset"] = round(position, 3)
                e["duration"] = round(mp3_duration(data), 3)
                position += e["duration"]
        if position > 0:
            os.replace(tmp, mp3)
        else:
            tmp.unlink(missing_ok=True)
        self._save(name, title, entries)
        log.info("Session %s: %.1f s of audio put together", name, position)
        if notify and self.on_built is not None:
            self.on_built(name)


def _write_atomic(path: Path, data: bytes | str) -> None:
    """Write via a temporary file. Windows refuses the swap while someone reads the file (the session
    list, a replay): then try again shortly, and finally write in place."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(raw)
    for _ in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)
    tmp.unlink(missing_ok=True)
    path.write_bytes(raw)


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
    # While the chat is open, every utterance is played from its own file.
    return {"title": title, "entries": entries, "parts": parts_dir(folder, name).is_dir()}


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
        duration = estimate_duration(mp3) if mp3.exists() else _parts_duration(folder, md.stem)
        items.append({
            "name": md.stem,
            "title": first[2:].strip() if first.startswith("# ") else md.stem,
            "character": re.sub(r"_\d+$", "", md.stem.split("_", 2)[2]) if md.stem.count("_") >= 2 else "",
            "updated": stat.st_mtime,
            "duration": duration,
        })
    items.sort(key=lambda s: s["updated"], reverse=True)
    return items[:limit]


def _parts_duration(folder: Path, name: str) -> float:
    """Length of a session that is not put together yet (sum of its single files)."""
    index = index_path(folder, name)
    total = 0.0
    if index.exists():
        for line in index.read_text(encoding="utf-8").splitlines():
            try:
                total += float(json.loads(line).get("duration") or 0)
            except (ValueError, AttributeError):
                continue
    return total


def session_title(character_name: str, started: float | None = None) -> str:
    when = datetime.fromtimestamp(started or time.time())
    return f"{character_name} — {when:%d %b %Y, %H:%M}"
