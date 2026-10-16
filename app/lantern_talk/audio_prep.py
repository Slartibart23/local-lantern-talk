"""Reference clip preparation with ffmpeg and a quick quality check (spec §7.1)."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

PREP_SAMPLE_RATE = 24_000
# Formats ffmpeg reads; messenger voice notes are often .opus / .ogg, iPhone memos .m4a.
VOICE_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".webm")
MIN_S = 3.0
RECOMMENDED_S = (10.0, 13.0)
TRIM_ABOVE_S = 20.0
NOISY_BELOW_DB = 15.0
CLIPPING_FRACTION = 0.001

# Trim silence at both ends (silenceremove only trims the start, hence the double reverse)
# and normalize loudness so quiet and loud clips sound alike.
_TRIM = "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.15"
FILTER = f"{_TRIM},areverse,{_TRIM},areverse,loudnorm=I=-20:TP=-2:LRA=11"


class PrepError(Exception):
    """A clip that cannot be used, with a message meant for the user."""


def _run(cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(cmd, capture_output=True, timeout=120,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def decode(src: Path, ffmpeg: Path, sample_rate: int = PREP_SAMPLE_RATE, filters: str | None = None) -> np.ndarray:
    """Any audio file -> mono float32 at ``sample_rate`` (optionally through a filter chain)."""
    cmd = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1",
           "-ar", str(sample_rate)]
    if filters:
        cmd += ["-af", filters]
    cmd += ["-f", "f32le", "-"]
    try:
        out = _run(cmd)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PrepError(f"ffmpeg could not be started ({exc}).") from exc
    if out.returncode != 0 or not out.stdout:
        raise PrepError(f"{src.name} could not be read as audio.")
    return np.frombuffer(out.stdout, dtype=np.float32).copy()  # -ar applies after the filters


def prepare(src: Path, dst: Path, ffmpeg: Path) -> np.ndarray:
    """Trimmed, loudness-normalized 24 kHz mono version of ``src``; also written to ``dst`` (WAV)."""
    import soundfile as sf

    audio = decode(src, ffmpeg, PREP_SAMPLE_RATE, FILTER)
    if len(audio) < PREP_SAMPLE_RATE // 2:
        raise PrepError(f"{src.name} contains (almost) no sound.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.wav")
    sf.write(str(tmp), audio, PREP_SAMPLE_RATE, subtype="FLOAT")
    tmp.replace(dst)
    return audio


@dataclass
class ClipReport:
    duration_s: float
    snr_db: float
    clipped: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze(original: np.ndarray, prepared: np.ndarray, sample_rate: int = PREP_SAMPLE_RATE) -> ClipReport:
    """Best-effort quality check: length, background noise (SNR) and clipping."""
    duration = len(prepared) / sample_rate
    frame = sample_rate // 20  # 50 ms
    n = len(prepared) // frame
    if n >= 4:
        rms = np.sqrt(np.mean(prepared[: n * frame].reshape(n, frame) ** 2, axis=1)) + 1e-9
        snr = float(20 * np.log10(np.percentile(rms, 90) / np.percentile(rms, 10)))
    else:
        snr = 0.0
    clipped = bool(np.mean(np.abs(original) >= 0.99) > CLIPPING_FRACTION) if len(original) else False

    warnings = []
    low, high = RECOMMENDED_S
    if duration < MIN_S:
        warnings.append(f"Only {duration:.1f} s of sound; too little to copy the voice. "
                        f"{low:.0f}-{high:.0f} s of clean speech clone best.")
    elif duration < low:
        warnings.append(f"{duration:.1f} s long; works, but {low:.0f}-{high:.0f} s usually clone best.")
    elif duration > TRIM_ABOVE_S:
        warnings.append(f"{duration:.0f} s long; without your own reference text only the first 15 s are used. "
                        f"{low:.0f}-{high:.0f} s work best.")
    elif duration > high:
        warnings.append(f"{duration:.0f} s long; {low:.0f}-{high:.0f} s usually clone best.")
    if snr < NOISY_BELOW_DB:
        warnings.append("Noticeable background noise or reverb; the clone will copy it. "
                        "A recording in a quiet room works better.")
    if clipped:
        warnings.append("The recording is distorted (too loud / clipping); the clone may sound harsh.")
    return ClipReport(duration_s=round(duration, 1), snr_db=round(snr, 1), clipped=clipped, warnings=warnings)
