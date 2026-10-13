"""Speech-to-text with faster-whisper (automatic language detection)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio_io import resample
from .gpu import free_cuda_cache, register_cuda_dlls

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str
    language_probability: float
    seconds: float  # processing time


class WhisperSTT:
    """Thin wrapper around :class:`faster_whisper.WhisperModel` with load / unload."""

    def __init__(self, model_dir: Path, device: str = "cuda", compute_type: str = "float16") -> None:
        self.model_dir = model_dir
        self.device = device
        self.compute_type = compute_type
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        if not (self.model_dir / "model.bin").exists():
            raise FileNotFoundError(f"Whisper model not found in {self.model_dir}. Run install.bat.")
        register_cuda_dlls()
        from faster_whisper import WhisperModel

        t0 = time.perf_counter()
        self._model = WhisperModel(str(self.model_dir), device=self.device, compute_type=self.compute_type)
        # The first call initializes cuBLAS / cuDNN (several seconds); do it now, not mid-conversation.
        segments, _ = self._model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language="en", beam_size=1)
        list(segments)  # segments are generated lazily
        log.info("Whisper loaded from %s in %.1fs", self.model_dir.name, time.perf_counter() - t0)

    def unload(self) -> None:
        self._model = None
        free_cuda_cache()

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> Transcript:
        """Transcribe mono float32 audio at 16 kHz."""
        self.load()
        assert self._model is not None
        t0 = time.perf_counter()
        segments, info = self._model.transcribe(
            audio.astype(np.float32),
            language=language,
            beam_size=1,
            vad_filter=False,  # Silero VAD already segmented the utterance
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return Transcript(
            text=text,
            language=info.language,
            language_probability=float(info.language_probability),
            seconds=time.perf_counter() - t0,
        )

    def transcribe_at(self, audio: np.ndarray, sample_rate: int, language: str | None = None) -> Transcript:
        """Transcribe mono float32 audio at any sample rate."""
        return self.transcribe(resample(audio, sample_rate, SAMPLE_RATE), language)
