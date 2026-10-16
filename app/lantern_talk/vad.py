"""Voice activity detection with Silero VAD and utterance segmentation."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512  # Silero VAD v5+ expects exactly 512 samples at 16 kHz (32 ms)
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE


class SileroVAD:
    """Loads the TorchScript Silero model from ``models\\vad`` and scores 32 ms frames."""

    def __init__(self, model_file: Path) -> None:
        self.model_file = model_file
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_file.exists():
            raise FileNotFoundError(f"Silero VAD model not found: {self.model_file}. Run install.bat.")
        import torch

        torch.set_num_threads(1)
        self._model = torch.jit.load(str(self.model_file), map_location="cpu")
        self._model.eval()
        log.info("Silero VAD loaded")

    def unload(self) -> None:
        self._model = None

    def reset(self) -> None:
        if self._model is not None:
            self._model.reset_states()

    def speech_probability(self, frame: np.ndarray) -> float:
        import torch

        self.load()
        assert self._model is not None
        with torch.no_grad():
            return float(self._model(torch.from_numpy(frame.astype(np.float32)), SAMPLE_RATE).item())


@dataclass
class SegmenterConfig:
    threshold: float = 0.5           # speech probability to start an utterance
    neg_threshold: float = 0.35      # probability below which a frame counts as silence
    end_silence_ms: int = 700        # end of utterance after this much silence (spec §9)
    min_speech_ms: int = 250         # shorter bursts are ignored (clicks, coughs)
    pre_roll_ms: int = 300           # audio kept from before speech start
    max_utterance_s: float = 30.0    # force an end for very long monologues
    smart_check_ms: int = 300        # Smart mode: pause after which the turn model is asked
    smart_max_silence_ms: int = 3000  # Smart mode: a pause this long ends the turn anyway


@dataclass
class Utterance:
    audio: np.ndarray                # 16 kHz mono float32
    duration_s: float
    start: int = 0                   # position in the 16 kHz stream since the last reset
    end: int = 0
    raw: np.ndarray | None = None    # the same audio at the capture rate (for session recording)
    raw_sr: int = 0
    collected: bool = False          # push-to-talk / "Hold on": may contain long pauses


@dataclass
class UtteranceSegmenter:
    """Turns a stream of 16 kHz frames into complete utterances."""

    vad: SileroVAD
    config: SegmenterConfig = field(default_factory=SegmenterConfig)
    # Smart mode: called with the utterance so far when the speaker pauses; True = they are done.
    turn_check: Callable[[np.ndarray], bool] | None = None

    def __post_init__(self) -> None:
        self._asked = False  # the turn model was asked during the current pause
        self._pending = np.zeros(0, dtype=np.float32)
        self._pre_roll: deque[np.ndarray] = deque(maxlen=max(1, self.config.pre_roll_ms // FRAME_MS))
        self._speech: list[np.ndarray] = []
        self._in_speech = False
        self._silence_frames = 0
        self._speech_frames = 0
        self.level = 0.0  # last frame RMS, for a level meter
        self.probability = 0.0
        self._pos = 0     # 16 kHz samples processed since the last reset

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def skip(self, samples: np.ndarray) -> None:
        """Count audio that is not listened to (push-to-talk), so positions stay in step with the capture."""
        self._pos += len(samples)

    @property
    def position(self) -> int:
        """16 kHz samples received since the last reset (processed and pending)."""
        return self._pos + len(self._pending)

    def take_speech(self) -> np.ndarray:
        """The utterance in progress (ending at :attr:`position`), and forget it; positions keep counting."""
        audio = np.concatenate(self._speech) if self._in_speech and self._speech else np.zeros(0, np.float32)
        pos, pending = self._pos, self._pending
        self.reset()
        self._pos, self._pending = pos, pending
        return np.concatenate([audio, pending])

    def reset(self) -> None:
        self._asked = False
        self._pending = np.zeros(0, dtype=np.float32)
        self._pre_roll.clear()
        self._speech = []
        self._in_speech = False
        self._silence_frames = 0
        self._speech_frames = 0
        self._pos = 0
        self.vad.reset()

    def feed(self, samples: np.ndarray) -> list[Utterance]:
        """Feed 16 kHz mono float32 samples; returns utterances that just ended."""
        self._pending = np.concatenate([self._pending, samples.astype(np.float32)])
        out: list[Utterance] = []
        while len(self._pending) >= FRAME_SAMPLES:
            frame, self._pending = self._pending[:FRAME_SAMPLES], self._pending[FRAME_SAMPLES:]
            utt = self._process(frame)
            self._pos += FRAME_SAMPLES
            if utt is not None:
                out.append(utt)
        return out

    def _process(self, frame: np.ndarray) -> Utterance | None:
        cfg = self.config
        self.level = float(np.sqrt(np.mean(frame**2)))
        p = self.vad.speech_probability(frame)
        self.probability = p
        if not self._in_speech:
            self._pre_roll.append(frame)
            if p >= cfg.threshold:
                self._in_speech = True
                self._speech = list(self._pre_roll)
                self._pre_roll.clear()
                self._silence_frames = 0
                self._speech_frames = 1
            return None

        self._speech.append(frame)
        if p < cfg.neg_threshold:
            self._silence_frames += 1
        else:
            self._silence_frames = 0
            self._speech_frames += 1
            self._asked = False

        total_s = len(self._speech) * FRAME_MS / 1000
        silence_ms = self._silence_frames * FRAME_MS
        if self.turn_check is None:
            ended = silence_ms >= cfg.end_silence_ms
        else:
            # Smart: a short pause ends the turn if the speaker sounds finished; while they
            # sound unfinished ("and then ...", "uhm"), only a long pause does.
            ended = silence_ms >= cfg.smart_max_silence_ms
            if not ended and not self._asked and silence_ms >= cfg.smart_check_ms:
                self._asked = True
                try:
                    ended = bool(self.turn_check(np.concatenate(self._speech)))
                except Exception:  # noqa: BLE001 - fall back to the plain pause rule
                    log.exception("Smart turn check failed")
                    ended = silence_ms >= cfg.end_silence_ms
        ended = ended or total_s >= cfg.max_utterance_s
        if not ended:
            return None

        speech_ms = self._speech_frames * FRAME_MS
        audio = np.concatenate(self._speech)
        # Keep a little trailing silence, drop the rest.
        keep_tail = max(0, self._silence_frames - 6) * FRAME_SAMPLES
        if keep_tail:
            audio = audio[:-keep_tail]
        self._speech = []
        self._in_speech = False
        self._silence_frames = 0
        self._speech_frames = 0
        self._asked = False
        self.vad.reset()
        if speech_ms < cfg.min_speech_ms:
            return None
        end = self._pos + FRAME_SAMPLES - keep_tail
        return Utterance(audio=audio, duration_s=len(audio) / SAMPLE_RATE, start=end - len(audio), end=end)
