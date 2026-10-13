"""Text-to-speech with OmniVoice, including voice prompt caching.

Every voice — built-in (voice design) or custom (reference clip) — is turned into
an OmniVoice ``VoiceClonePrompt`` once and cached on disk. Synthesis always uses
the cached prompt, so a voice sounds the same in every sentence.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .audio_io import load_audio
from .gpu import free_cuda_cache
from .textmatch import word_similarity

log = logging.getLogger(__name__)

OUTPUT_SAMPLE_RATE = 24_000
REF_SAMPLE_RATE = 24_000
DESIGN_ATTEMPTS = 5
DESIGN_MIN_SIMILARITY = 0.75


@dataclass(frozen=True)
class BuiltinVoice:
    id: str
    name: str
    language: str
    gender: str
    instruct: str
    reference_text: str


def load_builtin_voices(config_file: Path) -> dict[str, BuiltinVoice]:
    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    return {v["id"]: BuiltinVoice(**v) for v in data["voices"]}


def match_builtin_voice(voice: str, language: str | None, builtins: dict[str, BuiltinVoice]) -> str:
    """Spec §7.3: swap a built-in voice for the same-gender built-in voice of ``language``.

    A cloned voice keeps the accent of its reference clip, so an English voice
    speaking German sounds English. Custom voices, unknown languages and
    languages without a built-in voice are left unchanged.
    """
    if not language or not voice.startswith("builtin:"):
        return voice
    current = builtins.get(voice.removeprefix("builtin:"))
    if current is None or current.language == language:
        return voice
    for candidate in builtins.values():
        if candidate.language == language and candidate.gender == current.gender:
            return f"builtin:{candidate.id}"
    return voice


@dataclass
class SynthResult:
    audio: np.ndarray  # mono float32 at OUTPUT_SAMPLE_RATE
    seconds: float     # processing time

    @property
    def duration_s(self) -> float:
        return len(self.audio) / OUTPUT_SAMPLE_RATE


class OmniVoiceTTS:
    """Loads OmniVoice in-process; can be unloaded to free VRAM."""

    def __init__(self, model_dir: Path, device: str = "cuda:0", num_step: int = 32) -> None:
        self.model_dir = model_dir
        self.device = device
        self.num_step = num_step
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        if not (self.model_dir / "model.safetensors").exists():
            raise FileNotFoundError(f"OmniVoice model not found in {self.model_dir}. Run install.bat.")
        import torch
        from omnivoice import OmniVoice

        t0 = time.perf_counter()
        self._model = OmniVoice.from_pretrained(str(self.model_dir), device_map=self.device, dtype=torch.float16)
        log.info("OmniVoice loaded in %.1fs", time.perf_counter() - t0)

    def unload(self) -> None:
        self._model = None
        free_cuda_cache()

    @property
    def model(self):  # noqa: ANN201 - omnivoice type is optional at import time
        self.load()
        return self._model

    # -- synthesis ---------------------------------------------------------------
    def synthesize(self, text: str, prompt: object, language: str | None = None) -> SynthResult:
        import torch

        t0 = time.perf_counter()
        with torch.inference_mode():
            audio = self.model.generate(
                text=text,
                voice_clone_prompt=prompt,
                language=language,
                num_step=self.num_step,
            )[0]
        return SynthResult(audio=np.asarray(audio, dtype=np.float32), seconds=time.perf_counter() - t0)

    def design(self, text: str, instruct: str, language: str | None) -> np.ndarray:
        """Generate audio for a designed (attribute-described) voice."""
        import torch

        with torch.inference_mode():
            audio = self.model.generate(text=text, instruct=instruct, language=language, num_step=32)[0]
        return np.asarray(audio, dtype=np.float32)

    def make_prompt(self, audio: np.ndarray, sample_rate: int, ref_text: str) -> object:
        import torch

        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        return self.model.create_voice_clone_prompt(ref_audio=(wav, sample_rate), ref_text=ref_text)


def _load_prompt(path: Path) -> object:
    from omnivoice import VoiceClonePrompt

    return VoiceClonePrompt.load(str(path))


def file_fingerprint(path: Path) -> str:
    """Cache key that changes whenever the source file changes."""
    st = path.stat()
    h = hashlib.sha1()
    h.update(f"{path.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


class VoiceLibrary:
    """Resolves voice ids to cached OmniVoice prompts.

    * ``builtin:<id>`` — generated with voice design into ``models\\builtin_voices``.
    * a file path     — custom reference clip; cached in ``user_data\\voices\\.cache``.
    """

    def __init__(
        self,
        tts: OmniVoiceTTS,
        builtin_config: Path,
        builtin_dir: Path,
        custom_cache_dir: Path,
        transcribe: Callable[[np.ndarray, int, str | None], str],
    ) -> None:
        self.tts = tts
        self.builtins = load_builtin_voices(builtin_config)
        self.builtin_dir = builtin_dir
        self.custom_cache_dir = custom_cache_dir
        self.transcribe = transcribe
        self._prompts: dict[str, object] = {}

    def prompt_for(self, voice: str) -> object:
        if voice not in self._prompts:
            if voice.startswith("builtin:"):
                self._prompts[voice] = self._builtin_prompt(voice.removeprefix("builtin:"))
            else:
                self._prompts[voice] = self._custom_prompt(Path(voice))
        return self._prompts[voice]

    def matched(self, voice: str, language: str | None) -> str:
        return match_builtin_voice(voice, language, self.builtins)

    def language_of(self, voice: str) -> str | None:
        if voice.startswith("builtin:"):
            b = self.builtins.get(voice.removeprefix("builtin:"))
            return b.language if b else None
        return None

    # -- built-in ------------------------------------------------------------------
    def _builtin_prompt(self, voice_id: str, regenerate: bool = False) -> object:
        import soundfile as sf

        spec = self.builtins.get(voice_id)
        if spec is None:
            raise KeyError(f"Unknown built-in voice '{voice_id}'. See config/builtin_voices.yaml.")
        self.builtin_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = self.builtin_dir / f"{voice_id}.pt"
        wav_file = self.builtin_dir / f"{voice_id}.wav"
        if prompt_file.exists() and not regenerate:
            return _load_prompt(prompt_file)
        audio = self._generate_verified(spec)
        sf.write(wav_file, audio, OUTPUT_SAMPLE_RATE)
        prompt = self.tts.make_prompt(audio, OUTPUT_SAMPLE_RATE, spec.reference_text)
        prompt.save(str(prompt_file))  # type: ignore[attr-defined]
        return prompt

    def _generate_verified(self, spec: BuiltinVoice) -> np.ndarray:
        """Generate the reference clip, checking it with Whisper.

        Voice design occasionally produces noise instead of speech, so every
        attempt is transcribed and compared with the text it should contain.
        """
        best: tuple[float, np.ndarray] | None = None
        for attempt in range(1, DESIGN_ATTEMPTS + 1):
            log.info("Generating built-in voice %s (attempt %d) ...", spec.id, attempt)
            audio = self.tts.design(spec.reference_text, spec.instruct, spec.language)
            heard = self.transcribe(audio, OUTPUT_SAMPLE_RATE, spec.language)
            score = word_similarity(spec.reference_text, heard)
            log.info("Voice %s attempt %d: similarity %.2f (%r)", spec.id, attempt, score, heard)
            if score >= DESIGN_MIN_SIMILARITY:
                return audio
            if best is None or score > best[0]:
                best = (score, audio)
        assert best is not None
        log.warning("Built-in voice %s could not be verified (best similarity %.2f).", spec.id, best[0])
        return best[1]

    def regenerate_builtin(self, voice_id: str) -> None:
        self._prompts[f"builtin:{voice_id}"] = self._builtin_prompt(voice_id, regenerate=True)

    # -- custom reference clips --------------------------------------------------------
    def _custom_prompt(self, source: Path) -> object:
        if not source.exists():
            raise FileNotFoundError(f"Voice file not found: {source}")
        self.custom_cache_dir.mkdir(parents=True, exist_ok=True)
        key = file_fingerprint(source)
        prompt_file = self.custom_cache_dir / f"{source.stem}-{key}.pt"
        if prompt_file.exists():
            return _load_prompt(prompt_file)
        log.info("Preparing custom voice %s ...", source.name)
        audio = load_reference_audio(source)
        duration = len(audio) / REF_SAMPLE_RATE
        if duration < 3:
            log.warning("Reference clip %s is only %.1fs; 3-10 s of clean speech works best.", source.name, duration)
        elif duration > 20:
            log.warning("Reference clip %s is %.0fs long; it will be trimmed. 3-10 s works best.", source.name, duration)
            audio = audio[: 15 * REF_SAMPLE_RATE]
        ref_text = self.transcribe(audio, REF_SAMPLE_RATE, None)
        if not ref_text:
            raise ValueError(f"No speech found in voice clip {source.name}.")
        prompt = self.tts.make_prompt(audio, REF_SAMPLE_RATE, ref_text)
        prompt.save(str(prompt_file))  # type: ignore[attr-defined]
        for old in self.custom_cache_dir.glob(f"{source.stem}-*.pt"):
            if old != prompt_file:
                old.unlink(missing_ok=True)
        return prompt


def load_reference_audio(path: Path) -> np.ndarray:
    """Decode a reference clip (MP3/WAV/FLAC/OGG) to mono float32 at 24 kHz.

    Silence trimming and level normalization are done by OmniVoice's prompt
    preprocessing. (The ffmpeg-based preprocessing of spec §7.1 arrives in M5.)
    """
    return load_audio(path, REF_SAMPLE_RATE)
