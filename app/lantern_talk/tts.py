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

from .audio_io import load_audio, resample
from .gpu import free_cuda_cache
from .textmatch import word_similarity

log = logging.getLogger(__name__)

OUTPUT_SAMPLE_RATE = 24_000
REF_SAMPLE_RATE = 24_000
# OmniVoice weights are stored in fp32. fp16 / bf16 are ~2.6x faster than fp32 and
# need about half the VRAM (measured on an RTX 5090: 2.3 GB vs 4.2 GB).
PRECISIONS = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}
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

    def __init__(self, model_dir: Path, device: str = "cuda:0", num_step: int = 32, precision: str = "fp16") -> None:
        self.model_dir = model_dir
        self.device = device
        self.num_step = num_step
        self.precision = precision  # "fp16" | "bf16" | "fp32"; takes effect on the next load()
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
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()  # no "Loading weights" bars in the console
        dtype = PRECISIONS.get(self.precision, "float16")
        self._model = OmniVoice.from_pretrained(str(self.model_dir), device_map=self.device, dtype=getattr(torch, dtype))
        # The first generation is ~1.3 s slower (CUDA kernel setup); pay that now, not in the first reply.
        with torch.inference_mode():
            self._model.generate(text="Hello.", num_step=self.num_step)
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
    """Cache key that changes when the audio changes, but not when the file is renamed."""
    st = path.stat()
    h = hashlib.sha1()
    h.update(f"{st.st_size}|{st.st_mtime_ns}|".encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 18))  # first 256 KB: guards against an unchanged size and time
    return h.hexdigest()[:16]


class VoiceLibrary:
    """Resolves voice ids to cached OmniVoice prompts.

    * ``builtin:<id>`` — shipped clip in ``config\\builtin_voices`` (or generated with voice
      design when it is missing / regenerated); prompts in ``models\\builtin_voices``.
    * a file path     — custom reference clip; cached in ``user_data\\voices\\.cache``.
    """

    def __init__(
        self,
        tts: OmniVoiceTTS,
        builtin_config: Path,
        builtin_dir: Path,
        custom_cache_dir: Path,
        transcribe: Callable[[np.ndarray, int, str | None], str],
        ffmpeg: Path | None = None,
    ) -> None:
        self.tts = tts
        self.builtins = load_builtin_voices(builtin_config)
        self.shipped_dir = builtin_config.with_suffix("")  # config\builtin_voices\<id>.flac
        self.builtin_dir = builtin_dir
        self.custom_cache_dir = custom_cache_dir
        self.transcribe = transcribe
        self.ffmpeg = ffmpeg if ffmpeg and ffmpeg.exists() else None
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
        path = Path(voice)
        return self.custom_language(path) if path.exists() else None

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
        shipped = self.shipped_dir / f"{voice_id}.flac"
        if shipped.exists() and not regenerate:
            # The app ships the 8 reference clips, so a new install sounds the same
            # everywhere and does not spend minutes on voice design.
            audio, sr = sf.read(str(shipped), dtype="float32")
            if sr != OUTPUT_SAMPLE_RATE:
                audio = resample(audio, sr, OUTPUT_SAMPLE_RATE)
        else:
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
    # Cache files in user_data\voices\.cache, keyed by the audio fingerprint (not the name,
    # so renaming a voice keeps its cache):
    #   <fp>.prepared.wav   trimmed, loudness-normalized 24 kHz clip (ffmpeg)
    #   <fp>.meta.json      quality report (length, noise, clipping)
    #   <fp>.whisper.txt    Whisper transcript, if the user gave none
    #   <fp>-<text>.pt      OmniVoice prompt built from the clip and its transcript
    @staticmethod
    def text_file_for(source: Path) -> Path:
        """User-provided transcript next to the clip: ``clip.mp3`` -> ``clip.txt``."""
        return source.with_suffix(".txt")

    def reference_text(self, source: Path) -> tuple[str, str] | None:
        """``(text, origin)`` used for a custom clip; origin is ``"user"`` or ``"whisper"``.

        ``None`` while the clip has not been transcribed yet.
        """
        user_text = self._user_text(source)
        if user_text:
            return user_text, "user"
        cached = self._cache_file(source, ".whisper.txt")
        if cached.exists():
            return cached.read_text(encoding="utf-8").strip(), "whisper"
        return None

    def set_reference_text(self, source: Path, text: str) -> None:
        """Store the transcript of a clip; an empty text falls back to Whisper."""
        target = self.text_file_for(source)
        text = text.strip()
        if text:
            target.write_text(text + "\n", encoding="utf-8")
        else:
            target.unlink(missing_ok=True)
        self._prompts.pop(str(source), None)  # prepared again with the new text

    @staticmethod
    def language_file_for(source: Path) -> Path:
        """The language the user chose for a clip: ``clip.mp3`` -> ``clip.lang`` (e.g. ``de``)."""
        return source.with_suffix(".lang")

    def user_language(self, source: Path) -> str | None:
        f = self.language_file_for(source)
        if not f.exists():
            return None
        code = f.read_text(encoding="utf-8").strip().lower()
        return code or None

    def custom_language(self, source: Path) -> str | None:
        """Language of a custom clip: the user's choice, else guessed from its transcript."""
        from .textlang import guess_language

        chosen = self.user_language(source)
        if chosen:
            return chosen
        ref = self.reference_text(source)
        return guess_language(ref[0]) if ref else None

    def set_language(self, source: Path, code: str) -> None:
        """Store the language of a clip; an empty code means "detect it"."""
        target = self.language_file_for(source)
        code = code.strip().lower()
        if code == (self.user_language(source) or ""):
            return
        if code:
            target.write_text(code + "\n", encoding="utf-8")
        else:
            target.unlink(missing_ok=True)
        if not self._user_text(source):
            # Whisper transcribes the clip again, now knowing its language.
            self._cache_file(source, ".whisper.txt").unlink(missing_ok=True)
        self._prompts.pop(str(source), None)

    def clip_report(self, source: Path) -> dict[str, object] | None:
        """Quality report of a clip (computed once, then cached); None if it cannot be read."""
        import json

        from .audio_prep import PrepError, analyze

        meta = self._cache_file(source, ".meta.json")
        if meta.exists():
            try:
                return json.loads(meta.read_text(encoding="utf-8"))
            except ValueError:
                pass
        try:
            prepared = self._prepared_audio(source)
            original = load_audio(source, REF_SAMPLE_RATE) if self.ffmpeg is None else self._decode(source)
        except (PrepError, OSError, RuntimeError):
            return None
        report = analyze(original, prepared).to_dict()
        meta.write_text(json.dumps(report), encoding="utf-8")
        return report

    def forget(self, source: Path) -> None:
        """Drop everything cached for a clip (after it was deleted or replaced)."""
        self._prompts.pop(str(source), None)

    def cleanup_cache(self, voices_dir: Path) -> None:
        """Remove cache files that belong to no current voice file."""
        if not self.custom_cache_dir.is_dir():
            return
        current = {file_fingerprint(f) for f in voices_dir.iterdir()
                   if f.is_file() and f.suffix.lower() not in (".txt", ".lang")}
        for f in self.custom_cache_dir.iterdir():
            if f.is_file() and f.name[:16] not in current:
                f.unlink(missing_ok=True)

    def _cache_file(self, source: Path, suffix: str) -> Path:
        return self.custom_cache_dir / f"{file_fingerprint(source)}{suffix}"

    def _user_text(self, source: Path) -> str:
        f = self.text_file_for(source)
        return f.read_text(encoding="utf-8").strip() if f.exists() else ""

    def _decode(self, source: Path) -> np.ndarray:
        from .audio_prep import decode

        assert self.ffmpeg is not None
        return decode(source, self.ffmpeg, REF_SAMPLE_RATE)

    def _prepared_audio(self, source: Path) -> np.ndarray:
        """The clip as OmniVoice should hear it: trimmed and loudness-normalized (cached)."""
        import soundfile as sf

        from .audio_prep import prepare

        if self.ffmpeg is None:  # e.g. ffmpeg missing: OmniVoice's own preprocessing still applies
            return load_reference_audio(source)
        self.custom_cache_dir.mkdir(parents=True, exist_ok=True)
        cached = self._cache_file(source, ".prepared.wav")
        if cached.exists():
            audio, _ = sf.read(str(cached), dtype="float32")
            return audio
        return prepare(source, cached, self.ffmpeg)

    def _custom_prompt(self, source: Path) -> object:
        if not source.exists():
            raise FileNotFoundError(f"Voice file not found: {source}")
        self.custom_cache_dir.mkdir(parents=True, exist_ok=True)
        user_text = self._user_text(source)
        whisper_file = self._cache_file(source, ".whisper.txt")
        ref_text = user_text or (whisper_file.read_text(encoding="utf-8").strip() if whisper_file.exists() else "")
        if ref_text:
            prompt_file = self._prompt_file(source, ref_text)
            if prompt_file.exists():
                return _load_prompt(prompt_file)

        log.info("Preparing custom voice %s ...", source.name)
        audio = self._prepared_audio(source)
        duration = len(audio) / REF_SAMPLE_RATE
        if duration > 20 and not user_text:
            # Whisper transcribes what is kept; a user transcript covers the whole clip.
            audio = audio[: 15 * REF_SAMPLE_RATE]
        if not ref_text:
            ref_text = self.transcribe(audio, REF_SAMPLE_RATE, self.user_language(source)).strip()
            if not ref_text:
                raise ValueError(f"No speech found in voice clip {source.name}.")
            whisper_file.write_text(ref_text + "\n", encoding="utf-8")
        prompt = self.tts.make_prompt(audio, REF_SAMPLE_RATE, ref_text)
        prompt_file = self._prompt_file(source, ref_text)
        prompt.save(str(prompt_file))  # type: ignore[attr-defined]
        self._remove_old_prompts(source, keep=prompt_file)
        return prompt

    def _prompt_file(self, source: Path, ref_text: str) -> Path:
        text_key = hashlib.sha1(ref_text.encode("utf-8")).hexdigest()[:8]
        return self.custom_cache_dir / f"{file_fingerprint(source)}-{text_key}.pt"

    def _remove_old_prompts(self, source: Path, keep: Path) -> None:
        """Prompts built from an older transcript of the same clip."""
        for f in self.custom_cache_dir.glob(f"{file_fingerprint(source)}-*.pt"):
            if f != keep:
                f.unlink(missing_ok=True)


def load_reference_audio(path: Path) -> np.ndarray:
    """Decode a reference clip (MP3/WAV/FLAC/OGG) to mono float32 at 24 kHz.

    Silence trimming and level normalization are done by OmniVoice's prompt
    preprocessing. (The ffmpeg-based preprocessing of spec §7.1 arrives in M5.)
    """
    return load_audio(path, REF_SAMPLE_RATE)
