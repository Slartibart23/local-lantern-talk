"""Conversation engine: owns the models, the conversation state and the turn flow.

The web server is a thin layer around this class. All heavy work runs on two
background threads:

* the **worker** thread executes commands one at a time (load models, run a
  turn, apply settings), so model changes never race with a running turn;
* the **audio** thread resamples microphone audio to 16 kHz and runs VAD.

Microphone audio is only processed in the ``listening`` state, so the mic is
effectively muted while the assistant thinks and speaks (spec §9). Events for
the UI are delivered through the ``emit`` callback as JSON-able dicts, with an
optional binary payload (PCM audio).
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .audio_prep import VOICE_EXTENSIONS
from .characters import TEXT_COLORS, Character, CharacterError, CharacterStore
from .downloads import DownloadJob
from .errors import describe as describe_error
from .gpu import query_gpu
from .images import ImageError, ImageStore
from .images import valid_name as valid_image_name
from . import __version__
from .hardware import PROFILES, effective_profile, select_profile
from .llm import NO_THINKING_ARGS, ChatMessage, LlamaServer, LLMClient, SamplingConfig, ServerConfig
from .memory import (
    FACT_UPDATE_EVERY, FACTS_PROMPT, FACTS_SYSTEM, SUMMARY_PROMPT, SUMMARY_SYSTEM, MemoryStore, estimate_tokens,
    history_budget, memory_context, messages_to_summarize, parse_fact_update, summary_words, transcript_text,
)
from .catalog import PARTIAL_FACTOR
from .gguf_meta import GGUFError, model_info
from .models import ContextProfile, ModelError, ModelManager, NoGGUFError
from .paths import Paths
from .pipeline import Conversation, ConversationConfig, SegmentEvent, SpeechAudio, TurnDone, TurnError
from .prompts import DEFAULT_ASSISTANT_PERSONA, LANGUAGE_NAMES, user_context
from .recorder import CaptureBuffer, SessionItem, SessionRecorder, list_sessions, session_title, spoken_text
from .resample import StreamingResampler
from .sanitizer import split_segments
from .settings import Settings, SettingsStore
from .stt import WhisperSTT
from .textlang import guess_language
from .tts import PRECISIONS, OmniVoiceTTS, VoiceLibrary
from .vad import SAMPLE_RATE, SegmenterConfig, SileroVAD, Utterance, UtteranceSegmenter

log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = VOICE_EXTENSIONS
DEFAULT_VOICE = "builtin:en_female"
# Sample sentences for previewing a voice, in the voice's language.
PREVIEW_TEXT = {
    "de": "Hallo! So klinge ich. Ich freue mich auf unser Gespräch.",
    "en": "Hello! This is how I sound. I'm looking forward to our conversation.",
    "es": "¡Hola! Así sueno yo. Me alegra poder hablar contigo.",
    "fr": "Bonjour ! Voici ma voix. Je me réjouis de notre conversation.",
}
POST_PLAYBACK_MUTE_S = 0.3        # ignore the mic briefly after playback (room echo)
PLAYBACK_WAIT_MARGIN_S = 15.0     # safety timeout if the browser never reports playback end
LANGUAGE_CONFIDENCE = 0.6         # below this, keep the previously detected language
MAX_VOICE_UPLOAD_BYTES = 50 * 2**20

EmitFn = Callable[[dict[str, Any], bytes | None], None]


class State(str, Enum):
    LOADING = "loading"
    LISTENING = "listening"
    PAUSED = "paused"
    MIC_OFF = "mic_off"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


IDLE_STATES = {State.LISTENING, State.PAUSED, State.MIC_OFF}
CONTEXT_MIN, CONTEXT_MAX, CONTEXT_STEP = 2048, 131072, 1024
AUTO_CONTEXT_MIN = 4096          # "Automatic" never goes below this, even if it does not fit completely
AUTO_CONTEXT_RESERVE_MIB = 512   # extra room "Automatic" leaves on the GPU
SESSION_BITRATES = (32, 48, 64, 96, 128)


@dataclass
class Components:
    """Everything the engine drives; replaced by fakes in tests."""

    stt: Any
    vad: Any
    tts: Any
    voices: Any
    server: Any
    llm: Any


def build_components(paths: Paths, settings: Settings) -> Components:
    profile = effective_profile(asdict(settings))
    stt = WhisperSTT(paths.whisper_models / "large-v3-turbo",
                     device=profile.whisper_device, compute_type=profile.whisper_compute)
    tts = OmniVoiceTTS(paths.omnivoice_model, num_step=settings.num_step, precision=settings.tts_precision)
    voices = VoiceLibrary(
        tts, paths.config / "builtin_voices.yaml", paths.builtin_voices, paths.voice_cache,
        transcribe=lambda audio, sr, lang: stt.transcribe_at(audio, sr, lang).text,
        ffmpeg=paths.ffmpeg,
    )
    server = LlamaServer(paths.llama_server, paths.logs, env=dict(os.environ))
    return Components(
        stt=stt, vad=SileroVAD(paths.vad_models / "silero_vad.jit"), tts=tts,
        voices=voices, server=server, llm=LLMClient(server),
    )


class UserError(Exception):
    """An error with a message meant for the user (no stack trace in the UI)."""


class Engine:
    def __init__(
        self,
        paths: Paths,
        store: SettingsStore,
        components: Components,
        emit: EmitFn,
        persona: str = DEFAULT_ASSISTANT_PERSONA,
        models: ModelManager | None = None,
    ) -> None:
        self.paths = paths
        self.store = store
        self.c = components
        self._emit_fn = emit
        self.models = models or ModelManager(paths, self._on_download_update)
        self.vram_mb: dict[str, int] = {}          # measured VRAM per component (spec §5)
        self.model_status: dict[str, Any] = {}     # GPU layers of the loaded LLM
        s = store.settings
        if s.gpu_vram_mib and s.hardware_profile != select_profile(s.gpu_vram_mib).name:
            s.hardware_profile = select_profile(s.gpu_vram_mib).name  # e.g. "ultra", added after installing
            store.save()
        self.state = State.LOADING
        self.detail = "Starting…"
        self.fatal_error: str | None = None
        self.paused = False
        self.last_language: str | None = None
        self.transcript: list[dict[str, Any]] = []
        self.conversation = Conversation(
            self.c.llm, self.c.tts if s.voice_output else None, self.c.voices,
            ConversationConfig(persona=persona, voice=None, match_language=s.match_language),
        )
        self.conversation.config.user = user_context(s.user_name, s.user_about)
        self.images = ImageStore(paths.images)
        self._fresh_images: dict[str, float] = {}  # uploaded, maybe not saved yet (editor still open)
        self.characters = CharacterStore(paths.characters, paths.config / "default_characters")
        self.characters.load()
        self.memory = MemoryStore(paths.memory / "memory.db")
        self.character: Character | None = None
        self.resume: dict[str, Any] | None = None  # an unfinished chat that was loaded from memory
        self.replaying = False                     # a recorded message is playing in the browser
        self._msg_seq = 0                          # ids of transcript items (for replay buttons)
        self._replay_lock = threading.Lock()
        self.capture = CaptureBuffer()             # full-quality microphone audio (spec §12.1)
        self.recorder = SessionRecorder(
            paths.sessions, paths.ffmpeg, self._on_recorded,
            gap_ms=lambda: self.store.settings.session_gap_ms,
            bitrate_kbps=lambda: self.store.settings.session_bitrate_kbps)
        if self.characters.list():
            self._activate_character(s.character)
        self.segmenter = UtteranceSegmenter(
            self.c.vad, SegmenterConfig(threshold=s.vad_threshold, end_silence_ms=s.end_silence_ms))
        self._resampler: StreamingResampler | None = None
        self._reset_audio = threading.Event()
        self._playback_done = threading.Event()
        self._state_lock = threading.Lock()
        self._commands: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._audio: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue(maxsize=500)
        self._threads: list[threading.Thread] = []
        self._loaded_model: str | None = None
        self._loaded_context: int | None = None
        self._loaded_auto = False                  # the loaded context was chosen automatically
        self._context_profiles: dict[str, ContextProfile | None] = {}

    # ------------------------------------------------------------------ public API
    def start(self) -> None:
        for target, name in ((self._worker, "engine-worker"), (self._audio_loop, "engine-audio")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        self._commands.put(("load", None))

    def shutdown(self) -> None:
        self._commands.put(("stop", None))
        self._audio.put(None)
        self._playback_done.set()
        for t in self._threads:
            t.join(timeout=10)
        self.c.server.stop()
        self.recorder.close()  # writes what is still queued
        self.memory.close()

    def submit_text(self, text: str) -> None:
        text = text.strip()
        if text:
            self._commands.put(("text", text))

    def on_audio(self, pcm16: bytes, sample_rate: int) -> None:
        """Microphone audio from the browser: mono 16-bit PCM at ``sample_rate``."""
        if self.state is not State.LISTENING:
            return  # muted: thinking, speaking, paused, loading …
        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            self._audio.put_nowait((samples, sample_rate))
        except queue.Full:
            log.warning("Audio queue full; dropping microphone audio")

    def playback_ended(self) -> None:
        self._playback_done.set()

    def set_paused(self, paused: bool) -> None:
        self._commands.put(("pause", paused))

    def set_replaying(self, playing: bool) -> None:
        """A recorded message plays in the browser: keep the microphone muted meanwhile."""
        self._commands.put(("replay", playing))

    def open_sessions_folder(self) -> None:
        self.paths.sessions.mkdir(parents=True, exist_ok=True)
        os.startfile(self.paths.sessions)  # type: ignore[attr-defined]  # Windows only

    def update_settings(self, values: dict[str, Any]) -> None:
        self._commands.put(("settings", values))

    def new_conversation(self) -> None:
        self._commands.put(("new", None))

    # -- editing the end of the chat --------------------------------------------------------
    def delete_from(self, msg_id: int) -> None:
        """Delete this message and everything after it (chat, memory, LLM history, recording)."""
        self._commands.put(("delete_from", msg_id))

    def regenerate(self) -> None:
        """Write the last reply again."""
        self._commands.put(("regenerate", None))

    def edit_last(self, text: str) -> None:
        """Replace the last user message (and what followed it) with ``text`` and answer again."""
        text = text.strip()
        if text:
            self._commands.put(("edit_last", text))

    def stop_reply(self) -> None:
        """Stop the reply that is being written or spoken (not queued: the worker is busy with the turn)."""
        if self.state in (State.THINKING, State.SPEAKING):
            self.conversation.cancel.set()
            self._playback_done.set()

    def set_voice_language(self, voice: str, language: str) -> None:
        """Language spoken in a custom voice clip; "" means "detect it from the transcript"."""
        self._commands.put(("voice_language", (voice, language)))

    def set_reference_text(self, voice: str, text: str) -> None:
        """Transcript of a custom voice clip; an empty text means "use Whisper"."""
        self._commands.put(("reftext", (voice, text)))

    # -- model catalog (spec §13) -----------------------------------------------------
    def download_model(self, entry_id: str, quant: str | None = None, load_after: bool = True) -> None:
        self._commands.put(("download", (entry_id, quant, load_after)))

    def cancel_download(self) -> None:
        self.models.cancel_download()

    def delete_model(self, file_name: str) -> None:
        self._commands.put(("delete_model", file_name))

    def add_custom_model(self, text: str, quant: str | None, variant: str, download: bool = True) -> None:
        self._commands.put(("add_custom", (text, quant, variant, download)))

    def remove_custom_model(self, entry_id: str) -> None:
        self._commands.put(("remove_custom", entry_id))

    def set_model_label(self, file_name: str, variant: str) -> None:
        self._commands.put(("set_label", (file_name, variant)))

    def snapshot(self) -> dict[str, Any]:
        """Everything a freshly connected browser needs to render the UI."""
        job = self.models.downloader.job
        return {
            "type": "hello",
            "state": self.state.value,
            "detail": self.detail,
            "settings": asdict(self.store.settings),
            "voices": self.list_voices(),
            "models": self.list_models(),
            "model_status": self.model_status,
            "download": job.public() if job else None,
            "gpu": self.gpu_summary(),
            "transcript": self.transcript,
            "deletable_from": self._deletable_from(),
            "components": self.components_status(),
            "voice_info": self.voice_info(),
            "character": self.character.to_dict() if self.character else None,
            "characters": self.list_characters(),
            "memory": self.memory_view(),
            "resume": self.resume,
            "sessions": list_sessions(self.paths.sessions),
            "profiles": [{"name": p.name, "description": p.description, "quant": p.default_quant,
                          "whisper": p.whisper_compute, "auto_context": p.max_auto_context}
                         for p in PROFILES.values()],
            "context_info": self.context_info(),
            "version": __version__,
            "recording": self.recorder.files(),
        }

    # ------------------------------------------------------------------ pictures
    def add_image(self, kind: str, data: bytes) -> dict[str, str]:
        """Store an avatar or background the browser has already cropped and shrunk."""
        try:
            name = self.images.save(kind, data)
        except ImageError as exc:
            raise UserError(str(exc)) from exc
        self._fresh_images[name] = time.monotonic()
        return {"name": name}

    def _cleanup_images(self) -> None:
        """Pictures no character and no profile uses any more are removed (uploads of the last hour stay)."""
        s = self.store.settings
        keep = {s.user_avatar}
        for c in self.characters.list():
            keep.update((c.avatar, c.background))
        now = time.monotonic()
        self._fresh_images = {n: t for n, t in self._fresh_images.items() if now - t < 3600}
        self.images.cleanup(keep | set(self._fresh_images))

    # ------------------------------------------------------------------ voices
    def add_voice(self, filename: str, data: bytes) -> dict[str, Any]:
        """Store an uploaded reference clip in ``user_data/voices`` (spec §7.1).

        Returns the new voice id plus the clip's quality warnings (length, noise,
        clipping). The file name is sanitized and never overwrites an existing voice.
        """
        name = Path(filename).name  # drop any directory part
        suffix = Path(name).suffix.lower()
        if suffix not in AUDIO_EXTENSIONS:
            raise UserError("Please upload an audio file: MP3, WAV, FLAC, OGG, OPUS, M4A, AAC or WEBM.")
        if not data:
            raise UserError("The file is empty.")
        if len(data) > MAX_VOICE_UPLOAD_BYTES:
            raise UserError("The file is too large. A voice clip only needs 3-10 seconds of speech.")
        self.paths.voices.mkdir(parents=True, exist_ok=True)
        target = self._free_voice_path(_safe_stem(Path(name).stem), suffix)
        target.write_bytes(data)
        report = self.c.voices.clip_report(target)
        if report is None:
            target.unlink(missing_ok=True)
            raise UserError(f"Could not read {name} as audio.")
        self._emit({"type": "voices", "voices": self.list_voices()})
        return {"id": f"custom:{target.name}", "name": target.stem, "duration_s": report["duration_s"],
                "warnings": report["warnings"]}

    def _free_voice_path(self, stem: str, suffix: str, ignore: Path | None = None) -> Path:
        """``stem + suffix`` in user_data\\voices, numbered if the name is taken."""
        target = self.paths.voices / f"{stem}{suffix}"
        n = 2
        while target.exists() and target != ignore:
            target = self.paths.voices / f"{stem} ({n}){suffix}"
            n += 1
        return target

    def rename_voice(self, voice: str, new_name: str) -> None:
        self._commands.put(("rename_voice", (voice, new_name)))

    def delete_voice(self, voice: str) -> None:
        self._commands.put(("delete_voice", voice))

    def preview_voice(self, voice: str) -> None:
        self._commands.put(("preview_voice", voice))

    def regenerate_builtin_voices(self) -> None:
        self._commands.put(("regenerate_builtins", None))

    # ------------------------------------------------------------------ listings
    def list_voices(self) -> dict[str, list[dict[str, str]]]:
        builtin = [
            {"id": f"builtin:{v.id}", "name": v.name, "language": v.language, "gender": v.gender}
            for v in self.c.voices.builtins.values()
        ]
        custom = []
        if self.paths.voices.is_dir():
            for f in sorted(self.paths.voices.iterdir()):
                if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
                    custom.append({"id": f"custom:{f.name}", "name": f.stem})
        return {"builtin": builtin, "custom": custom}

    def list_models(self) -> list[dict[str, Any]]:
        loaded = self._loaded_model if self.c.server.running else None
        return self.models.view(self.available_llm_gb(), self.preferred_quant(), loaded)

    def components_status(self) -> dict[str, Any]:
        return {
            "whisper": bool(self.c.stt.loaded),
            "vad": bool(self.c.vad.loaded),
            "omnivoice": bool(self.c.tts.loaded),
            "llm": self._loaded_model if self.c.server.running else None,
            "vram_mb": {k: v for k, v in self.vram_mb.items() if self._component_loaded(k)},
        }

    def _component_loaded(self, name: str) -> bool:
        return {"whisper": self.c.stt.loaded, "omnivoice": self.c.tts.loaded,
                "llm": self.c.server.running}.get(name, False)

    # -- VRAM bookkeeping (spec §5) --------------------------------------------------------
    def gpu_summary(self) -> dict[str, Any]:
        s = self.store.settings
        profile = effective_profile(asdict(s))
        return {"name": s.gpu_name, "vram_mib": s.gpu_vram_mib, "profile": profile.name,
                "recommended_quant": profile.default_quant, "available_gb": round(self.available_llm_gb(), 1)}

    def preferred_quant(self) -> str:
        return effective_profile(asdict(self.store.settings)).default_quant

    def llm_margin_mib(self) -> int:
        """Free VRAM kept for other components: OmniVoice grows by ~2.4 GB while speaking."""
        return 1024 + (2560 if self.store.settings.voice_output else 0)

    def available_llm_gb(self) -> float:
        """VRAM a language model may use: free memory plus what the current model occupies."""
        gpu = query_gpu()
        if gpu is None:
            total = self.store.settings.gpu_vram_mib or 0
            return max(0.0, (total - self.llm_margin_mib()) / 1024) if total else 0.0
        llm = self.vram_mb.get("llm", 0) if self.c.server.running else 0
        free = int(gpu["total_mb"]) - int(gpu["used_mb"]) + llm
        return max(0.0, (free - self.llm_margin_mib()) / 1024)

    @staticmethod
    def _gpu_used_mb() -> int:
        gpu = query_gpu()
        return int(gpu["used_mb"]) if gpu else 0

    def _wait_for_vram_release(self, below_mb: int, timeout_s: float = 5.0) -> None:
        """After stopping llama-server, wait until the driver has actually freed its memory."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and self._gpu_used_mb() > below_mb:
            time.sleep(0.25)

    def _measured(self, name: str, load: Callable[[], None]) -> None:
        """Run ``load`` and remember how much VRAM it took (nvidia-smi delta)."""
        before = self._gpu_used_mb()
        load()
        after = self._gpu_used_mb()
        if before and after:
            self.vram_mb[name] = max(0, after - before)

    # ------------------------------------------------------------------ events / state
    def _emit(self, msg: dict[str, Any], binary: bytes | None = None) -> None:
        try:
            self._emit_fn(msg, binary)
        except Exception:  # noqa: BLE001 - a broken client must never stop the engine
            log.exception("emit failed")

    def _set_state(self, state: State, detail: str = "") -> None:
        with self._state_lock:
            self.state = state
            self.detail = detail
        self._emit({"type": "state", "state": state.value, "detail": detail})

    def _idle_state(self) -> State:
        if self.fatal_error:
            return State.ERROR
        if not self.store.settings.mic_enabled:
            return State.MIC_OFF
        return State.PAUSED if (self.paused or self.replaying) else State.LISTENING

    def _enter_idle(self) -> None:
        self._reset_audio.set()  # the audio thread resets VAD state before listening again
        state = self._idle_state()
        detail = self.fatal_error or ("Replay is playing" if self.replaying and state is State.PAUSED else "")
        self._set_state(state, detail)

    # ------------------------------------------------------------------ audio thread
    def _audio_loop(self) -> None:
        while True:
            item = self._audio.get()
            if item is None:
                return
            if self._reset_audio.is_set():
                self._reset_audio.clear()
                self.segmenter.reset()
                self.capture.reset()
                if self._resampler is not None:
                    self._resampler.reset()
                self._drain_audio()
                continue
            if self.state is not State.LISTENING:
                continue
            samples, sr = item
            try:
                if self._resampler is None or self._resampler.orig_sr != sr:
                    self._resampler = StreamingResampler(sr, SAMPLE_RATE)
                    self.segmenter.reset()
                    self.capture.reset()
                self.capture.feed(samples, sr)
                for utt in self.segmenter.feed(self._resampler.feed(samples)):
                    utt.raw = self.capture.slice_16k(utt.start, utt.end)
                    utt.raw_sr = self.capture.sample_rate
                    # Stop listening right away; the worker transcribes the utterance.
                    self._set_state(State.THINKING, "Transcribing…")
                    self._commands.put(("utterance", utt))
                    break
            except Exception:  # noqa: BLE001
                log.exception("Audio processing failed")

    def _drain_audio(self) -> None:
        while True:
            try:
                if self._audio.get_nowait() is None:
                    self._audio.put(None)  # keep the stop signal
                    return
            except queue.Empty:
                return

    # ------------------------------------------------------------------ worker thread
    def _worker(self) -> None:
        handlers: dict[str, Callable[[Any], None]] = {
            "load": lambda _: self._load_all(),
            "text": self._handle_typed,
            "utterance": self._handle_utterance,
            "pause": self._handle_pause,
            "settings": self._apply_settings,
            "new": lambda _: self._clear_conversation(),
            "reftext": lambda arg: self._handle_reference_text(*arg),
            "download": lambda arg: self._handle_download(*arg),
            "download_done": self._handle_download_done,
            "delete_model": self._handle_delete_model,
            "add_custom": lambda arg: self._handle_add_custom(*arg),
            "remove_custom": self._handle_remove_custom,
            "set_label": lambda arg: self._handle_set_label(*arg),
            "rename_voice": lambda arg: self._handle_rename_voice(*arg),
            "delete_voice": self._handle_delete_voice,
            "preview_voice": self._handle_preview_voice,
            "regenerate_builtins": self._handle_regenerate_builtins,
            "select_character": self._handle_select_character,
            "save_character": lambda arg: self._handle_save_character(*arg),
            "duplicate_character": self._handle_duplicate_character,
            "delete_character": self._handle_delete_character,
            "add_fact": self._handle_add_fact,
            "update_fact": lambda arg: self._handle_update_fact(*arg),
            "delete_fact": self._handle_delete_fact,
            "reset_memory": self._handle_reset_memory,
            "replay": self._handle_replay,
            "voice_language": lambda arg: self._handle_voice_language(*arg),
            "delete_from": self._handle_delete_from,
            "regenerate": self._handle_regenerate,
            "edit_last": self._handle_edit_last,
        }
        while True:
            name, arg = self._commands.get()
            if name == "stop":
                return
            try:
                handlers[name](arg)
            except (UserError, ModelError) as exc:
                self._emit({"type": "error", "message": str(exc)})
                self._enter_idle()
            except Exception as exc:  # noqa: BLE001 - details go to the log
                log.exception("Command %s failed", name)
                self._emit({"type": "error", "message": describe_error(exc)})
                self._enter_idle()

    # -- model catalog ----------------------------------------------------------------
    def _emit_models(self) -> None:
        self._emit({"type": "models", "models": self.list_models()})

    def _on_download_update(self, job: DownloadJob) -> None:
        """Called from the download thread (progress, finished, failed)."""
        self._emit({"type": "download", **job.public()})

    def _handle_download(self, entry_id: str, quant: str | None, load_after: bool) -> None:
        def on_done(job: DownloadJob) -> None:
            self._commands.put(("download_done", (job, load_after)))

        self.models.start_download(entry_id, quant, self.available_llm_gb(), self.preferred_quant(), on_done)
        self._emit_models()

    def _handle_download_done(self, arg: tuple[DownloadJob, bool]) -> None:
        job, load_after = arg
        if job.state == "error":
            self._emit({"type": "error", "message": job.error})
        elif job.state == "done" and load_after:
            # Spec §13.3: a model selected for the first time is loaded right after its download.
            self.store.settings.model = job.file_name
            self.store.save()
            self._ensure_llm()
            self.fatal_error = None
            self._emit({"type": "settings", "settings": asdict(self.store.settings)})
            self._emit({"type": "components", "components": self.components_status()})
            self._enter_idle()
        self._emit_models()

    def _handle_delete_model(self, file_name: str) -> None:
        if file_name == self._loaded_model and self.c.server.running:
            raise UserError("This model is loaded. Load another model before deleting it.")
        self.models.delete_file(file_name)
        self._emit_models()

    def _handle_add_custom(self, text: str, quant: str | None, variant: str, download: bool) -> None:
        try:
            entry = self.models.add_custom(text, quant, variant, self.preferred_quant())
        except NoGGUFError as exc:
            # Not an error the user caused: offer the GGUF versions of the same model instead.
            self._emit({"type": "custom_suggestions", "message": str(exc), "suggestions": exc.suggestions,
                        "quant": quant or "", "variant": variant})
            return
        self._emit_models()
        if download:
            self._handle_download(entry.id, next(iter(entry.files)), True)

    def _handle_remove_custom(self, entry_id: str) -> None:
        self.models.remove_custom(entry_id)
        self._emit_models()

    def _handle_set_label(self, file_name: str, variant: str) -> None:
        self.models.set_label(file_name, variant)
        self._emit_models()

    # -- loading -------------------------------------------------------------------
    def _load_all(self) -> None:
        s = self.store.settings
        try:
            if self.paths.voices.is_dir():
                self.c.voices.cleanup_cache(self.paths.voices)  # caches of deleted / replaced clips
            if s.mic_enabled:
                self._load_mic()
            if s.voice_output:
                self._load_voice_output()
            self._ensure_llm()
            self.fatal_error = None
        except Exception as exc:  # noqa: BLE001
            log.exception("Loading failed")
            self.fatal_error = str(exc) if isinstance(exc, (UserError, FileNotFoundError)) else (
                f"Loading failed: {exc}. Details are in logs\\server.log.")
        # Loading may have filled in defaults (e.g. the model file); show them.
        self._emit({"type": "settings", "settings": asdict(self.store.settings)})
        self._emit({"type": "components", "components": self.components_status()})
        self._enter_idle()

    def _load_mic(self) -> None:
        self._set_state(State.LOADING, "Loading speech recognition…")
        self._measured("whisper", lambda: (self.c.stt.load(), self.c.vad.load()))

    def _unload_mic(self) -> None:
        self.c.stt.unload()
        self.c.vad.unload()

    def _load_voice_output(self) -> None:
        self._set_state(State.LOADING, "Loading voice synthesis…")
        self._measured("omnivoice", self.c.tts.load)
        self._sync_tts()
        self._prepare_voice()

    def _prepare_voice(self) -> None:
        voice = self.store.settings.voice
        ref = self._voice_ref(voice)
        self.conversation.config.voice = ref
        if self.store.settings.voice_output:
            self._set_state(State.LOADING, f"Preparing voice {self._voice_label(voice)}…")
            self.c.voices.prompt_for(ref)
        self._emit({"type": "voice_info", **self.voice_info()})

    def _voice_ref(self, voice: str) -> str:
        """Map a UI voice id to a VoiceLibrary id (built-in id or file path)."""
        if voice.startswith("custom:"):
            path = self.paths.voices / voice.removeprefix("custom:")
            if not path.exists():
                raise UserError(f"Voice file {path.name} was not found in user_data\\voices.")
            return str(path)
        if voice.removeprefix("builtin:") not in self.c.voices.builtins:
            raise UserError(f"Unknown voice {voice}.")
        return voice

    def _voice_name(self, voice: str) -> str:
        """Readable name of a UI voice id ("German female", or the clip name)."""
        if voice.startswith("builtin:"):
            spec = self.c.voices.builtins.get(voice.removeprefix("builtin:"))
            if spec is not None:
                return spec.name
        return Path(voice.split(":", 1)[-1]).stem

    @staticmethod
    def _voice_label(voice: str) -> str:
        return voice.split(":", 1)[-1]

    def _model_path(self) -> Path:
        names = sorted(self.models.downloaded_files())
        if not names:
            raise UserError("No language model yet. Open the model catalog (Change model…) and download one.")
        wanted = self.store.settings.model
        name = wanted if wanted in names else names[0]
        if name != wanted:
            self.store.settings.model = name
            self.store.save()
        return self.paths.llm_models / name

    # -- context length -----------------------------------------------------------------
    @property
    def context(self) -> int:
        """Context length of the loaded model (the setting until a model is loaded)."""
        return self._loaded_context or self.store.settings.context

    def _context_profile(self, file_name: str) -> ContextProfile | None:
        """How the model's GPU memory grows with the context; measured once per model file."""
        if file_name not in self._context_profiles:
            self._context_profiles[file_name] = self.models.context_profile(file_name)
        return self._context_profiles[file_name]

    def _model_max_context(self, file_name: str, profile: ContextProfile | None) -> int:
        trained = profile.max_context if profile else 0
        if not trained:
            try:
                trained = model_info(self.paths.llm_models / file_name).context_length
            except (OSError, GGUFError):
                trained = 0
        return min(trained or CONTEXT_MAX, CONTEXT_MAX)

    def _target_context(self, file_name: str) -> int:
        """The setting, or with "Automatic" the largest context that fits completely on the GPU."""
        s = self.store.settings
        profile = self._context_profile(file_name)
        top = self._model_max_context(file_name, profile)
        if not s.context_auto:
            return max(CONTEXT_MIN, min(s.context, top))
        cap = min(effective_profile(asdict(s)).max_auto_context, top)
        if profile is None:  # llama-fit-params did not run: the old default
            return min(8192, cap)
        available = self.available_llm_gb() * 1024 - AUTO_CONTEXT_RESERVE_MIB
        return profile.largest_fitting(available, AUTO_CONTEXT_MIN, cap, CONTEXT_STEP) or min(AUTO_CONTEXT_MIN, cap)

    def context_info(self) -> dict[str, Any]:
        """For the context slider: memory need per context of the selected model and what is free."""
        s = self.store.settings
        file_name = self._loaded_model or s.model
        profile = self._context_profiles.get(file_name) if file_name else None
        return {
            "auto": s.context_auto, "value": s.context, "loaded": self._loaded_context if self._loaded_model else None,
            "min": CONTEXT_MIN, "max": self._model_max_context(file_name, profile) if file_name else 32768,
            "step": CONTEXT_STEP, "auto_max": effective_profile(asdict(s)).max_auto_context,
            "available_mib": round(self.available_llm_gb() * 1024), "reserve_mib": AUTO_CONTEXT_RESERVE_MIB,
            "partial_factor": PARTIAL_FACTOR, "profile": profile.public() if profile else None,
        }

    def _emit_context_info(self) -> None:
        self._emit({"type": "context_info", **self.context_info()})

    def _ensure_llm(self) -> None:
        path = self._model_path()
        s = self.store.settings
        running = self.c.server.running and self._loaded_model == path.name
        if running and s.context_auto and self._loaded_auto:
            return  # chosen when it was loaded; free memory measured now would differ a little
        entry = self.models.entry_for_file(path.name)
        label = entry.display_name if entry else path.name
        if path.name not in self._context_profiles:
            self._set_state(State.LOADING, f"Checking how much context fits for {label}…")
        ctx = self._target_context(path.name)
        if running and self._loaded_context == ctx:
            self._loaded_auto = s.context_auto
            return
        self._set_state(State.LOADING, f"Loading model {label}…")
        # Stop the old model first so the fit check sees the memory it frees.
        was_running = self.c.server.running
        before_stop = self._gpu_used_mb()
        self.c.server.stop()
        self._loaded_model = None
        freed = self.vram_mb.pop("llm", 0)
        if was_running and before_stop:
            self._wait_for_vram_release(before_stop - freed // 2)
        margin = self.llm_margin_mib()
        gpu = query_gpu()
        true_free = int(gpu["total_mb"]) - int(gpu["used_mb"]) if gpu else None
        fit = self.models.fit(path.name, ctx, margin, true_free)
        if fit.nothing_fits:
            raise UserError(f"{label} is too big for the free GPU memory. Pick a smaller model or "
                            "quantization, or turn voice output off to free memory.")
        no_thinking = NO_THINKING_ARGS
        if fit.known:
            # Apply llama-fit-params' split exactly (incl. MoE expert offload), keeping
            # room for OmniVoice; llama-server's own --fit would ignore an explicit -ngl.
            ngl = fit.args[fit.args.index("-ngl") + 1]
            extra = [a for i, a in enumerate(fit.args) if a != "-ngl" and (i == 0 or fit.args[i - 1] != "-ngl")]
            config = ServerConfig(model=path, context=ctx, gpu_layers="all" if ngl == "-1" else ngl,
                                  extra_args=[*extra, "--fit", "off", *no_thinking])
        else:
            config = ServerConfig(model=path, context=ctx, gpu_layers="auto",
                                  extra_args=["--fit-target", str(margin), *no_thinking])
        try:
            self._measured("llm", lambda: self.c.server.start(config))
        except Exception as exc:
            raise UserError(f"Could not load {label}: {exc}") from exc
        self._loaded_model, self._loaded_context, self._loaded_auto = path.name, ctx, s.context_auto
        if fit.cpu_blocks:
            notice = (f"Part of the model (the experts of {fit.cpu_blocks} of {fit.total_layers} blocks) runs on "
                      "the CPU to leave room for speech, so replies are a little slower. Mixture-of-experts "
                      "models handle this well; a smaller quantization would be faster.")
        elif fit.partial:
            notice = (f"{fit.gpu_layers} of {fit.total_layers} layers fit on the GPU; the rest runs on the CPU, "
                      "so replies are slower. A smaller quantization or model would be faster.")
        else:
            notice = ""
        self.model_status = {
            "file": path.name, "name": label, "gpu_layers": fit.gpu_layers, "total_layers": fit.total_layers,
            "cpu_blocks": fit.cpu_blocks, "partial": fit.partial, "notice": notice,
            "context": ctx, "context_auto": s.context_auto,
        }
        self._emit({"type": "model_status", **self.model_status})
        self._emit({"type": "models", "models": self.list_models()})
        self._emit_context_info()

    # -- turns ---------------------------------------------------------------------
    def _handle_utterance(self, utt: Utterance) -> None:
        self._set_state(State.THINKING, "Transcribing…")
        tr = self.c.stt.transcribe(utt.audio)
        text = tr.text.strip()
        if not text or not text.strip(" .,!?…"):
            self._enter_idle()
            return
        if tr.language_probability >= LANGUAGE_CONFIDENCE or self.last_language is None:
            self.last_language = tr.language
        raw = utt.raw if utt.raw is not None and len(utt.raw) else utt.audio
        self._turn(text, self.last_language, "voice", stt_ms=round(tr.seconds * 1000),
                   audio=raw, audio_sr=utt.raw_sr if raw is utt.raw else SAMPLE_RATE)

    def _handle_typed(self, text: str) -> None:
        # Typed text has no Whisper language; guess it so the reply voice matches.
        language = guess_language(text)
        if language:
            self.last_language = language
        self._turn(text, self.last_language, "text")

    def _turn(self, text: str, language: str | None, source: str, stt_ms: int | None = None,
              audio: np.ndarray | None = None, audio_sr: int = 0, user: dict[str, Any] | None = None) -> None:
        """One exchange. ``user`` is an existing message to answer again (Regenerate)."""
        self._recover_llm()
        if self.fatal_error or not self.c.server.running:
            raise UserError(self.fatal_error or "The language model is not loaded.")
        self._set_state(State.THINKING, "")
        self.resume = None  # talking on continues the chat
        recording = self._ensure_session()
        if user is None:
            user = {"id": self._new_id(), "role": "user", "text": text, "language": language, "source": source}
            self.transcript.append(user)
            self._emit({"type": "user_message", **user, "stt_ms": stt_ms})
            if recording:  # encoded in the background while the model answers
                tags = ["spoken", language] if source == "voice" and language else [
                    "spoken" if source == "voice" else "typed"]
                self.recorder.add(SessionItem(user["id"], "user", self._user_label(), text, tags,
                                              audio if source == "voice" else None, audio_sr or SAMPLE_RATE))

        voice = self.conversation.voice_for(language)
        reply: dict[str, Any] = {"id": self._new_id(), "role": "assistant", "segments": [],
                                 "voice": self._display_voice(voice)}
        self.transcript.append(reply)
        self._emit({"type": "assistant_start", "id": reply["id"], "voice": reply["voice"]})
        spoken: list[np.ndarray] = []
        spoken_sr = 24_000

        self._playback_done.clear()
        chunks = 0
        audio_seconds = 0.0
        reply_text = ""
        for event in self.conversation.run_turn(text, language):
            if isinstance(event, SegmentEvent):
                seg = event.segment
                _append_segment(reply["segments"], seg.kind, seg.text)
                self._emit({"type": "segment", "kind": seg.kind, "text": seg.text})
            elif isinstance(event, SpeechAudio):
                if chunks == 0:
                    self._set_state(State.SPEAKING, "")
                pcm = (np.clip(event.audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                self._emit({"type": "audio", "index": event.index, "sample_rate": event.sample_rate,
                            "text": event.text}, pcm)
                spoken.append(event.audio)
                spoken_sr = event.sample_rate
                chunks += 1
                audio_seconds += len(event.audio) / event.sample_rate
            elif isinstance(event, TurnError):
                self._emit({"type": "error", "message": event.message})
            elif isinstance(event, TurnDone):
                reply_text = event.reply
                m = event.metrics
                stopped = self.conversation.cancel.is_set()
                if stopped:
                    reply["stopped"] = True
                self._emit({"type": "turn_done", "audio_chunks": chunks, "metrics": m.summary(), "stopped": stopped,
                            "first_audio_ms": None if m.first_audio is None else round(m.first_audio * 1000)})
        if chunks and not self.conversation.cancel.is_set():
            # Mic stays muted until the browser has played everything (spec §9 step 5).
            if not self._playback_done.wait(timeout=audio_seconds + PLAYBACK_WAIT_MARGIN_S):
                log.warning("Browser did not report the end of playback; continuing.")
            time.sleep(POST_PLAYBACK_MUTE_S)
        if recording and reply_text:
            c = self.character
            tags = [self._voice_name(reply["voice"])] if spoken and reply["voice"] else ["text only"]
            self.recorder.add(SessionItem(
                reply["id"], "assistant", c.name if c else "Assistant", spoken_text(reply["segments"]),
                tags, np.concatenate(spoken) if spoken else None, spoken_sr, segments=reply["segments"]))
        if reply_text:
            user["in_history"] = reply["in_history"] = True  # the pipeline added both to the LLM history
        self._remember_turn(user, reply, language, source, reply_text)
        self._enter_idle()
        if reply_text:
            self._maintain_memory()  # between turns; a new utterance waits in the queue
        self._emit_editable()

    def _recover_llm(self) -> None:
        """llama-server stopped on its own (crash, killed): start it again before the next turn."""
        if self.fatal_error or self.c.server.running or self._loaded_model is None:
            return
        log.warning("llama-server is not running any more; restarting it")
        self._loaded_model = None
        self.vram_mb.pop("llm", None)
        self._ensure_llm()
        self._emit({"type": "notice", "message": "The language model had stopped and was started again."})

    def _display_voice(self, ref: str | None) -> str | None:
        if ref is None:
            return None
        if ref.startswith("builtin:"):
            return ref
        return f"custom:{Path(ref).name}"

    def _clear_conversation(self) -> None:
        """Start over with the active character; with memory on, the old talk goes into the summary."""
        self._end_session()
        c = self.character
        if c is not None and c.memory_enabled:
            old = self.memory.active_messages(c.id)
            if any(m.role == "user" for m in old):
                if self.c.server.running:
                    self._set_state(State.THINKING, "Saving the conversation to memory…")
                    try:
                        self._summarize(c, old)
                        self._update_facts(c)
                    except Exception:  # noqa: BLE001 - never block a new conversation
                        log.exception("Could not summarize the conversation")
                self.memory.mark_summarized(c.id, old[-1].id)
            elif old:
                self.memory.mark_summarized(c.id, old[-1].id)  # only the greeting
            self._refresh_memory_context()
        if c is not None:
            self._load_history()
        else:
            self.conversation.history.clear()
            self.transcript.clear()
        self._emit({"type": "cleared"})
        self._emit_character_state()
        self._enter_idle()

    # -- deleting, regenerating and editing the end of the chat ----------------------------------
    def _deletable_from(self) -> int | None:
        """UI id of the oldest message that can still be deleted (together with everything after it).

        The greeting stays, and so does what is already folded into the memory summary.
        """
        c = self.character
        active = {m.id for m in self.memory.active_messages(c.id)} if c is not None and c.memory_enabled else None
        first = None
        for item in reversed(self.transcript):
            if item.get("greeting") or (active is not None and item.get("db_id") and item["db_id"] not in active):
                break
            first = item["id"]
        return first

    def _emit_editable(self) -> None:
        self._emit({"type": "editable", "deletable_from": self._deletable_from()})

    def _emit_transcript(self) -> None:
        self._emit({"type": "transcript", "transcript": self.transcript, "deletable_from": self._deletable_from()})
        self._emit_sessions()

    def _check_deletable(self, index: int) -> None:
        first = self._deletable_from()
        if first is None or self.transcript[index]["id"] < first:
            raise UserError("This part of the chat is already in the character's memory summary and cannot be "
                            "deleted any more.")

    def _truncate(self, index: int, keep_user: bool = False) -> None:
        """Remove ``transcript[index:]`` from the chat, the memory, the LLM history and the recording.

        With ``keep_user`` the user message before ``index`` stays in the chat and the recording
        but leaves the memory and the LLM history, so it can be answered again.
        """
        removed = self.transcript[index:]
        forget = ([self.transcript[index - 1]] if keep_user else []) + removed
        self.recorder.wait_idle()  # every finished message knows its place in the recording
        cuts: dict[str, int] = {}
        for item in removed:
            if item.get("session"):
                cuts[item["session"]] = cuts.get(item["session"], 0) + 1
        for name, count in cuts.items():
            self.recorder.truncate(name, count)
        c = self.character
        if c is not None and c.memory_enabled:
            ids = [item["db_id"] for item in forget if item.get("db_id")]
            if ids:
                self.memory.delete_from(c.id, min(ids))
            self._reload_llm_history()
        else:
            count = sum(1 for item in forget if item.get("in_history"))
            if count:
                del self.conversation.history[-count:]
        for item in forget:
            item.pop("in_history", None)
            item.pop("db_id", None)
        del self.transcript[index:]
        self.recorder.wait_idle()
        log.info("Deleted the last %d messages of the chat", len(removed))

    def _handle_delete_from(self, msg_id: int) -> None:
        index = next((i for i, t in enumerate(self.transcript) if t.get("id") == msg_id), None)
        if index is None:
            raise UserError("This message is no longer in the chat.")
        self._check_deletable(index)
        self._truncate(index)
        self._emit_transcript()
        self._enter_idle()

    def _handle_regenerate(self, _: Any = None) -> None:
        items = self.transcript
        if (len(items) < 2 or items[-1].get("role") != "assistant" or items[-1].get("greeting")
                or items[-2].get("role") != "user"):
            raise UserError("There is no reply to write again.")
        self._check_deletable(len(items) - 2)
        user = items[-2]
        self._truncate(len(items) - 1, keep_user=True)
        self._emit_transcript()
        if user.get("language"):
            self.last_language = user["language"]
        self._turn(user["text"], self.last_language, user.get("source") or "text", user=user)

    def _handle_edit_last(self, text: str) -> None:
        index = next((i for i in range(len(self.transcript) - 1, -1, -1)
                      if self.transcript[i].get("role") == "user"), None)
        if index is not None:
            self._check_deletable(index)
            self._truncate(index)
            self._emit_transcript()
        self._handle_typed(text)

    # -- characters (spec §10) -----------------------------------------------------------
    def list_characters(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.characters.list()]

    def memory_view(self) -> dict[str, Any]:
        c = self.character
        if c is None:
            return {"character": None, "enabled": False, "facts": [], "summary": ""}
        return {
            "character": c.id, "enabled": c.memory_enabled, "summary": self.memory.summary(c.id),
            "facts": [{"id": f.id, "text": f.text, "source": f.source} for f in self.memory.facts(c.id)],
        }

    def select_character(self, char_id: str) -> None:
        self._commands.put(("select_character", char_id))

    def save_character(self, data: dict[str, Any], char_id: str | None = None) -> None:
        self._commands.put(("save_character", (data, char_id)))

    def duplicate_character(self, char_id: str) -> None:
        self._commands.put(("duplicate_character", char_id))

    def delete_character(self, char_id: str) -> None:
        self._commands.put(("delete_character", char_id))

    def add_fact(self, text: str) -> None:
        self._commands.put(("add_fact", text))

    def update_fact(self, fact_id: int, text: str) -> None:
        self._commands.put(("update_fact", (fact_id, text)))

    def delete_fact(self, fact_id: int) -> None:
        self._commands.put(("delete_fact", fact_id))

    def reset_memory(self) -> None:
        self._commands.put(("reset_memory", None))

    def _voice_output_on(self) -> bool:
        """Voice output is on globally and for the active character (spec §7.4)."""
        c = self.character
        return bool(self.store.settings.voice_output and (c is None or c.voice_output))

    def _sync_tts(self) -> None:
        self.conversation.tts = self.c.tts if (self._voice_output_on() and self.c.tts.loaded) else None

    def _activate_character(self, char_id: str | None) -> None:
        """Switch persona, voice, model, memory and history in one step (spec §10)."""
        char = self.characters.get(char_id or "") or self.characters.first()
        if char is None:
            raise UserError("No characters found. Create one with + New character.")
        if self.character is not None and self.character.id != char.id:
            self._end_session()
        self.character = char
        s = self.store.settings
        s.character = char.id
        if char.voice:
            try:
                self._voice_ref(char.voice)
                s.voice = char.voice
            except UserError:
                log.warning("Voice %s of %s is missing; keeping %s", char.voice, char.id, s.voice)
        s.show_actions = char.show_actions_as_text
        if char.model and (self.paths.llm_models / char.model).exists():
            s.model = char.model
        self.store.save()
        cfg = self.conversation.config
        cfg.persona = char.system_prompt or char.persona or DEFAULT_ASSISTANT_PERSONA
        cfg.roleplay = char.kind == "roleplay"
        self._refresh_memory_context()
        self._load_history()
        self._sync_tts()

    def _refresh_memory_context(self) -> None:
        c = self.character
        enabled = c is not None and c.memory_enabled
        self.conversation.config.memory = (
            memory_context(self.memory.summary(c.id), self.memory.facts(c.id)) if enabled and c else "")

    def _load_history(self) -> None:
        """LLM history and visible transcript of the active character (from its memory)."""
        c = self.character
        assert c is not None
        self.conversation.history = []
        self.transcript = []
        self.resume = None
        if c.memory_enabled:
            stored = self.memory.active_messages(c.id)
            if any(m.role == "user" for m in stored):
                # Offer "Continue" / "Start new chat" in the UI (the chat is loaded either way).
                self.resume = {"messages": len(stored), "last": stored[-1].created}
            for m in stored:
                self.conversation.history.append(ChatMessage(m.role, m.content))
                item: dict[str, Any] = {"id": self._new_id(), "db_id": m.id, "session": m.replay_session,
                                        "offset": m.replay_offset, "duration": m.replay_duration,
                                        "in_history": True}
                if m.role == "user":
                    item.update({"role": "user", "text": m.content, "language": m.language,
                                 "source": m.source or "text"})
                else:
                    item.update({"role": "assistant", "voice": m.voice,
                                 "greeting": m.voice is None and m.content == c.greeting,
                                 "segments": [{"kind": seg.kind, "text": seg.text}
                                              for seg in split_segments(m.content)]})
                self.transcript.append(item)
        if self.resume is None:
            self.memory.set_meta(c.id, "session", "")  # a fresh chat gets a new session recording
        if not self.conversation.history and c.greeting:
            self.conversation.history.append(ChatMessage("assistant", c.greeting))
            greeting: dict[str, Any] = {"id": self._new_id(), "role": "assistant", "voice": None, "greeting": True,
                                        "in_history": True,
                                        "segments": [{"kind": seg.kind, "text": seg.text}
                                                     for seg in split_segments(c.greeting)]}
            self.transcript.append(greeting)
            if c.memory_enabled:
                greeting["db_id"] = self.memory.add_message(c.id, "assistant", c.greeting)

    def _reload_llm_history(self) -> None:
        """After summarizing: the LLM sees only the unsummarized messages (the transcript stays)."""
        c = self.character
        if c is not None and c.memory_enabled:
            self.conversation.history = [ChatMessage(m.role, m.content) for m in self.memory.active_messages(c.id)]

    def _emit_character_state(self) -> None:
        self._emit({"type": "character", "character": self.character.to_dict() if self.character else None,
                    "characters": self.list_characters(), "transcript": self.transcript,
                    "deletable_from": self._deletable_from(), "memory": self.memory_view(), "resume": self.resume})
        self._emit({"type": "settings", "settings": asdict(self.store.settings)})
        self._emit({"type": "voice_info", **self.voice_info()})

    def _handle_select_character(self, char_id: str) -> None:
        if self.character is not None and self.character.id == char_id:
            return
        if self.characters.get(char_id) is None:
            raise UserError("This character no longer exists.")
        self._activate_character(char_id)
        self._prepare_voice()
        self._ensure_llm()
        self._emit_character_state()
        self._emit({"type": "components", "components": self.components_status()})
        self._enter_idle()

    def _handle_save_character(self, data: dict[str, Any], char_id: str | None) -> None:
        voice = str(data.get("voice") or "")
        if voice:
            self._voice_ref(voice)  # raises a clear error for unknown voices
        try:
            char = self.characters.save(data, char_id)
        except CharacterError as exc:
            raise UserError(str(exc)) from exc
        if char_id is None or (self.character and self.character.id == char.id):
            self._activate_character(char.id)  # a new or edited active character takes effect now
            self._prepare_voice()
            self._ensure_llm()
        self._cleanup_images()
        self._emit_character_state()
        self._enter_idle()

    def _handle_duplicate_character(self, char_id: str) -> None:
        try:
            copy = self.characters.duplicate(char_id)
            if copy.avatar or copy.background:  # the copy gets its own pictures
                data = copy.to_dict()
                data["avatar"], data["background"] = self.images.copy(copy.avatar), self.images.copy(copy.background)
                self.characters.save(data, copy.id)
        except CharacterError as exc:
            raise UserError(str(exc)) from exc
        self._emit_character_state()

    def _handle_delete_character(self, char_id: str) -> None:
        try:
            self.characters.delete(char_id)
        except CharacterError as exc:
            raise UserError(str(exc)) from exc
        self.memory.reset(char_id)  # its history and facts go with it
        self._cleanup_images()
        if self.character is not None and self.character.id == char_id:
            self._activate_character(None)
            self._prepare_voice()
            self._ensure_llm()
        self._emit_character_state()
        self._enter_idle()

    # -- memory (spec §11) -----------------------------------------------------------------
    def _remember_turn(self, user: dict[str, Any], reply: dict[str, Any], language: str | None, source: str,
                       reply_text: str) -> None:
        c = self.character
        if c is None or not c.memory_enabled or not reply_text:
            self._trim_history()
            return
        user_db = self.memory.add_message(c.id, "user", user["text"], language, source)
        reply_db = self.memory.add_message(c.id, "assistant", reply_text, language, None, reply["voice"])
        for item, db_id in ((user, user_db), (reply, reply_db)):
            with self._replay_lock:  # the recorder may already have finished this message
                item["db_id"] = db_id
                if item.get("session"):
                    self.memory.set_replay(db_id, item["session"], item["offset"], item.get("duration"))

    def _trim_history(self) -> None:
        """Without memory the oldest exchanges drop out once the history outgrows its context share."""
        history = self.conversation.history
        budget = history_budget(self.context)
        while len(history) > 2 and sum(estimate_tokens(m.content) for m in history) > budget:
            del history[:2]

    def _llm_text(self, system: str, prompt: str, max_tokens: int = 400) -> str:
        """One quick, low-temperature request used to maintain the memory."""
        messages = [ChatMessage("system", system), ChatMessage("user", prompt)]
        text = self.c.llm.complete(messages, SamplingConfig(temperature=0.3, max_tokens=max_tokens))
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    def _maintain_memory(self) -> None:
        """Between turns: summarize old history when it gets long; update the facts every few turns."""
        c = self.character
        if c is None or not c.memory_enabled or not self.c.server.running:
            return
        try:
            old = messages_to_summarize(self.memory.active_messages(c.id), self.context)
            if old:
                self._summarize(c, old)
            turns = int(self.memory.get_meta(c.id, "turns_since_facts", "0") or 0) + 1
            if turns >= FACT_UPDATE_EVERY:
                self._update_facts(c)
                turns = 0
            self.memory.set_meta(c.id, "turns_since_facts", str(turns))
        except Exception:  # noqa: BLE001 - memory upkeep must never break the conversation
            log.exception("Memory maintenance failed")
        self._refresh_memory_context()
        self._emit({"type": "memory", **self.memory_view()})

    def _summarize(self, c: Character, old: list[Any]) -> None:
        words = summary_words(self.context)
        summary = self._llm_text(
            SUMMARY_SYSTEM.format(name=c.name),
            SUMMARY_PROMPT.format(summary=self.memory.summary(c.id) or "(none)",
                                  transcript=transcript_text(old, c.name, self.store.settings.user_name),
                                  words=words),
            max_tokens=max(400, words * 2))
        if summary:
            self.memory.set_summary(c.id, summary)
            self.memory.mark_summarized(c.id, old[-1].id)
            self._reload_llm_history()
            log.info("Summarized %d messages of %s", len(old), c.id)

    def _update_facts(self, c: Character) -> None:
        recent = self.memory.active_messages(c.id)[-8:]
        facts = self.memory.facts(c.id)
        reply = self._llm_text(
            FACTS_SYSTEM.format(name=c.name),
            FACTS_PROMPT.format(facts="\n".join(f"- {f.text}" for f in facts) or "(none)",
                                transcript=transcript_text(recent, c.name, self.store.settings.user_name)),
            max_tokens=300)
        add, remove = parse_fact_update(reply)
        self.memory.remove_facts_by_text(c.id, remove)
        for text in add:
            self.memory.add_fact(c.id, text, source="llm")

    def _handle_add_fact(self, text: str) -> None:
        c = self._memory_character()
        self.memory.add_fact(c.id, text, source="user")
        self._refresh_memory_context()
        self._emit({"type": "memory", **self.memory_view()})

    def _handle_update_fact(self, fact_id: int, text: str) -> None:
        c = self._memory_character()
        self.memory.update_fact(c.id, int(fact_id), text)
        self._refresh_memory_context()
        self._emit({"type": "memory", **self.memory_view()})

    def _handle_delete_fact(self, fact_id: int) -> None:
        c = self._memory_character()
        self.memory.delete_fact(c.id, int(fact_id))
        self._refresh_memory_context()
        self._emit({"type": "memory", **self.memory_view()})

    def _handle_reset_memory(self, _: Any = None) -> None:
        c = self._memory_character()
        self._end_session()
        self.memory.reset(c.id)
        self._refresh_memory_context()
        self._load_history()
        self._emit_character_state()

    def _memory_character(self) -> Character:
        if self.character is None:
            raise UserError("No character is active.")
        return self.character

    # -- session recording (spec §12) -----------------------------------------------------
    def _new_id(self) -> int:
        self._msg_seq += 1
        return self._msg_seq

    def _user_label(self) -> str:
        """The user's name in the session transcript ("You" without a profile name)."""
        return " ".join(self.store.settings.user_name.split()) or "You"

    def _recording_on(self) -> bool:
        c = self.character
        return bool(self.store.settings.record_sessions and (c is None or c.recording_enabled))

    def _ensure_session(self) -> bool:
        """Start (or continue) the session recording of the current chat; False when not recording."""
        if not self._recording_on():
            self._end_session()
            return False
        if self.recorder.name is not None:
            return True
        c = self.character
        char_id = c.id if c else "chat"
        name = self.memory.get_meta(char_id, "session", "")
        if name and SessionRecorder.valid_name(name) and (self.paths.sessions / f"{name}.md").exists():
            self.recorder.resume(name)  # a resumed chat continues its recording
        else:
            name = self.recorder.new_name(char_id)
            self.recorder.start(name, session_title(c.name if c else "Assistant"))
            self.memory.set_meta(char_id, "session", name)
            for item in self.transcript:  # the greeting that opened this chat
                if item.get("greeting"):
                    self.recorder.add(SessionItem(item["id"], "assistant", c.name if c else "Assistant",
                                                  spoken_text(item["segments"]), ["greeting"],
                                                  segments=item["segments"]))
        self._emit_sessions()
        return True

    def _end_session(self) -> None:
        if self.recorder.name is not None:
            self.recorder.stop()
            self._emit_sessions()

    def _emit_sessions(self) -> None:
        self._emit({"type": "sessions", "sessions": list_sessions(self.paths.sessions),
                    "recording": self.recorder.files()})

    def _on_recorded(self, msg_id: int, session: str, offset: float, duration: float | None) -> None:
        """Called by the recorder thread once a message is in the session files."""
        with self._replay_lock:
            item = next((t for t in self.transcript if t.get("id") == msg_id), None)
            if item is not None:
                item.update(session=session, offset=offset, duration=duration)
                if item.get("db_id"):
                    self.memory.set_replay(item["db_id"], session, offset, duration)
        self._emit({"type": "recorded", "id": msg_id, "session": session, "offset": offset, "duration": duration})

    def _handle_replay(self, playing: bool) -> None:
        self.replaying = bool(playing)
        if self.state in IDLE_STATES:
            self._enter_idle()

    # -- custom voice reference text -----------------------------------------------------
    def voice_info(self) -> dict[str, Any]:
        """Details of the selected voice for the UI (reference text of custom clips)."""
        voice = self.store.settings.voice
        info: dict[str, Any] = {"voice": voice, "custom": voice.startswith("custom:"),
                                "reference_text": None, "origin": None}
        if info["custom"]:
            path = self.paths.voices / voice.removeprefix("custom:")
            ref = self.c.voices.reference_text(path) if path.exists() else None
            if ref:
                info["reference_text"], info["origin"] = ref
            info["text_file"] = str(path.with_suffix(".txt").relative_to(self.paths.root))
            info["report"] = self.c.voices.clip_report(path) if path.exists() else None
            info["language"] = self.c.voices.custom_language(path) if path.exists() else None
            info["language_set"] = bool(path.exists() and self.c.voices.user_language(path))
        return info

    # -- managing voices (spec §7.1 / §7.2) -------------------------------------------------
    def _custom_path(self, voice: str) -> Path:
        if not voice.startswith("custom:"):
            raise UserError("Built-in voices cannot be renamed or deleted.")
        return Path(self._voice_ref(voice))

    def _handle_rename_voice(self, voice: str, new_name: str) -> None:
        path = self._custom_path(voice)
        target = self._free_voice_path(_safe_stem(new_name), path.suffix, ignore=path)
        if target == path:
            return
        path.rename(target)  # keeps the time stamp, so the voice cache stays valid
        for sidecar in (".txt", ".lang"):  # transcript and chosen language move along
            if path.with_suffix(sidecar).exists():
                path.with_suffix(sidecar).rename(target.with_suffix(sidecar))
        self.c.voices.forget(path)
        new_id = f"custom:{target.name}"
        if self.store.settings.voice == voice:
            self.store.settings.voice = new_id
            self.store.save()
            self.conversation.config.voice = str(target)
            self._emit({"type": "settings", "settings": asdict(self.store.settings)})
        self._emit({"type": "voices", "voices": self.list_voices()})
        self._emit({"type": "voice_info", **self.voice_info()})

    def _handle_delete_voice(self, voice: str) -> None:
        path = self._custom_path(voice)
        path.unlink(missing_ok=True)
        path.with_suffix(".txt").unlink(missing_ok=True)
        path.with_suffix(".lang").unlink(missing_ok=True)
        self.c.voices.forget(path)
        self.c.voices.cleanup_cache(self.paths.voices)
        if self.store.settings.voice == voice:
            # The selected voice is gone: fall back to a built-in one (it follows the language).
            self._apply_settings({"voice": DEFAULT_VOICE})
        self._emit({"type": "voices", "voices": self.list_voices()})
        self._emit({"type": "voice_info", **self.voice_info()})

    def _handle_preview_voice(self, voice: str) -> None:
        """Speak a sample sentence with ``voice`` (does not change the selected voice)."""
        if not self.store.settings.voice_output or self.conversation.tts is None:
            raise UserError("Turn voice output on to preview voices.")
        ref = self._voice_ref(voice)
        language = self.c.voices.language_of(ref) or self.last_language or "en"
        text = PREVIEW_TEXT.get(language, PREVIEW_TEXT["en"])
        self._set_state(State.LOADING, f"Preparing voice {self._voice_label(voice)}…")
        prompt = self.c.voices.prompt_for(ref)
        result = self.c.tts.synthesize(text, prompt, language)
        self._playback_done.clear()
        self._set_state(State.SPEAKING, "Voice preview")
        self._emit({"type": "preview_start", "voice": voice, "text": text})
        pcm = (np.clip(result.audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        self._emit({"type": "audio", "index": 0, "sample_rate": 24_000, "text": text}, pcm)
        self._emit({"type": "preview_done"})
        if not self._playback_done.wait(timeout=result.duration_s + PLAYBACK_WAIT_MARGIN_S):
            log.warning("Browser did not report the end of the preview.")
        time.sleep(POST_PLAYBACK_MUTE_S)
        self._enter_idle()

    def _handle_regenerate_builtins(self, _: Any = None) -> None:
        if not self.store.settings.voice_output or not self.c.tts.loaded:
            raise UserError("Turn voice output on to generate the built-in voices.")
        builtins = list(self.c.voices.builtins.values())
        for n, spec in enumerate(builtins, 1):
            self._set_state(State.LOADING, f"Generating built-in voice {n} of {len(builtins)}: {spec.name}…")
            self.c.voices.regenerate_builtin(spec.id)
        self._prepare_voice()
        self._emit({"type": "notice", "message": "All 8 built-in voices were generated again."})
        self._enter_idle()

    def _handle_voice_language(self, voice: str, language: str) -> None:
        if not voice.startswith("custom:"):
            raise UserError("Built-in voices have a fixed language.")
        language = language.strip().lower()
        if language and language not in LANGUAGE_NAMES:
            raise UserError(f"Unknown language {language}.")
        path = Path(self._voice_ref(voice))
        self.c.voices.set_language(path, language)
        if voice == self.store.settings.voice:
            self._prepare_voice()  # transcribed again with the right language if needed
        self._emit({"type": "voice_info", **self.voice_info()})
        self._enter_idle()

    def _handle_reference_text(self, voice: str, text: str) -> None:
        if not voice.startswith("custom:"):
            raise UserError("Only your own voices have a reference text.")
        path = Path(self._voice_ref(voice))
        self.c.voices.set_reference_text(path, text[:2000])
        if voice == self.store.settings.voice:
            self._prepare_voice()  # rebuild the voice with the new transcript
        self._emit({"type": "voice_info", **self.voice_info()})
        self._enter_idle()

    # -- settings ----------------------------------------------------------------------
    def _handle_pause(self, paused: bool) -> None:
        self.paused = bool(paused)
        if self.state in IDLE_STATES:
            self._enter_idle()

    def _apply_settings(self, values: dict[str, Any]) -> None:
        s = self.store.settings
        before = asdict(s)
        changed = s.update(values)
        if not changed:
            return
        self.store.save()
        try:
            self._apply_changes(changed)
        except Exception:
            # Roll back so the UI shows what is actually active.
            s.update(before)
            self.store.save()
            raise
        finally:
            self._emit({"type": "settings", "settings": asdict(s)})
            self._emit({"type": "components", "components": self.components_status()})
            self._emit_context_info()  # free memory changes with the microphone and voice output
            if self.state is State.LOADING:
                self._enter_idle()

    def _apply_changes(self, changed: dict[str, Any]) -> None:
        s = self.store.settings
        self._validate(changed)
        if "end_silence_ms" in changed:
            self.segmenter.config.end_silence_ms = s.end_silence_ms
        if "vad_threshold" in changed:
            self.segmenter.config.threshold = s.vad_threshold
            self.segmenter.config.neg_threshold = max(0.1, s.vad_threshold - 0.15)
        if "match_language" in changed:
            self.conversation.config.match_language = s.match_language
        if changed.keys() & {"user_name", "user_about"}:
            s.user_name, s.user_about = " ".join(s.user_name.split()), s.user_about.strip()
            self.conversation.config.user = user_context(s.user_name, s.user_about)
        if "user_avatar" in changed:
            self._cleanup_images()
        if "num_step" in changed:
            if not 8 <= s.num_step <= 96:
                raise UserError("Speech quality steps must be between 8 and 96.")
            self.c.tts.num_step = s.num_step
        if "tts_precision" in changed:
            if s.tts_precision not in PRECISIONS:
                raise UserError(f"Unknown precision {s.tts_precision}.")
            self.c.tts.precision = s.tts_precision
            if s.voice_output and "voice_output" not in changed:
                self.c.tts.unload()  # reloaded below with the new precision
                self._load_voice_output()
        if "mic_enabled" in changed:
            if s.mic_enabled:
                self._load_mic()
            else:
                self._unload_mic()  # text-only input: free Whisper / VAD VRAM (spec §5)
        if "voice_output" in changed:
            if s.voice_output:
                self._load_voice_output()
            else:
                self.conversation.tts = None
                self.c.tts.unload()  # replies as text only, OmniVoice leaves VRAM (spec §7.4)
        elif "voice" in changed:
            self._prepare_voice()
        if changed.keys() & {"model", "context", "context_auto", "hardware_profile_override"}:
            self._ensure_llm()
            self.fatal_error = None
        if "record_sessions" in changed and not s.record_sessions:
            self._end_session()
        if self.character is not None and changed.keys() & {"voice", "show_actions", "voice_output"}:
            self._save_character_from_settings(changed)
        if changed.keys() - {"show_actions", "user_name", "user_about", "user_avatar", "user_color"}:
            self._enter_idle()

    def _validate(self, changed: dict[str, Any]) -> None:
        """Reject values the settings screen never sends (hand-edited or stale clients)."""
        s = self.store.settings
        checks = {
            "vad_threshold": (0.2 <= s.vad_threshold <= 0.9, "Speech detection must be between 0.2 and 0.9."),
            "end_silence_ms": (200 <= s.end_silence_ms <= 3000, "End of speech must be between 200 and 3000 ms."),
            "context": (CONTEXT_MIN <= s.context <= CONTEXT_MAX and s.context % CONTEXT_STEP == 0,
                        f"Context length must be {CONTEXT_MIN} to {CONTEXT_MAX} tokens in steps of {CONTEXT_STEP}."),
            "session_bitrate_kbps": (s.session_bitrate_kbps in SESSION_BITRATES,
                                     f"MP3 quality must be one of {', '.join(map(str, SESSION_BITRATES))} kbit/s."),
            "session_gap_ms": (0 <= s.session_gap_ms <= 3000, "The pause between turns must be 0 to 3000 ms."),
            "hardware_profile_override": (s.hardware_profile_override in ("", *PROFILES),
                                          f"Unknown hardware profile {s.hardware_profile_override}."),
            "user_name": (len(s.user_name) <= 60, "Your name is too long (60 characters at most)."),
            "user_about": (len(s.user_about) <= 1000, "The text about you is too long (1000 characters at most)."),
            "user_avatar": (s.user_avatar == "" or valid_image_name(s.user_avatar), "Unknown picture."),
            "user_color": (s.user_color in ("", *TEXT_COLORS), f"Unknown colour {s.user_color}."),
        }
        for key, (ok, message) in checks.items():
            if key in changed and not ok:
                raise UserError(message)

    def _save_character_from_settings(self, changed: dict[str, Any]) -> None:
        """Voice and show-actions picked in the side panel belong to the active character (spec §10)."""
        c = self.character
        assert c is not None
        data = c.to_dict()
        if "voice" in changed:
            data["voice"] = self.store.settings.voice
        if "show_actions" in changed:
            data["show_actions_as_text"] = self.store.settings.show_actions
        if "voice_output" in changed and self.store.settings.voice_output and not c.voice_output:
            data["voice_output"] = True  # turning voice on in the panel also turns it on for this character
        if data != c.to_dict():
            self.character = self.characters.save(data, c.id)
            self._sync_tts()
            self._emit({"type": "characters", "characters": self.list_characters(),
                        "character": self.character.to_dict()})


def _append_segment(segments: list[dict[str, str]], kind: str, text: str) -> None:
    """Store streamed segments compactly: consecutive speech fragments are merged."""
    if segments and kind == "speech" and segments[-1]["kind"] == "speech":
        segments[-1]["text"] += text
    else:
        segments.append({"kind": kind, "text": text})


def _safe_stem(stem: str) -> str:
    """A file name stem without path separators or characters Windows does not allow."""
    return re.sub(r"[^\w\- ()]+", "_", stem).strip(" ._") or "voice"
