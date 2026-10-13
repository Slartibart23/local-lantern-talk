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

from .llm import LlamaServer, LLMClient, ServerConfig
from .paths import Paths
from .pipeline import Conversation, ConversationConfig, SegmentEvent, SpeechAudio, TurnDone, TurnError
from .prompts import DEFAULT_ASSISTANT_PERSONA
from .resample import StreamingResampler
from .hardware import effective_profile
from .settings import Settings, SettingsStore
from .stt import WhisperSTT
from .textlang import guess_language
from .tts import PRECISIONS, OmniVoiceTTS, VoiceLibrary
from .vad import SAMPLE_RATE, SegmenterConfig, SileroVAD, Utterance, UtteranceSegmenter

log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg")
POST_PLAYBACK_MUTE_S = 0.3        # ignore the mic briefly after playback (room echo)
PLAYBACK_WAIT_MARGIN_S = 15.0     # safety timeout if the browser never reports playback end
LANGUAGE_CONFIDENCE = 0.6         # below this, keep the previously detected language
MAX_VOICE_UPLOAD_BYTES = 50 * 2**20
RECOMMENDED_CLIP_S = (3.0, 10.0)  # spec §7.1: 3-10 s of clean speech

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
    ) -> None:
        self.paths = paths
        self.store = store
        self.c = components
        self._emit_fn = emit
        s = store.settings
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

    def update_settings(self, values: dict[str, Any]) -> None:
        self._commands.put(("settings", values))

    def new_conversation(self) -> None:
        self._commands.put(("new", None))

    def set_reference_text(self, voice: str, text: str) -> None:
        """Transcript of a custom voice clip; an empty text means "use Whisper"."""
        self._commands.put(("reftext", (voice, text)))

    def snapshot(self) -> dict[str, Any]:
        """Everything a freshly connected browser needs to render the UI."""
        return {
            "type": "hello",
            "state": self.state.value,
            "detail": self.detail,
            "settings": asdict(self.store.settings),
            "voices": self.list_voices(),
            "models": self.list_models(),
            "transcript": self.transcript,
            "components": self.components_status(),
            "voice_info": self.voice_info(),
        }

    # ------------------------------------------------------------------ voices
    def add_voice(self, filename: str, data: bytes) -> dict[str, Any]:
        """Store an uploaded reference clip in ``user_data/voices`` (spec §7.1).

        Returns the new voice id plus warnings about the clip length. The file name
        is sanitized and never overwrites an existing voice.
        """
        import soundfile as sf

        name = Path(filename).name  # drop any directory part
        suffix = Path(name).suffix.lower()
        if suffix not in AUDIO_EXTENSIONS:
            raise UserError("Please upload an MP3, WAV, FLAC or OGG file.")
        if not data:
            raise UserError("The file is empty.")
        if len(data) > MAX_VOICE_UPLOAD_BYTES:
            raise UserError("The file is too large. A voice clip only needs 3-10 seconds of speech.")
        stem = re.sub(r"[^\w\- ]+", "_", Path(name).stem).strip(" ._") or "voice"
        self.paths.voices.mkdir(parents=True, exist_ok=True)
        target = self.paths.voices / f"{stem}{suffix}"
        n = 2
        while target.exists():
            target = self.paths.voices / f"{stem} ({n}){suffix}"
            n += 1
        target.write_bytes(data)
        try:
            duration = sf.info(str(target)).duration
        except Exception as exc:  # noqa: BLE001 - unreadable upload
            target.unlink(missing_ok=True)
            raise UserError(f"Could not read {name} as audio.") from exc
        warnings = []
        low, high = RECOMMENDED_CLIP_S
        if duration < low:
            warnings.append(f"The clip is only {duration:.1f} s long; {low:.0f}-{high:.0f} s of clean speech clone best.")
        elif duration > 20:
            warnings.append(f"The clip is {duration:.0f} s long; only the first 15 s are used. {low:.0f}-{high:.0f} s work best.")
        elif duration > high:
            warnings.append(f"The clip is {duration:.0f} s long; {low:.0f}-{high:.0f} s usually clone best.")
        self._emit({"type": "voices", "voices": self.list_voices()})
        return {"id": f"custom:{target.name}", "name": target.stem, "duration_s": round(duration, 1),
                "warnings": warnings}

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
        models = []
        if self.paths.llm_models.is_dir():
            for f in sorted(self.paths.llm_models.glob("*.gguf")):
                models.append({"file": f.name, "size_gb": round(f.stat().st_size / 1e9, 1), "quant": _quant(f.name)})
        return models

    def components_status(self) -> dict[str, Any]:
        return {
            "whisper": bool(self.c.stt.loaded),
            "vad": bool(self.c.vad.loaded),
            "omnivoice": bool(self.c.tts.loaded),
            "llm": self._loaded_model if self.c.server.running else None,
        }

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
        return State.PAUSED if self.paused else State.LISTENING

    def _enter_idle(self) -> None:
        self._reset_audio.set()  # the audio thread resets VAD state before listening again
        state = self._idle_state()
        self._set_state(state, self.fatal_error or "")

    # ------------------------------------------------------------------ audio thread
    def _audio_loop(self) -> None:
        while True:
            item = self._audio.get()
            if item is None:
                return
            if self._reset_audio.is_set():
                self._reset_audio.clear()
                self.segmenter.reset()
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
                for utt in self.segmenter.feed(self._resampler.feed(samples)):
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
        }
        while True:
            name, arg = self._commands.get()
            if name == "stop":
                return
            try:
                handlers[name](arg)
            except UserError as exc:
                self._emit({"type": "error", "message": str(exc)})
                self._enter_idle()
            except Exception as exc:  # noqa: BLE001 - details go to the log
                log.exception("Command %s failed", name)
                self._emit({"type": "error", "message": f"Something went wrong ({exc}). Details are in logs\\server.log."})
                self._enter_idle()

    # -- loading -------------------------------------------------------------------
    def _load_all(self) -> None:
        s = self.store.settings
        try:
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
        self.c.stt.load()
        self.c.vad.load()

    def _unload_mic(self) -> None:
        self.c.stt.unload()
        self.c.vad.unload()

    def _load_voice_output(self) -> None:
        self._set_state(State.LOADING, "Loading voice synthesis…")
        self.c.tts.load()
        self.conversation.tts = self.c.tts
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

    @staticmethod
    def _voice_label(voice: str) -> str:
        return voice.split(":", 1)[-1]

    def _model_path(self) -> Path:
        models = self.list_models()
        if not models:
            raise UserError("No language model found. Put a GGUF file into models\\llm (or run install.bat).")
        wanted = self.store.settings.model
        names = [m["file"] for m in models]
        name = wanted if wanted in names else names[0]
        if name != wanted:
            self.store.settings.model = name
            self.store.save()
        return self.paths.llm_models / name

    def _ensure_llm(self) -> None:
        path = self._model_path()
        ctx = self.store.settings.context
        if self.c.server.running and self._loaded_model == path.name and self._loaded_context == ctx:
            return
        self._set_state(State.LOADING, f"Loading model {path.name}…")
        self._loaded_model = None
        try:
            self.c.server.start(ServerConfig(model=path, context=ctx))
        except Exception as exc:
            raise UserError(f"Could not load {path.name}: {exc}") from exc
        self._loaded_model, self._loaded_context = path.name, ctx

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
        self._turn(text, self.last_language, "voice", stt_ms=round(tr.seconds * 1000))

    def _handle_typed(self, text: str) -> None:
        # Typed text has no Whisper language; guess it so the reply voice matches.
        language = guess_language(text)
        if language:
            self.last_language = language
        self._turn(text, self.last_language, "text")

    def _turn(self, text: str, language: str | None, source: str, stt_ms: int | None = None) -> None:
        if self.fatal_error or not self.c.server.running:
            raise UserError(self.fatal_error or "The language model is not loaded.")
        self._set_state(State.THINKING, "")
        user = {"role": "user", "text": text, "language": language, "source": source}
        self.transcript.append(user)
        self._emit({"type": "user_message", **user, "stt_ms": stt_ms})

        voice = self.conversation.voice_for(language)
        reply: dict[str, Any] = {"role": "assistant", "segments": [],
                                 "voice": self._display_voice(voice)}
        self.transcript.append(reply)
        self._emit({"type": "assistant_start", "voice": reply["voice"]})

        self._playback_done.clear()
        chunks = 0
        audio_seconds = 0.0
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
                chunks += 1
                audio_seconds += len(event.audio) / event.sample_rate
            elif isinstance(event, TurnError):
                self._emit({"type": "error", "message": event.message})
            elif isinstance(event, TurnDone):
                m = event.metrics
                self._emit({"type": "turn_done", "audio_chunks": chunks, "metrics": m.summary(),
                            "first_audio_ms": None if m.first_audio is None else round(m.first_audio * 1000)})
        if chunks:
            # Mic stays muted until the browser has played everything (spec §9 step 5).
            if not self._playback_done.wait(timeout=audio_seconds + PLAYBACK_WAIT_MARGIN_S):
                log.warning("Browser did not report the end of playback; continuing.")
            time.sleep(POST_PLAYBACK_MUTE_S)
        self._enter_idle()

    def _display_voice(self, ref: str | None) -> str | None:
        if ref is None:
            return None
        if ref.startswith("builtin:"):
            return ref
        return f"custom:{Path(ref).name}"

    def _clear_conversation(self) -> None:
        self.conversation.history.clear()
        self.transcript.clear()
        self._emit({"type": "cleared"})

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
        return info

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
            if self.state is State.LOADING:
                self._enter_idle()

    def _apply_changes(self, changed: dict[str, Any]) -> None:
        s = self.store.settings
        if "end_silence_ms" in changed:
            self.segmenter.config.end_silence_ms = s.end_silence_ms
        if "vad_threshold" in changed:
            self.segmenter.config.threshold = s.vad_threshold
        if "match_language" in changed:
            self.conversation.config.match_language = s.match_language
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
        if "model" in changed or "context" in changed:
            self._ensure_llm()
            self.fatal_error = None
        if changed.keys() - {"show_actions"}:
            self._enter_idle()


def _append_segment(segments: list[dict[str, str]], kind: str, text: str) -> None:
    """Store streamed segments compactly: consecutive speech fragments are merged."""
    if segments and kind == "speech" and segments[-1]["kind"] == "speech":
        segments[-1]["text"] += text
    else:
        segments.append({"kind": kind, "text": text})


def _quant(file_name: str) -> str:
    m = re.search(r"(IQ\d_[A-Z0-9]+|Q\d_K_[SML]|Q\d_K|Q\d_\d|Q\d|BF16|F16|F32)", file_name, re.IGNORECASE)
    return m.group(1).upper() if m else "?"
