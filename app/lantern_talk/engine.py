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
from .characters import TEMPLATES, TEXT_COLORS, Character, CharacterError, CharacterStore, slugify
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
from .group import (
    DEFAULT_STORY_RULES, IMPULSE_EVERY, IMPULSES, MAX_CAST, MAX_RULES, NamePrefixFilter, build_group_prompt, cast_title,
    director_lines, group_key, group_messages, mentioned, next_speaker, persona_text, session_stem, stop_sequences,
)
from .pipeline import Conversation, ConversationConfig, SegmentEvent, SpeechAudio, TurnDone, TurnError
from .prompts import CONTINUE_TEXT, DEFAULT_ASSISTANT_PERSONA, LANGUAGE_NAMES, director_context, language_name, user_context
from .recorder import CaptureBuffer, SessionItem, SessionRecorder, list_sessions, session_title, spoken_text
from .resample import StreamingResampler
from .sanitizer import split_segments
from .settings import Settings, SettingsStore
from .stt import WhisperSTT
from .textlang import guess_language
from .tts import PRECISIONS, OmniVoiceTTS, VoiceLibrary
from .turn import MODEL_FILE as SMART_TURN_FILE
from .turn import SmartTurn
from .vad import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, SegmenterConfig, SileroVAD, Utterance, UtteranceSegmenter

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
FALLBACK_LANGUAGE = "en"          # Automatic, before a language is known (or when detection is unsure)
# Conversation languages offered in the side panel (Whisper and the language models know many more).
CONVERSATION_LANGUAGES = ("en", "de", "es", "fr", "it", "nl", "pt", "pl", "tr", "ru")
MAX_VOICE_UPLOAD_BYTES = 50 * 2**20
TALK_MODES = ("auto", "smart", "ptt")
CHARACTER_FILE_FORMAT = "local-lantern-talk-character"
MAX_COLLECT_S = 120.0             # push-to-talk / "Hold on" end by themselves after this
GROUP_MODES = ("chat", "story")
STORY_AUTO_PAUSE_LINES = 50       # story mode pauses by itself after this many lines without you
MAX_RECENT_GROUPS = 6
SESSION_TTS_RATE = 24_000         # OmniVoice's output rate
ENGINE_OWNED_SETTINGS = {"group", "group_mode", "recent_groups", "character"}  # changed by their own commands

DRAFT_SYSTEM = "You write character sheets for a local voice chat app. You answer with JSON only."
DRAFT_PROMPT = """Write the instructions for a character of a voice chat app from this profile.

Name: {name}
Kind: {kind}
Profile, in the user's own words (it may be in another language):
{profile}

Answer with one JSON object with these keys:
"system_prompt": the instructions, in English, addressed to the character in the second person ("You are {name}, ..."): age, background, personality, what they know and what they don't, the setting{extra}. 120 to 220 words. Do not write rules about formatting, markdown or speech output.
"speaking_style": how {name} speaks (tone, word choice, sentence length, habits), in English, one or two sentences.
"group_style": how {name} behaves in a group with other characters, in English, one or two sentences.
"persona": a short description for the character list, in English, at most twelve words.
"greeting": what {name} says first when the user arrives, in {language}, one to three spoken sentences{action}."""
DRAFT_KEYS = ("system_prompt", "speaking_style", "group_style", "persona", "greeting")

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
    turn: Any = None   # Smart Turn model (talk mode "smart")


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
        voices=voices, server=server, llm=LLMClient(server), turn=SmartTurn(paths.vad_models / SMART_TURN_FILE),
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
        self.last_language: str | None = None if s.language == "auto" else s.language
        self.transcript: list[dict[str, Any]] = []
        self.conversation = Conversation(
            self.c.llm, self.c.tts if s.voice_output else None, self.c.voices,
            ConversationConfig(persona=persona, voice=None, match_language=s.match_language),
        )
        self.conversation.config.user = user_context(s.user_name, s.user_about)
        self.images = ImageStore(paths.images)
        self._fresh_images: dict[str, float] = {}  # uploaded, maybe not saved yet (editor still open)
        self.characters = CharacterStore(paths.characters, paths.config / "default_characters", self.images)
        self.characters.load()
        self.memory = MemoryStore(paths.memory / "memory.db")
        self.character: Character | None = None
        self.resume: dict[str, Any] | None = None  # an unfinished chat that was loaded from memory
        self.pending_note = ""                     # director's note for the next reply
        self.group_notes: dict[str, str] = {}      # group chats: one-time notes per character id ("" = anyone)
        self.director_draft: tuple[str, str] = ("", "")  # (text, target) still in the director's field
        self.story_playing = False                 # story mode: the characters go on by themselves
        self._story_gen = 0                        # Play / Pause bump this; stale "story_next" commands are dropped
        self._story_scheduled = False              # a "story_next" command is already waiting
        self._story_lines = 0                      # lines since you last said or directed something
        self._lines_since_note = 0                 # story mode: lines since the last director's note or nudge
        self.next_speaker: str | None = None       # group chats: who you picked to speak next
        self._finishing: str | None = None         # End chat: the recording being put together
        self.replaying = False                     # a recorded message is playing in the browser
        self._msg_seq = 0                          # ids of transcript items (for replay buttons)
        self._replay_lock = threading.Lock()
        self.capture = CaptureBuffer(keep_s=MAX_COLLECT_S + 10)  # full-quality microphone audio (spec §12.1)
        self._collect: dict[str, Any] | None = None  # push-to-talk / "Hold on": audio collected until "send"
        self.recorder = SessionRecorder(
            paths.sessions, paths.ffmpeg, self._on_recorded,
            gap_ms=lambda: self.store.settings.session_gap_ms,
            bitrate_kbps=lambda: self.store.settings.session_bitrate_kbps, on_built=self._on_session_built)
        # Chats that were closed with the app keep their single files; the others get their whole MP3.
        self.recorder.finish_pending(self.memory.meta_values("session"))
        if self.characters.list():
            self._activate_character(s.character)
        self.segmenter = UtteranceSegmenter(
            self.c.vad, SegmenterConfig(threshold=s.vad_threshold, end_silence_ms=s.end_silence_ms))
        if s.talk_mode not in TALK_MODES:
            s.talk_mode = "auto"
        self._resampler: StreamingResampler | None = None
        self._reset_audio = threading.Event()
        self._playback_done = threading.Event()
        self._state_lock = threading.Lock()
        self._commands: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._audio: queue.Queue[tuple[np.ndarray, int] | str | None] = queue.Queue(maxsize=500)
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

    # -- director's notes ---------------------------------------------------------------------
    def set_pending_note(self, text: str, target: str = "") -> None:
        """A director's note for the next reply, however it is started (typed, spoken, Continue).

        In a group chat ``target`` is the character it is meant for ("" = whoever speaks next).
        """
        self._commands.put(("pending_note", (" ".join(text.split())[:300], target)))

    def set_director_draft(self, text: str, target: str = "") -> None:
        """What is in the director's field right now: it goes with your next spoken or typed message."""
        self.director_draft = (" ".join(str(text).split())[:300], str(target))

    def _take_draft(self) -> tuple[str, str]:
        """The note in the director's field, for the message that is being answered now (the field empties)."""
        text, target = self.director_draft
        self.director_draft = ("", "")
        if text:
            self._emit({"type": "director_draft_used"})
        return text, target

    def pin_note(self, text: str, target: str = "") -> None:
        self._commands.put(("pin_note", (text, target)))

    def unpin_note(self, index: int) -> None:
        self._commands.put(("unpin_note", index))

    def continue_story(self, speaker: str = "") -> None:
        """The character goes on by itself (with the pending director's note, if any).

        In a group chat one character speaks: ``speaker``, or whoever is next.
        """
        self._commands.put(("continue", speaker))

    # -- group chats and story mode ---------------------------------------------------------------
    def set_group(self, ids: list[str]) -> None:
        """The other characters in the chat (up to three); [] = only the character picked on the left."""
        self._commands.put(("set_group", [str(i) for i in ids][:MAX_CAST]))

    def open_group(self, ids: list[str]) -> None:
        """Open a group chat again (from the recent groups): the first id is the one picked on the left."""
        self._commands.put(("open_group", [str(i) for i in ids][:MAX_CAST]))

    def set_group_mode(self, mode: str) -> None:
        self._commands.put(("group_mode", mode))

    def story_play(self) -> None:
        self._commands.put(("story_play", None))

    def story_pause(self) -> None:
        """Stop after the line that is being spoken (immediately, not queued behind the story)."""
        if self.story_playing:
            self.story_playing = False
            self._story_gen += 1
            self._emit_group()

    def set_next_speaker(self, char_id: str) -> None:
        self._commands.put(("next_speaker", char_id))

    def set_premise(self, text: str) -> None:
        self._commands.put(("premise", text))

    def new_story(self, text: str) -> None:
        """Start the story over with this setup: the group forgets the old story (recordings stay)."""
        self._commands.put(("new_story", text))

    def draft_character(self, data: dict[str, Any]) -> None:
        """Let the loaded model write instructions, speaking style and greeting from a profile."""
        self._commands.put(("draft_character", data))

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

    # -- push-to-talk and "Hold on" ------------------------------------------------------------
    def talk_start(self) -> None:
        """Collect everything said from now on (incl. a sentence in progress) until :meth:`talk_end`."""
        self._audio_marker("talk_start")

    def talk_end(self, send: bool = True) -> None:
        self._audio_marker("talk_end" if send else "talk_cancel")

    def _audio_marker(self, marker: str) -> None:
        try:
            self._audio.put_nowait(marker)  # in order with the microphone audio before it
        except queue.Full:
            log.warning("Audio queue full; dropping %s", marker)

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

    def edit_message(self, msg_id: int, text: str) -> None:
        """Change the text of a message; a character's reply is spoken again in the recording."""
        text = text.strip()
        if text:
            self._commands.put(("edit_message", (msg_id, text)))

    def stop_reply(self) -> None:
        """Stop the reply that is being written or spoken (not queued: the worker is busy with the turn)."""
        self.story_pause()  # a story stops with it
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
            "director": self.director_view(),
            "group": self.group_view(),
            "templates": TEMPLATES,
            "story_rules_default": DEFAULT_STORY_RULES,
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

    # -- character files (export / import) ---------------------------------------------------
    def export_character(self, char_id: str) -> tuple[str, bytes]:
        """One character as a file: settings, look and pictures; never its memory or a cloned voice."""
        import base64
        import json

        c = self.characters.get(char_id)
        if c is None:
            raise UserError("This character no longer exists.")
        data = c.to_dict()
        data.pop("id", None)
        if data["voice"].startswith("custom:"):
            data["voice"] = ""  # a cloned voice belongs to its speaker and is not passed on
        pictures = {}
        for key in ("avatar", "background"):
            path = self.images.path(data[key])
            if path is not None:
                pictures[key] = {"file": path.suffix.lstrip("."), "data": base64.b64encode(path.read_bytes()).decode()}
            data[key] = ""
        payload = {"format": CHARACTER_FILE_FORMAT, "version": 1, "app": __version__, "character": data,
                   "pictures": pictures}
        name = re.sub(r"[^\w\- ]+", "", c.name).strip() or "character"
        return f"{name}.llt-character.json", json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")

    def import_character(self, raw: bytes) -> dict[str, str]:
        """Add a character from an exported file (called from the web server thread)."""
        import base64
        import binascii
        import json

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise UserError("This is not a Local Lantern Talk character file.") from exc
        if not isinstance(payload, dict) or payload.get("format") != CHARACTER_FILE_FORMAT \
                or not isinstance(payload.get("character"), dict):
            raise UserError("This is not a Local Lantern Talk character file.")
        data = dict(payload["character"])
        data.pop("id", None)
        if data.get("voice") and str(data["voice"]).startswith("custom:"):
            data["voice"] = ""
        if data.get("voice"):
            try:
                self._voice_ref(str(data["voice"]))
            except UserError:
                data["voice"] = ""
        if not data.get("voice"):
            data["voice"] = DEFAULT_VOICE
        for key in ("avatar", "background"):
            data[key] = ""
            picture = (payload.get("pictures") or {}).get(key)
            if isinstance(picture, dict) and picture.get("data"):
                try:
                    data[key] = self.add_image("avatar" if key == "avatar" else "background",
                                               base64.b64decode(picture["data"], validate=True))["name"]
                except (binascii.Error, ValueError, UserError):
                    log.warning("Skipping an unusable %s in an imported character", key)
        if self.characters.get(slugify(str(data.get("name", "")))) is not None:
            data["name"] = f"{str(data.get('name', '')).strip()} (imported)"[:60]
        self._commands.put(("import_character", data))
        return {"name": str(data.get("name", ""))}

    def _handle_import_character(self, data: dict[str, Any]) -> None:
        try:
            char = self.characters.save(data)
        except CharacterError as exc:
            raise UserError(f"The character could not be imported: {exc}") from exc
        self._emit_character_state()
        self._emit({"type": "notice", "message": f"{char.name} was imported. Pick them on the left to talk to them."})

    def _handle_draft_character(self, data: dict[str, Any]) -> None:
        """The loaded model drafts a character from its profile; the editor shows it before anything is saved."""
        import json

        name = " ".join(str(data.get("name") or "").split())[:60]
        profile = str(data.get("profile") or "").strip()[:2000]
        roleplay = data.get("kind") == "roleplay"
        if not name or not profile:
            raise UserError("Write a name and a profile first; the instructions are written from them.")
        if not self.c.server.running:
            raise UserError(self.fatal_error or "Load a language model first; it writes the instructions.")
        self._set_state(State.THINKING, f"Writing the instructions for {name}…")
        language = language_name(self._fixed_language() or self.last_language or FALLBACK_LANGUAGE) or "English"
        extra = (", who the user is in this story, and that they stay in character and drive the story forward"
                 if roleplay else ", what they help the user with, and that they say honestly when they don't "
                 "know something")
        text = self._llm_text(DRAFT_SYSTEM, DRAFT_PROMPT.format(
            name=name, kind="roleplay character" if roleplay else "assistant", profile=profile, extra=extra,
            language=language, action=", it may start with a short action in *asterisks*" if roleplay else ""),
            max_tokens=1200)
        draft: dict[str, str] = {}
        match = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            parsed = json.loads(match.group(0)) if match else {}
        except ValueError:
            parsed = {}
        if isinstance(parsed, dict):
            draft = {k: " ".join(str(parsed[k]).split()) for k in DRAFT_KEYS if isinstance(parsed.get(k), str)}
        self._enter_idle()
        if not draft.get("system_prompt"):
            raise UserError("The model did not write usable instructions. Try again, or try another model.")
        self._emit({"type": "character_draft", "draft": draft})

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
            raise UserError("The file is too large. A voice clip only needs 10-13 seconds of speech.")
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
                # Listening (again): forget the audio queued meanwhile. Button presses still count.
                self._reset_audio.clear()
                self._stop_collecting()
                self.segmenter.reset()
                self.capture.reset()
                if self._resampler is not None:
                    self._resampler.reset()
                if isinstance(item, str):
                    self._talk_marker(item)
                self._drain_audio()
                continue
            if isinstance(item, str):
                try:
                    self._talk_marker(item)
                except Exception:  # noqa: BLE001
                    log.exception("Push-to-talk failed")
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
                audio16 = self._resampler.feed(samples)
                if self._collect is not None:
                    self._collect_feed(audio16)
                    continue
                if self.store.settings.talk_mode == "ptt":
                    self.segmenter.skip(audio16)  # push-to-talk: nothing is heard until the button is held
                    continue
                for utt in self.segmenter.feed(audio16):
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
                item = self._audio.get_nowait()
            except queue.Empty:
                return
            if item is None:
                self._audio.put(None)  # keep the stop signal
                return
            if isinstance(item, str):
                self._talk_marker(item)  # a button press is never lost

    # -- push-to-talk / "Hold on" (audio thread) ----------------------------------------------
    def _talk_marker(self, marker: str) -> None:
        if marker == "talk_start":
            if self._collect is None and self.state is State.LISTENING:
                prefill = self.segmenter.take_speech()  # a sentence in progress belongs to it
                self._collect = {"chunks": [prefill], "start": self.segmenter.position - len(prefill),
                                 "samples": len(prefill), "pending": np.zeros(0, np.float32),
                                 "speech_frames": len(prefill) // FRAME_SAMPLES}
                self._emit({"type": "talk", "collecting": True})
        elif marker in ("talk_end", "talk_cancel"):
            self._finish_collecting(send=marker == "talk_end")

    def _collect_feed(self, audio16: np.ndarray) -> None:
        col = self._collect
        assert col is not None
        col["chunks"].append(audio16)
        col["samples"] += len(audio16)
        self.segmenter.skip(audio16)
        # Count speech frames, so that a press without speaking sends nothing.
        pending = np.concatenate([col["pending"], audio16])
        while len(pending) >= FRAME_SAMPLES:
            frame, pending = pending[:FRAME_SAMPLES], pending[FRAME_SAMPLES:]
            if self.c.vad.speech_probability(frame) >= self.segmenter.config.threshold:
                col["speech_frames"] += 1
        col["pending"] = pending
        if col["samples"] >= MAX_COLLECT_S * SAMPLE_RATE:
            self._finish_collecting(send=True)

    def _stop_collecting(self) -> None:
        if self._collect is not None:
            self._collect = None
            self._emit({"type": "talk", "collecting": False})

    def _finish_collecting(self, send: bool) -> None:
        col = self._collect
        self._stop_collecting()
        if col is None or not send:
            return
        audio = np.concatenate(col["chunks"]) if col["chunks"] else np.zeros(0, np.float32)
        if col["speech_frames"] * FRAME_MS < self.segmenter.config.min_speech_ms or not len(audio):
            self._emit({"type": "notice", "message": "Nothing was heard, so nothing was sent."})
            return
        start = max(0, col["start"])
        utt = Utterance(audio=audio, duration_s=len(audio) / SAMPLE_RATE, start=start, end=start + len(audio),
                        collected=True)
        utt.raw = self.capture.slice_16k(utt.start, utt.end)
        utt.raw_sr = self.capture.sample_rate
        self.c.vad.reset()
        self._set_state(State.THINKING, "Transcribing…")
        self._commands.put(("utterance", utt))

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
            "pending_note": self._handle_pending_note,
            "pin_note": self._handle_pin_note,
            "unpin_note": self._handle_unpin_note,
            "continue": self._handle_continue,
            "import_character": self._handle_import_character,
            "set_group": self._handle_set_group,
            "open_group": self._handle_open_group,
            "group_mode": self._handle_group_mode,
            "story_play": self._handle_story_play,
            "story_next": self._handle_story_next,
            "next_speaker": self._handle_next_speaker,
            "premise": self._handle_premise,
            "new_story": self._handle_new_story,
            "draft_character": self._handle_draft_character,
            "edit_message": lambda arg: self._handle_edit_message(*arg),
        }
        while True:
            name, arg = self._commands.get()
            if name == "stop":
                return
            try:
                handlers[name](arg)
            except (UserError, ModelError) as exc:
                self._emit({"type": "error", "message": str(exc)})
                self.story_pause()
                self._enter_idle()
            except Exception as exc:  # noqa: BLE001 - details go to the log
                log.exception("Command %s failed", name)
                self._emit({"type": "error", "message": describe_error(exc)})
                self.story_pause()
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
        if self.store.settings.talk_mode == "smart":
            try:
                self._apply_talk_mode()
            except UserError as exc:
                log.warning("Smart talk mode is not available: %s", exc)
                self.store.settings.talk_mode = "auto"
                self.store.save()
                self._emit({"type": "notice", "message": f"{exc} Talk mode is Automatic for now."})

    def _apply_talk_mode(self) -> None:
        """Smart mode asks the Smart Turn model at every pause; the other modes do not need it."""
        turn = self.c.turn
        if self.store.settings.talk_mode != "smart" or turn is None:
            self.segmenter.turn_check = None
            return
        if not turn.available:
            self._download_smart_turn()
        self._set_state(State.LOADING, "Loading the Smart turn model…")
        try:
            turn.load()
        except Exception as exc:  # noqa: BLE001
            log.exception("Smart Turn could not be loaded")
            raise UserError(f"The Smart turn model could not be loaded ({exc}).") from exc
        self.segmenter.turn_check = turn.is_complete
        self.segmenter.config.smart_max_silence_ms = max(2500, self.store.settings.end_silence_ms * 2)

    def _download_smart_turn(self) -> None:
        """Smart Turn ships with new installations; older ones fetch the 9 MB model once."""
        import json

        spec = json.loads((self.paths.config / "runtime_pins.json").read_text(encoding="utf-8"))["models"]["smart_turn"]
        self._set_state(State.LOADING, f"Downloading the Smart turn model ({spec['download_mb']} MB)…")
        try:
            from .setup_models import fetch_smart_turn

            fetch_smart_turn(self.paths, spec)
        except Exception as exc:  # noqa: BLE001
            log.exception("Smart Turn download failed")
            raise UserError("The Smart turn model could not be downloaded. Check the internet connection and "
                            "try again, or use Automatic or Push-to-talk.") from exc

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
    def _fixed_language(self) -> str | None:
        """The conversation language from the side panel; None = Automatic (follow the speaker)."""
        language = self.store.settings.language
        return None if language == "auto" else language

    def _handle_utterance(self, utt: Utterance) -> None:
        self._set_state(State.THINKING, "Transcribing…")
        fixed = self._fixed_language()
        kwargs: dict[str, Any] = {"long_pauses": True} if utt.collected else {}
        if fixed:
            kwargs["language"] = fixed  # Whisper does not have to guess (and cannot guess wrong)
        tr = self.c.stt.transcribe(utt.audio, **kwargs)
        text = tr.text.strip()
        if not text or not text.strip(" .,!?…"):
            self._enter_idle()
            return
        if fixed:
            self.last_language = fixed
        elif tr.language_probability >= LANGUAGE_CONFIDENCE:
            self.last_language = tr.language
        elif self.last_language is None:
            self.last_language = FALLBACK_LANGUAGE  # an unsure first guess (noise, a short word) is not trusted
        raw = utt.raw if utt.raw is not None and len(utt.raw) else utt.audio
        self._turn(text, self.last_language, "voice", stt_ms=round(tr.seconds * 1000),
                   audio=raw, audio_sr=utt.raw_sr if raw is utt.raw else SAMPLE_RATE)

    def _handle_typed(self, text: str) -> None:
        # Typed text has no Whisper language; guess it so the reply voice matches.
        language = self._fixed_language() or guess_language(text)
        if language:
            self.last_language = language
        elif self.last_language is None:
            self.last_language = FALLBACK_LANGUAGE
        self._turn(text, self.last_language, "text")

    def _turn(self, text: str, language: str | None, source: str, stt_ms: int | None = None,
              audio: np.ndarray | None = None, audio_sr: int = 0, user: dict[str, Any] | None = None) -> None:
        """One exchange. ``user`` is an existing message to answer again (Regenerate).

        ``source`` "director": the user only gave a director's note; the character goes on by itself.
        """
        self._recover_llm()
        if self.fatal_error or not self.c.server.running:
            raise UserError(self.fatal_error or "The language model is not loaded.")
        if self.in_group:
            self._group_turn(text, language, source, stt_ms, audio, audio_sr)
            return
        if source == "director" and user is None and not self.pending_note and not self.transcript:
            raise UserError("Nothing to continue yet.")
        self._set_state(State.THINKING, "")
        self.resume = None  # talking on continues the chat
        recording = self._ensure_session()
        if user is None:
            note, self.pending_note = self.pending_note, ""
            if source in ("voice", "text"):
                draft = self._take_draft()[0]
                note = " ".join(n for n in (note, draft) if n)
            user = {"id": self._new_id(), "role": "user", "text": text, "language": language, "source": source}
            if note:
                user["note"] = note
            self.transcript.append(user)
            self._emit({"type": "user_message", **user, "stt_ms": stt_ms})
            self._emit_director()
            if recording and source != "director":  # encoded in the background while the model answers
                tags = ["spoken", language] if source == "voice" and language else [
                    "spoken" if source == "voice" else "typed"]
                self.recorder.add(SessionItem(user["id"], "user", self._user_label(), text, tags,
                                              audio if source == "voice" else None, audio_sr or SAMPLE_RATE))

        voice = self.conversation.voice_for(language)
        reply: dict[str, Any] = {"id": self._new_id(), "role": "assistant", "segments": [],
                                 "voice": self._display_voice(voice)}
        self.transcript.append(reply)
        self._emit({"type": "assistant_start", "id": reply["id"], "voice": reply["voice"]})
        llm_text = text or CONTINUE_TEXT  # a director's note alone: "(continue)"
        reply_text, spoken, spoken_sr = self._play(reply, self.conversation.run_turn(llm_text, language,
                                                                                     user.get("note", "")))
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

    def _play(self, reply: dict[str, Any], events: Any) -> tuple[str, list[np.ndarray], int]:
        """Show and send one reply as it is written and spoken; wait until the browser has played it.

        Returns the reply text and the spoken audio (for the session recording).
        """
        spoken: list[np.ndarray] = []
        spoken_sr = 24_000
        self._playback_done.clear()
        chunks = 0
        audio_seconds = 0.0
        reply_text = ""
        for event in events:
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
        return reply_text, spoken, spoken_sr

    # -- group chats and story mode ------------------------------------------------------------
    @property
    def members(self) -> list[Character]:
        """Everyone in the chat: the character picked on the left first, then the others."""
        if self.character is None:
            return []
        others = [self.characters.get(i) for i in self.store.settings.group if i != self.character.id]
        return [self.character, *[c for c in others if c is not None]][:MAX_CAST]

    @property
    def in_group(self) -> bool:
        return len(self.members) > 1

    def _story(self) -> bool:
        return self.in_group and self.store.settings.group_mode == "story"

    def _mem_key(self) -> str:
        """Where the chat is remembered: the character's id, or one shared key for a group."""
        members = self.members
        if not members:
            return "chat"
        return group_key([m.id for m in members]) if len(members) > 1 else members[0].id

    def _mem_on(self) -> bool:
        """Memory is on when every character in the chat has it on."""
        members = self.members
        return bool(members) and all(m.memory_enabled for m in members)

    def _mem_title(self) -> str:
        return cast_title([m.name for m in self.members]) or "Assistant"

    def _names(self) -> dict[str, str]:
        return {m.id: m.name for m in self.members}

    def group_view(self) -> dict[str, Any]:
        s = self.store.settings
        return {"members": [m.id for m in self.members], "mode": s.group_mode, "playing": self.story_playing,
                "next": self.next_speaker, "premise": self._group_meta("premise") if self.in_group else "",
                "recent": [g for g in s.recent_groups if all(self.characters.get(i) for i in g)]}

    def _emit_group(self) -> None:
        self._emit({"type": "group", **self.group_view()})

    def _group_meta(self, key: str) -> str:
        return self.memory.get_meta(self._mem_key(), key, "")

    def _group_pins(self) -> list[dict[str, str]]:
        import json

        try:
            pins = json.loads(self._group_meta("pins") or "[]")
        except ValueError:
            pins = []
        return [p for p in pins if isinstance(p, dict) and p.get("text")] if isinstance(pins, list) else []

    def _member(self, char_id: str) -> Character:
        member = next((m for m in self.members if m.id == char_id), None)
        if member is None:
            raise UserError("This character is not in the chat.")
        return member

    def _member_voice(self, member: Character, language: str | None) -> str | None:
        """The voice ``member`` speaks with (None: text only)."""
        if not member.voice_output:
            return None
        try:
            ref = self._voice_ref(member.voice or DEFAULT_VOICE)
        except UserError:
            ref = DEFAULT_VOICE
        return self.conversation.voice_for(language, ref)

    def _prepare_group_voices(self) -> None:
        """Every voice of the group is ready before the first line, so switching speakers costs nothing."""
        if not (self.store.settings.voice_output and self.c.tts.loaded):
            return
        for m in self.members[1:]:
            ref = self._member_voice(m, self._fixed_language() or self.last_language)
            if ref is not None:
                self._set_state(State.LOADING, f"Preparing the voice of {m.name}…")
                self.c.voices.prompt_for(ref)

    def _group_turn(self, text: str, language: str | None, source: str, stt_ms: int | None = None,
                    audio: np.ndarray | None = None, audio_sr: int = 0, speaker: str = "") -> None:
        """You said something (``text``) or the story goes on (``source`` "director"): one or more lines.

        In a group chat the character you address answers (or whoever is next), and anyone that
        reply addresses may answer too. In story mode one line follows; with Play the story goes on.
        """
        self._set_state(State.THINKING, "")
        self.resume = None
        recording = self._ensure_session()
        members = self.members
        if source != "director" and text:
            draft, target = self._take_draft()
            if draft:
                target = target if target in {m.id for m in members} else ""
                self.group_notes[target] = " ".join(n for n in (self.group_notes.get(target, ""), draft) if n)
            user: dict[str, Any] = {"id": self._new_id(), "role": "user", "text": text, "language": language,
                                    "source": source, "in_history": True}
            self.transcript.append(user)
            self._emit({"type": "user_message", **user, "stt_ms": stt_ms})
            if recording:
                tags = ["spoken", language] if source == "voice" and language else [
                    "spoken" if source == "voice" else "typed"]
                self.recorder.add(SessionItem(user["id"], "user", self._user_label(), text, tags,
                                              audio if source == "voice" else None, audio_sr or SAMPLE_RATE))
            self.conversation.history.append(ChatMessage("user", text))
            self._store_line(user, "user", text, language, source)
            self._story_lines = 0
            queue_ = mentioned(text, members) or [speaker or self.next_speaker or
                                                  next_speaker(members, self.conversation.history)]
        else:
            queue_ = [speaker or self.next_speaker or next_speaker(members, self.conversation.history)]
        self.next_speaker = None
        spoken_now: set[str] = set()
        while queue_ and len(spoken_now) < len(members):
            sid = queue_.pop(0)
            if sid in spoken_now or sid not in {m.id for m in members}:
                continue
            reply_text = self._group_line(self._member(sid), language, recording)
            spoken_now.add(sid)
            if not reply_text or self.conversation.cancel.is_set():
                break
            if source != "director" and not self._story():
                queue_ += [cid for cid in mentioned(reply_text, members, [sid]) if cid not in spoken_now]
        self._emit_group()
        self._enter_idle()
        self._maintain_memory()
        self._emit_editable()
        self._schedule_story()

    def _group_line(self, member: Character, language: str | None, recording: bool) -> str:
        """One line of ``member``, written and spoken; returns its text."""
        s = self.store.settings
        story = self._story()
        note = " ".join(n for n in (self.group_notes.pop("", ""), self.group_notes.pop(member.id, "")) if n)
        nudge = ""
        if story and not note:
            self._lines_since_note += 1
            if self._lines_since_note >= IMPULSE_EVERY:
                import random

                nudge, self._lines_since_note = random.choice(IMPULSES), 0
        elif note:
            self._lines_since_note = 0
        names = self._names()
        system = build_group_prompt(
            member, [m for m in self.members if m.id != member.id], default_persona=DEFAULT_ASSISTANT_PERSONA,
            story=story, story_rules=s.story_rules, premise=self._group_meta("premise"),
            user=self.conversation.config.user, user_name=s.user_name, memory=self.conversation.config.memory,
            director=director_context(director_lines(self._group_pins(), member.id), note or nudge),
            reply_language=language)
        messages = group_messages(member.id, system, self.conversation.history[-200:], names, s.user_name)
        sampling = SamplingConfig(temperature=0.9 if story else 0.8, max_tokens=400, dry_multiplier=0.8,
                                  stop=stop_sequences(member.id, names, s.user_name))
        voice = self._member_voice(member, language)
        reply: dict[str, Any] = {"id": self._new_id(), "role": "assistant", "speaker": member.id, "segments": [],
                                 "voice": self._display_voice(voice)}
        if note:
            reply["note"] = note
        self.transcript.append(reply)
        self._emit({"type": "assistant_start", "id": reply["id"], "voice": reply["voice"], "speaker": member.id,
                    "note": note or None})
        self._emit_director()
        self._set_state(State.THINKING, f"{member.name} is thinking…")
        reply_text, spoken, spoken_sr = self._play(
            reply, self.conversation.stream(messages, voice, language, sampling, NamePrefixFilter(member.name)))
        if not reply_text:
            return ""
        if recording:
            tags = [self._voice_name(reply["voice"])] if spoken and reply["voice"] else ["text only"]
            self.recorder.add(SessionItem(
                reply["id"], "assistant", member.name, spoken_text(reply["segments"]), tags,
                np.concatenate(spoken) if spoken else None, spoken_sr, segments=reply["segments"]))
        self.conversation.history.append(ChatMessage("assistant", reply_text, name=member.id))
        reply["in_history"] = True
        self._store_line(reply, "assistant", reply_text, language, None, note=note, speaker=member.id)
        self._story_lines += 1
        return reply_text

    def _store_line(self, item: dict[str, Any], role: str, text: str, language: str | None, source: str | None,
                    note: str = "", speaker: str | None = None) -> None:
        """A group line goes into the group's memory right away (or the history is trimmed without memory)."""
        if not self._mem_on():
            self._trim_history()
            return
        db_id = self.memory.add_message(self._mem_key(), role, text, language, source,
                                        item.get("voice") if role == "assistant" else None, note=note or None,
                                        speaker=speaker)
        with self._replay_lock:  # the recorder may already have finished this message
            item["db_id"] = db_id
            if item.get("session"):
                self.memory.set_replay(db_id, item["session"], item["seq"], item.get("part"), item.get("duration"))

    def _schedule_story(self) -> None:
        """Story mode with Play: the next line is queued behind whatever you did meanwhile."""
        if not (self.story_playing and self._story()) or self._story_scheduled:
            return
        if self._story_lines >= STORY_AUTO_PAUSE_LINES:
            self.story_pause()
            self._emit({"type": "notice", "message": f"The story paused after {STORY_AUTO_PAUSE_LINES} lines "
                                                     "without you. Press Play to go on."})
            return
        self._story_scheduled = True
        self._commands.put(("story_next", self._story_gen))

    def _handle_story_next(self, gen: int) -> None:
        self._story_scheduled = False
        if gen != self._story_gen or not self.story_playing or not self._story():
            return
        self._turn("", self._fixed_language() or self.last_language or FALLBACK_LANGUAGE, "director")

    def _handle_story_play(self, _: Any = None) -> None:
        if not self.in_group:
            raise UserError("Story mode needs at least two characters. Add someone to the chat first.")
        if not self.c.server.running:
            raise UserError(self.fatal_error or "The language model is not loaded.")
        self.store.settings.group_mode = "story"
        self.store.save()
        self.story_playing = True
        self._story_gen += 1
        self._story_lines = 0
        self._emit_group()
        self._schedule_story()

    def _handle_continue(self, speaker: str = "") -> None:
        language = self._fixed_language() or self.last_language or FALLBACK_LANGUAGE
        if self._story() and self.story_playing:  # the story goes on by itself; the note goes into its next line
            if speaker:
                self.next_speaker = self._member(speaker).id
                self._emit_group()
            return
        if self.in_group:
            if speaker:
                self._member(speaker)
            self._turn("", language, "director") if not speaker else self._continue_as(speaker, language)
        else:
            self._turn("", language, "director")

    def _continue_as(self, speaker: str, language: str) -> None:
        self._recover_llm()
        if self.fatal_error or not self.c.server.running:
            raise UserError(self.fatal_error or "The language model is not loaded.")
        self._group_turn("", language, "director", speaker=speaker)

    def _handle_next_speaker(self, char_id: str) -> None:
        self.next_speaker = self._member(char_id).id if char_id else None
        self._emit_group()

    def _handle_premise(self, text: str) -> None:
        if not self.in_group:
            raise UserError("A story setup belongs to a group chat.")
        self.memory.set_meta(self._mem_key(), "premise", str(text).strip()[:MAX_RULES])
        self._emit_group()

    def _handle_new_story(self, text: str) -> None:
        if not self.in_group:
            raise UserError("A story setup belongs to a group chat.")
        pins = self._group_meta("pins")  # pinned notes stay
        self._handle_reset_memory()
        key = self._mem_key()
        self.memory.set_meta(key, "premise", str(text).strip()[:MAX_RULES])
        if pins:
            self.memory.set_meta(key, "pins", pins)
        self._emit_group()
        self._emit_director()
        self._emit({"type": "notice", "message": "A fresh story with the new setup: the group has forgotten the old "
                                                 "one. Its recordings stay in the session folder."})
        self._enter_idle()

    def _handle_group_mode(self, mode: str) -> None:
        if mode not in GROUP_MODES:
            raise UserError(f"Unknown mode {mode}.")
        changed = mode != self.store.settings.group_mode
        self.store.settings.group_mode = mode
        self.store.save()
        if mode != "story":
            self.story_pause()
        elif changed and self.in_group and self.c.server.running:
            self._handle_story_play()  # switching to Story starts it; Pause stops it
            return
        self._emit_group()

    def _handle_set_group(self, ids: list[str]) -> None:
        c = self._memory_character()
        others: list[str] = []
        for i in ids:
            if i != c.id and i not in others:
                if self.characters.get(i) is None:
                    raise UserError("This character no longer exists.")
                others.append(i)
        if len(others) > MAX_CAST - 1:
            raise UserError(f"A chat has at most {MAX_CAST} characters.")
        self._change_cast(c.id, others)

    def _handle_open_group(self, ids: list[str]) -> None:
        if not ids or any(self.characters.get(i) is None for i in ids):
            raise UserError("Someone in this group no longer exists.")
        self._change_cast(ids[0], ids[1:])

    def _change_cast(self, lead: str, others: list[str]) -> None:
        """Switch to another set of characters: their own chat, memory and recording."""
        s = self.store.settings
        if lead == (self.character.id if self.character else None) and others == list(s.group):
            return
        self.story_pause()
        self._end_session()
        self.group_notes.clear()
        self.next_speaker = None
        self.pending_note = ""
        s.group = list(others)
        if others:
            cast = [lead, *others]
            s.recent_groups = [cast, *[g for g in s.recent_groups if sorted(g) != sorted(cast)]][:MAX_RECENT_GROUPS]
        self.store.save()
        if self.character is None or self.character.id != lead:
            self._activate_character(lead, keep_group=True)
            self._prepare_voice()
            self._ensure_llm()
        else:
            self._refresh_memory_context()
            self._load_history()
            self._sync_tts()
        self._prepare_group_voices()
        self._emit_character_state()
        self._enter_idle()

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
        """Start over with the same characters. The chat window is cleared at once; with memory on,
        the old talk is folded into the summary afterwards."""
        self.story_pause()
        self._finish_session()
        self.pending_note = ""
        self.group_notes.clear()
        self.next_speaker = None
        c = self.character
        old: list[Any] = []
        if c is not None and self._mem_on():
            old = self.memory.active_messages(self._mem_key())
            if old:
                self.memory.mark_summarized(self._mem_key(), old[-1].id)
        if c is not None:
            self._load_history()
        else:
            self.conversation.history.clear()
            self.transcript.clear()
        self._emit({"type": "cleared"})
        self._emit_character_state()
        worth_keeping = any(m.role == "user" for m in old) or (self.in_group and old)
        if worth_keeping and self.c.server.running:
            self._set_state(State.THINKING, "Saving the conversation to memory…")
            try:
                self._summarize(old)
                self._update_facts(old[-8:])
            except Exception:  # noqa: BLE001 - never block a new conversation
                log.exception("Could not summarize the conversation")
            self._refresh_memory_context()
            self._emit({"type": "memory", **self.memory_view()})
        self._enter_idle()

    # -- deleting, regenerating and editing the end of the chat ----------------------------------
    def _deletable_from(self) -> int | None:
        """UI id of the oldest message that can still be deleted (together with everything after it).

        The greeting stays, and so does what is already folded into the memory summary.
        """
        c = self.character
        active = {m.id for m in self.memory.active_messages(self._mem_key())} if c is not None and self._mem_on() \
            else None
        first = None
        for item in reversed(self.transcript):
            if item.get("greeting") or (active is not None and item.get("db_id") and item["db_id"] not in active):
                break
            first = item["id"]
        return first

    def director_view(self) -> dict[str, Any]:
        c = self.character
        if self.in_group:
            names = self._names()
            return {"group": True,
                    "pending": [{"target": t, "name": names.get(t, ""), "text": n} for t, n in self.group_notes.items()],
                    "pinned": [{"target": p.get("target", ""), "name": names.get(p.get("target", ""), ""),
                                "text": p["text"]} for p in self._group_pins()]}
        return {"group": False, "pending": self.pending_note, "pinned": list(c.director_notes) if c else []}

    def _emit_director(self) -> None:
        self._emit({"type": "director", **self.director_view()})

    def _handle_pending_note(self, arg: tuple[str, str]) -> None:
        text, target = arg
        if self.in_group:
            target = self._member(target).id if target else ""
            if text:
                self.group_notes[target] = text
            else:
                self.group_notes.pop(target, None)
        else:
            self.pending_note = text
        self._emit_director()

    def _handle_pin_note(self, arg: tuple[str, str]) -> None:
        import json

        from .characters import MAX_NOTES

        text, target = arg
        c = self._memory_character()
        text = " ".join(str(text).split())[:300]
        if not text:
            return
        if self.in_group:
            pins = self._group_pins()
            if len(pins) >= MAX_NOTES:
                raise UserError(f"This group already has {MAX_NOTES} pinned notes. Remove one first.")
            pins.append({"target": self._member(target).id if target else "", "text": text})
            self.memory.set_meta(self._mem_key(), "pins", json.dumps(pins))
            self._emit_director()
            return
        if len(c.director_notes) >= MAX_NOTES:
            raise UserError(f"{c.name} already has {MAX_NOTES} pinned notes. Remove one first.")
        self._save_director_notes([*c.director_notes, text])

    def _handle_unpin_note(self, index: int) -> None:
        import json

        c = self._memory_character()
        if self.in_group:
            pins = self._group_pins()
            if 0 <= int(index) < len(pins):
                del pins[int(index)]
                self.memory.set_meta(self._mem_key(), "pins", json.dumps(pins))
            self._emit_director()
            return
        notes = list(c.director_notes)
        if 0 <= int(index) < len(notes):
            del notes[int(index)]
            self._save_director_notes(notes)

    def _save_director_notes(self, notes: list[str]) -> None:
        c = self._memory_character()
        self.character = self.characters.save({**c.to_dict(), "director_notes": notes}, c.id)
        self.conversation.config.director = list(self.character.director_notes)
        self._emit({"type": "characters", "characters": self.list_characters(), "character": self.character.to_dict()})
        self._emit_director()

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
        if c is not None and self._mem_on():
            ids = [item["db_id"] for item in forget if item.get("db_id")]
            if ids:
                self.memory.delete_from(self._mem_key(), min(ids))
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
        if self.in_group:
            last = items[-1] if items else None
            if last is None or last.get("role") != "assistant" or not last.get("speaker"):
                raise UserError("There is no reply to write again.")
            self._check_deletable(len(items) - 1)
            speaker = self._member(last["speaker"]).id
            if last.get("note"):
                self.group_notes[speaker] = last["note"]  # the same direction again
            self._truncate(len(items) - 1)
            self._emit_transcript()
            self._continue_as(speaker, self._fixed_language() or self.last_language or FALLBACK_LANGUAGE)
            return
        if (len(items) < 2 or items[-1].get("role") != "assistant" or items[-1].get("greeting")
                or items[-2].get("role") != "user"):
            raise UserError("There is no reply to write again.")
        self._check_deletable(len(items) - 2)
        user = items[-2]
        self._truncate(len(items) - 1, keep_user=True)
        self._emit_transcript()
        if user.get("language"):
            self.last_language = user["language"]
        if self._fixed_language():
            self.last_language = self._fixed_language()
        self._turn(user["text"], self.last_language, user.get("source") or "text", user=user)  # keeps its note

    def _handle_edit_message(self, msg_id: int, text: str) -> None:
        """Edit one message in place; the messages after it stay. The model sees the new text from now on.

        The session recording follows: a character's reply is spoken again with its voice and replaces
        the old audio; your own message keeps your recorded voice, only its text changes.
        """
        index = next((i for i, t in enumerate(self.transcript) if t.get("id") == msg_id), None)
        if index is None:
            raise UserError("This message is no longer in the chat.")
        self._check_deletable(index)
        item = self.transcript[index]
        if item.get("role") == "user" and item.get("source") == "director" and not item.get("text"):
            raise UserError("A director's note alone cannot be edited; delete it instead.")
        if item.get("role") == "user":
            item["text"] = text
        else:
            item["segments"] = [{"kind": seg.kind, "text": seg.text.strip()} for seg in split_segments(text)
                                if seg.text.strip()]
        item["edited"] = True
        if item.get("db_id"):
            self.memory.update_message(item["db_id"], text)
        if self._mem_on():
            self._reload_llm_history()
        elif item.get("in_history"):
            later = sum(1 for t in self.transcript[index + 1:] if t.get("in_history"))
            history = self.conversation.history
            if later < len(history):
                old = history[-1 - later]
                history[-1 - later] = ChatMessage(old.role, text, name=old.name)
        if item.get("session") and item.get("seq"):
            self._record_edit(item)
        self._emit_transcript()
        self._enter_idle()

    def _record_edit(self, item: dict[str, Any]) -> None:
        """Replace the edited message in its session recording (a reply with newly spoken audio)."""
        from .sanitizer import clean_for_speech

        if item["role"] == "user":
            self.recorder.replace(item["session"], item["seq"],
                                  SessionItem(item["id"], "user", self._user_label(), item["text"]), keep_audio=True)
            return
        speech = spoken_text(item["segments"])
        speaker = self.characters.get(item.get("speaker") or "") or self.character
        audio = None
        keep_audio = False
        if item.get("duration") is not None and clean_for_speech(speech):  # it was spoken: speak it again
            if self.c.tts.loaded and self.store.settings.voice_output and item.get("voice"):
                try:
                    ref = self._voice_ref(item["voice"])
                except UserError:
                    ref = self._voice_ref(DEFAULT_VOICE)
                self._set_state(State.THINKING, "Speaking the new text for the recording…")
                result = self.c.tts.synthesize(clean_for_speech(speech), self.c.voices.prompt_for(ref),
                                               self.c.voices.language_of(ref) or self.last_language)
                audio = result.audio
            else:
                keep_audio = True
                self._emit({"type": "notice", "message": "Voice output is off, so the recording keeps the old "
                                                         "audio of this message; only its text was changed."})
        self.recorder.replace(item["session"], item["seq"], SessionItem(
            item["id"], "assistant", speaker.name if speaker else "Assistant", speech, [], audio,
            SESSION_TTS_RATE, segments=item["segments"]), keep_audio=keep_audio)

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
        key = self._mem_key()
        return {
            "character": key, "title": self._mem_title(), "group": self.in_group, "enabled": self._mem_on(),
            "summary": self.memory.summary(key),
            "facts": [{"id": f.id, "text": f.text, "source": f.source} for f in self.memory.facts(key)],
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
        """Voice output is on globally and for the active character (spec §7.4), or anyone in the group."""
        members = self.members
        return bool(self.store.settings.voice_output and (not members or any(m.voice_output for m in members)))

    def _sync_tts(self) -> None:
        self.conversation.tts = self.c.tts if (self._voice_output_on() and self.c.tts.loaded) else None

    def _activate_character(self, char_id: str | None, keep_group: bool = False) -> None:
        """Switch persona, voice, model, memory and history in one step (spec §10).

        Picking a character on the left starts a chat with them alone (``keep_group``: a group is opened).
        """
        char = self.characters.get(char_id or "") or self.characters.first()
        if char is None:
            raise UserError("No characters found. Create one with + New character.")
        s = self.store.settings
        if self.character is not None and self.character.id != char.id:
            self._end_session()
            self.pending_note = ""  # a note was meant for the other character
            self.group_notes.clear()
            self.story_pause()
            if not keep_group:
                s.group = []
        self.character = char
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
        cfg.persona = persona_text(char, DEFAULT_ASSISTANT_PERSONA)
        cfg.roleplay = char.kind == "roleplay"
        cfg.director = list(char.director_notes)
        self._refresh_memory_context()
        self._load_history()
        self._sync_tts()

    def _refresh_memory_context(self) -> None:
        key = self._mem_key()
        self.conversation.config.memory = (
            memory_context(self.memory.summary(key), self.memory.facts(key), group=self.in_group)
            if self.character is not None and self._mem_on() else "")

    def _load_history(self) -> None:
        """LLM history and visible transcript of the active character (from its memory)."""
        c = self.character
        assert c is not None
        self.conversation.history = []
        self.transcript = []
        self.resume = None
        group = self.in_group
        key = self._mem_key()
        if self._mem_on():
            stored = self.memory.active_messages(key)
            if any(m.role == "user" for m in stored) or (group and stored):
                # Offer "Continue" / "Start new chat" in the UI (the chat is loaded either way).
                self.resume = {"messages": len(stored), "last": stored[-1].created}
            for m in stored:
                self.conversation.history.append(ChatMessage(m.role, m.content, name=m.speaker))
                item: dict[str, Any] = {"id": self._new_id(), "db_id": m.id, "session": m.replay_session,
                                        "offset": m.replay_offset, "duration": m.replay_duration,
                                        "seq": m.replay_seq, "part": m.replay_part,
                                        "in_history": True}
                if m.role == "user":
                    director = m.source == "director"
                    item.update({"role": "user", "text": "" if director else m.content, "language": m.language,
                                 "source": m.source or "text"})
                    if m.note:
                        item["note"] = m.note
                else:
                    item.update({"role": "assistant", "voice": m.voice,
                                 "greeting": not group and m.voice is None and m.content == c.greeting,
                                 "segments": [{"kind": seg.kind, "text": seg.text}
                                              for seg in split_segments(m.content)]})
                    if m.speaker:
                        item["speaker"] = m.speaker
                    if m.note and group:
                        item["note"] = m.note
                self.transcript.append(item)
        if self.resume is None:
            self.memory.set_meta(key, "session", "")  # a fresh chat gets a new session recording
        if not self.conversation.history and c.greeting and not group:  # a group starts with you or Play
            self.conversation.history.append(ChatMessage("assistant", c.greeting))
            greeting: dict[str, Any] = {"id": self._new_id(), "role": "assistant", "voice": None, "greeting": True,
                                        "in_history": True,
                                        "segments": [{"kind": seg.kind, "text": seg.text}
                                                     for seg in split_segments(c.greeting)]}
            self.transcript.append(greeting)
            if self._mem_on():
                greeting["db_id"] = self.memory.add_message(key, "assistant", c.greeting)

    def _reload_llm_history(self) -> None:
        """After summarizing: the LLM sees only the unsummarized messages (the transcript stays)."""
        if self.character is not None and self._mem_on():
            self.conversation.history = [ChatMessage(m.role, m.content, name=m.speaker)
                                         for m in self.memory.active_messages(self._mem_key())]

    def _emit_character_state(self) -> None:
        self._emit({"type": "character", "character": self.character.to_dict() if self.character else None,
                    "characters": self.list_characters(), "transcript": self.transcript,
                    "deletable_from": self._deletable_from(), "memory": self.memory_view(), "resume": self.resume,
                    "director": self.director_view(), "group": self.group_view()})
        self._emit({"type": "settings", "settings": asdict(self.store.settings)})
        self._emit({"type": "voice_info", **self.voice_info()})

    def _handle_select_character(self, char_id: str) -> None:
        if self.character is not None and self.character.id == char_id:
            if self.store.settings.group:  # back from a group to a chat with them alone
                self._change_cast(char_id, [])
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
            self._activate_character(char.id, keep_group=char_id is not None)  # takes effect now
            self._prepare_voice()
            self._ensure_llm()
        elif char.id in self.store.settings.group:
            self._refresh_memory_context()  # e.g. memory turned off for someone in the group
            self._sync_tts()
            self._prepare_group_voices()
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
        for key in self.memory.groups_with(char_id):
            self.memory.reset(key)  # and so do the groups it was in
        s = self.store.settings
        s.group = [i for i in s.group if i != char_id]
        s.recent_groups = [g for g in s.recent_groups if char_id not in g]
        self.store.save()
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
        if c is None or not self._mem_on() or not reply_text:
            self._trim_history()
            return
        user_db = self.memory.add_message(c.id, "user", user["text"] or CONTINUE_TEXT, language, source,
                                          note=user.get("note"))
        reply_db = self.memory.add_message(c.id, "assistant", reply_text, language, None, reply["voice"])
        for item, db_id in ((user, user_db), (reply, reply_db)):
            with self._replay_lock:  # the recorder may already have finished this message
                item["db_id"] = db_id
                if item.get("session"):
                    self.memory.set_replay(db_id, item["session"], item["seq"], item.get("part"), item.get("duration"))

    def _trim_history(self) -> None:
        """Without memory the oldest exchanges drop out once the history outgrows its context share."""
        history = self.conversation.history
        budget = history_budget(self.context)
        step = 1 if self.in_group else 2  # a group's lines are not in pairs
        while len(history) > 2 and sum(estimate_tokens(m.content) for m in history) > budget:
            del history[:step]

    def _llm_text(self, system: str, prompt: str, max_tokens: int = 400) -> str:
        """One quick, low-temperature request used to maintain the memory."""
        messages = [ChatMessage("system", system), ChatMessage("user", prompt)]
        text = self.c.llm.complete(messages, SamplingConfig(temperature=0.3, max_tokens=max_tokens))
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    def _maintain_memory(self) -> None:
        """Between turns: summarize old history when it gets long; update the facts every few turns."""
        if self.character is None or not self._mem_on() or not self.c.server.running:
            return
        key = self._mem_key()
        try:
            old = messages_to_summarize(self.memory.active_messages(key), self.context)
            if old:
                self._summarize(old)
            turns = int(self.memory.get_meta(key, "turns_since_facts", "0") or 0) + 1
            if turns >= FACT_UPDATE_EVERY:
                self._update_facts()
                turns = 0
            self.memory.set_meta(key, "turns_since_facts", str(turns))
        except Exception:  # noqa: BLE001 - memory upkeep must never break the conversation
            log.exception("Memory maintenance failed")
        self._refresh_memory_context()
        self._emit({"type": "memory", **self.memory_view()})

    def _summarize(self, old: list[Any]) -> None:
        key, title, names = self._mem_key(), self._mem_title(), self._names()
        words = summary_words(self.context)
        summary = self._llm_text(
            SUMMARY_SYSTEM.format(name=title),
            SUMMARY_PROMPT.format(summary=self.memory.summary(key) or "(none)",
                                  transcript=transcript_text(old, title, self.store.settings.user_name, names),
                                  words=words),
            max_tokens=max(400, words * 2))
        if summary:
            self.memory.set_summary(key, summary)
            self.memory.mark_summarized(key, old[-1].id)
            self._reload_llm_history()
            log.info("Summarized %d messages of %s", len(old), key)

    def _update_facts(self, recent: list[Any] | None = None) -> None:
        """``recent``: the messages to learn from (by default the last eight of the chat)."""
        key, title, names = self._mem_key(), self._mem_title(), self._names()
        if recent is None:
            recent = self.memory.active_messages(key)[-8:]
        facts = self.memory.facts(key)
        reply = self._llm_text(
            FACTS_SYSTEM.format(name=title),
            FACTS_PROMPT.format(facts="\n".join(f"- {f.text}" for f in facts) or "(none)",
                                transcript=transcript_text(recent, title, self.store.settings.user_name, names)),
            max_tokens=300)
        add, remove = parse_fact_update(reply)
        self.memory.remove_facts_by_text(key, remove)
        for text in add:
            self.memory.add_fact(key, text, source="llm")

    def _handle_add_fact(self, text: str) -> None:
        self._memory_character()
        self.memory.add_fact(self._mem_key(), text, source="user")
        self._refresh_memory_context()
        self._emit({"type": "memory", **self.memory_view()})

    def _handle_update_fact(self, fact_id: int, text: str) -> None:
        self._memory_character()
        self.memory.update_fact(self._mem_key(), int(fact_id), text)
        self._refresh_memory_context()
        self._emit({"type": "memory", **self.memory_view()})

    def _handle_delete_fact(self, fact_id: int) -> None:
        self._memory_character()
        self.memory.delete_fact(self._mem_key(), int(fact_id))
        self._refresh_memory_context()
        self._emit({"type": "memory", **self.memory_view()})

    def _handle_reset_memory(self, _: Any = None) -> None:
        self._memory_character()
        self.story_pause()
        self._finish_session()
        self.memory.reset(self._mem_key())
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
        members = self.members
        return bool(self.store.settings.record_sessions and all(m.recording_enabled for m in members))

    def _ensure_session(self) -> bool:
        """Start (or continue) the session recording of the current chat; False when not recording."""
        if not self._recording_on():
            self._end_session()
            return False
        if self.recorder.name is not None:
            return True
        c = self.character
        key = self._mem_key()
        name = self.memory.get_meta(key, "session", "")
        if name and SessionRecorder.valid_name(name) and (self.paths.sessions / f"{name}.md").exists():
            self.recorder.resume(name)  # a resumed chat continues its recording
        else:
            name = self.recorder.new_name(session_stem([m.id for m in self.members]) if self.in_group
                                          else (c.id if c else "chat"))
            self.recorder.start(name, session_title(self._mem_title()))
            self.memory.set_meta(key, "session", name)
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

    def _on_recorded(self, msg_id: int, session: str, seq: int, part: str | None, duration: float | None) -> None:
        """Called by the recorder thread once a message is in the session files (its own MP3)."""
        with self._replay_lock:
            item = next((t for t in self.transcript if t.get("id") == msg_id), None)
            if item is not None:
                item.update(session=session, seq=seq, part=part, duration=duration, offset=None)
                if item.get("db_id"):
                    self.memory.set_replay(item["db_id"], session, seq, part, duration)
        self._emit({"type": "recorded", "id": msg_id, "session": session, "seq": seq, "part": part,
                    "duration": duration})

    def _on_session_built(self, name: str) -> None:
        """Called by the recorder thread: the MP3 of a whole session is ready (or there was nothing to do)."""
        self._emit_sessions()
        if name == self._finishing:
            self._finishing = None
            if (self.paths.sessions / f"{name}.mp3").exists():
                self._emit({"type": "notice",
                            "message": f"The recording is ready: user_data\\sessions\\{name}.mp3"})

    def _finish_session(self) -> None:
        """End chat: the recording of this chat is put together into one MP3."""
        name = self.recorder.name or self.memory.get_meta(self._mem_key(), "session", "")
        if self.recorder.name != name:
            self._end_session()
        if not name or not SessionRecorder.valid_name(name):
            return
        spoken = any(t.get("duration") or t.get("source") == "voice" or (t.get("role") == "assistant" and
                                                                          t.get("voice")) for t in self.transcript)
        if spoken:
            self._finishing = name
            self._emit({"type": "notice", "message": "Putting the recording of this chat together into one MP3…"})
        self.recorder.finish(name)  # also ends the recording of this chat
        self._emit_sessions()

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
        values = {k: v for k, v in values.items() if k not in ENGINE_OWNED_SETTINGS}
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
        if "talk_mode" in changed:
            self._apply_talk_mode()
        if "vad_threshold" in changed:
            self.segmenter.config.threshold = s.vad_threshold
            self.segmenter.config.neg_threshold = max(0.1, s.vad_threshold - 0.15)
        if "match_language" in changed:
            self.conversation.config.match_language = s.match_language
        if "language" in changed:
            self.last_language = self._fixed_language()  # the next reply already uses the new language
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
        if changed.keys() - {"show_actions", "user_name", "user_about", "user_avatar", "user_color", "language",
                             "story_rules"}:
            self._enter_idle()

    def _validate(self, changed: dict[str, Any]) -> None:
        """Reject values the settings screen never sends (hand-edited or stale clients)."""
        s = self.store.settings
        checks = {
            "vad_threshold": (0.2 <= s.vad_threshold <= 0.9, "Speech detection must be between 0.2 and 0.9."),
            "end_silence_ms": (200 <= s.end_silence_ms <= 3000, "End of speech must be between 200 and 3000 ms."),
            "talk_mode": (s.talk_mode in TALK_MODES, f"Unknown talk mode {s.talk_mode}."),
            "language": (s.language in ("auto", *CONVERSATION_LANGUAGES), f"Unknown language {s.language}."),
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
            "story_rules": (len(s.story_rules) <= MAX_RULES, f"The story rules are too long ({MAX_RULES} "
                                                             "characters at most)."),
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
