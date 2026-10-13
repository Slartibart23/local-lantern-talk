"""Engine flow tests with fake components (no GPU, no models)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lantern_talk.engine import Components, Engine, State
from lantern_talk.paths import Paths
from lantern_talk.settings import SettingsStore
from lantern_talk.tts import load_builtin_voices

ROOT = Path(__file__).parents[1]


@dataclass
class FakeTranscript:
    text: str
    language: str = "de"
    language_probability: float = 0.99
    seconds: float = 0.05


class Loadable:
    def __init__(self) -> None:
        self.loaded = False
        self.load_calls = 0

    def load(self) -> None:
        self.loaded = True
        self.load_calls += 1

    def unload(self) -> None:
        self.loaded = False


class FakeSTT(Loadable):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> FakeTranscript:
        self.calls += 1
        return FakeTranscript("Hallo, wie geht es dir?")


class FakeVAD(Loadable):
    """Speech whenever the frame is loud."""

    def __init__(self) -> None:
        super().__init__()
        self.frames = 0

    def reset(self) -> None:
        pass

    def speech_probability(self, frame: np.ndarray) -> float:
        self.frames += 1
        return 0.9 if float(np.abs(frame).mean()) > 0.1 else 0.0


@dataclass
class FakeSynth:
    audio: np.ndarray
    seconds: float = 0.01

    @property
    def duration_s(self) -> float:
        return len(self.audio) / 24_000


class FakeTTS(Loadable):
    num_step = 32

    def synthesize(self, text: str, prompt: object, language: str | None = None) -> FakeSynth:
        return FakeSynth(np.zeros(2400, dtype=np.float32))


class FakeVoices:
    def __init__(self) -> None:
        self.builtins = load_builtin_voices(ROOT / "config" / "builtin_voices.yaml")
        self.prepared: list[str] = []

    def prompt_for(self, voice: str) -> object:
        self.prepared.append(voice)
        return object()

    def matched(self, voice: str, language: str | None) -> str:
        from lantern_talk.tts import match_builtin_voice

        return match_builtin_voice(voice, language, self.builtins)

    def language_of(self, voice: str) -> str | None:
        return None


class FakeServer:
    def __init__(self) -> None:
        self.running = False
        self.started_with: list[str] = []

    def start(self, config: Any) -> None:
        self.running = True
        self.started_with.append(config.model.name)

    def stop(self) -> None:
        self.running = False


class FakeLLM:
    reply = "*lächelt* Mir geht es gut. Und dir?"

    def stream_chat(self, messages: list[Any]):  # noqa: ANN201
        for i in range(0, len(self.reply), 4):
            yield self.reply[i: i + 4]


class Recorder:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.binaries: list[bytes] = []
        self.cv = threading.Condition()

    def __call__(self, msg: dict[str, Any], binary: bytes | None) -> None:
        with self.cv:
            self.events.append(msg)
            if binary is not None:
                self.binaries.append(binary)
            self.cv.notify_all()

    def wait_for(self, predicate, timeout: float = 5.0) -> None:  # noqa: ANN001
        deadline = time.monotonic() + timeout
        with self.cv:
            while not predicate(self.events):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(f"timeout; events: {[e['type'] for e in self.events]}")
                self.cv.wait(remaining)

    def wait_state(self, state: str, after: int = 0) -> None:
        self.wait_for(lambda ev: any(e["type"] == "state" and e["state"] == state for e in ev[after:]))


@pytest.fixture()
def setup(tmp_path: Path):  # noqa: ANN201
    root = tmp_path / "root"
    (root / "app").mkdir(parents=True)
    (root / "config").mkdir()
    paths = Paths(root)
    paths.llm_models.mkdir(parents=True)
    (paths.llm_models / "test-model-Q4_K_M.gguf").write_bytes(b"x")
    store = SettingsStore(paths.settings_file)
    comps = Components(stt=FakeSTT(), vad=FakeVAD(), tts=FakeTTS(), voices=FakeVoices(),
                       server=FakeServer(), llm=FakeLLM())
    rec = Recorder()
    engine = Engine(paths, store, comps, rec)
    engine.start()
    rec.wait_state("listening")
    yield engine, comps, rec
    engine.shutdown()


def test_startup_loads_everything(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    assert comps.stt.loaded and comps.vad.loaded and comps.tts.loaded
    assert comps.server.started_with == ["test-model-Q4_K_M.gguf"]
    assert engine.store.settings.model == "test-model-Q4_K_M.gguf"
    snap = engine.snapshot()
    assert snap["state"] == "listening"
    assert len(snap["voices"]["builtin"]) == 8


def test_text_turn_flow(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    start = len(rec.events)
    engine.submit_text("Hallo!")
    rec.wait_state("speaking", start)
    rec.wait_for(lambda ev: any(e["type"] == "turn_done" for e in ev[start:]))
    types = [e["type"] for e in rec.events[start:]]
    assert types.index("user_message") < types.index("assistant_start") < types.index("audio")
    segs = [e for e in rec.events[start:] if e["type"] == "segment"]
    assert {"kind": "action", "text": "lächelt"} in [{k: s[k] for k in ("kind", "text")} for s in segs]
    spoken = " ".join(e["text"] for e in rec.events[start:] if e["type"] == "audio")
    assert "lächelt" not in spoken
    assert rec.binaries, "PCM audio must be sent"
    # Still speaking until the browser reports the end of playback.
    time.sleep(0.2)
    assert engine.state is State.SPEAKING
    engine.playback_ended()
    rec.wait_state("listening", start)
    assert engine.transcript[-1]["segments"][0] == {"kind": "action", "text": "lächelt"}


@pytest.mark.parametrize("text, voice", [("Hallo! Wer bist du?", "builtin:de_female"),
                                         ("Hello! Who are you?", "builtin:en_female"),
                                         ("¿Hola, quién eres?", "builtin:es_female")])
def test_typed_message_language_picks_voice(setup, text: str, voice: str) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    start = len(rec.events)
    engine.submit_text(text)
    rec.wait_for(lambda ev: any(e["type"] == "assistant_start" for e in ev[start:]))
    assert [e for e in rec.events[start:] if e["type"] == "assistant_start"][0]["voice"] == voice
    engine.playback_ended()


def test_ambiguous_typed_message_keeps_last_language(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    engine.last_language = "de"
    start = len(rec.events)
    engine.submit_text("ok")
    rec.wait_for(lambda ev: any(e["type"] == "assistant_start" for e in ev[start:]))
    assert [e for e in rec.events[start:] if e["type"] == "assistant_start"][0]["voice"] == "builtin:de_female"
    engine.playback_ended()


def loud(n: int = 4800) -> bytes:
    return (np.full(n, 0.5) * 32767).astype(np.int16).tobytes()


def quiet(n: int = 4800) -> bytes:
    return np.zeros(n, dtype=np.int16).tobytes()


def test_voice_utterance_and_mic_muted_while_busy(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    start = len(rec.events)
    for _ in range(5):
        engine.on_audio(loud(), 48_000)       # 0.5 s speech
    for _ in range(12):
        engine.on_audio(quiet(), 48_000)      # 1.2 s silence -> end of utterance
    rec.wait_for(lambda ev: any(e["type"] == "user_message" for e in ev[start:]))
    msg = [e for e in rec.events[start:] if e["type"] == "user_message"][0]
    assert msg["source"] == "voice" and msg["text"] == "Hallo, wie geht es dir?"
    rec.wait_state("speaking", start)
    frames_before = comps.vad.frames
    for _ in range(10):
        engine.on_audio(loud(), 48_000)       # the assistant's own voice: must be ignored
    time.sleep(0.2)
    assert comps.vad.frames == frames_before
    engine.playback_ended()
    rec.wait_state("listening", start)
    assert comps.stt.calls == 1


def test_mic_off_unloads_whisper_and_vad(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    start = len(rec.events)
    engine.update_settings({"mic_enabled": False})
    rec.wait_state("mic_off", start)
    assert not comps.stt.loaded and not comps.vad.loaded
    engine.update_settings({"mic_enabled": True})
    rec.wait_state("listening", start)
    assert comps.stt.loaded and comps.vad.loaded


def test_voice_output_off_gives_text_only_replies(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    engine.update_settings({"voice_output": False})
    rec.wait_for(lambda ev: any(e["type"] == "settings" and not e["settings"]["voice_output"] for e in ev))
    assert not comps.tts.loaded
    start = len(rec.events)
    engine.submit_text("Hallo!")
    rec.wait_for(lambda ev: any(e["type"] == "turn_done" for e in ev[start:]))
    rec.wait_state("listening", start)  # no playback wait without audio
    assert not any(e["type"] == "audio" for e in rec.events[start:])


def test_pause(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    start = len(rec.events)
    engine.set_paused(True)
    rec.wait_state("paused", start)
    engine.on_audio(loud(), 48_000)
    assert engine._audio.empty()
    engine.set_paused(False)
    rec.wait_state("listening", start)


def test_unknown_voice_is_rejected_and_rolled_back(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    start = len(rec.events)
    engine.update_settings({"voice": "custom:missing.mp3"})
    rec.wait_for(lambda ev: any(e["type"] == "error" for e in ev[start:]))
    assert engine.store.settings.voice == "builtin:en_female"
    rec.wait_state("listening", start)


def test_new_conversation_clears_history(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    engine.store.settings.voice_output = False
    engine.conversation.tts = None
    engine.submit_text("Hallo!")
    rec.wait_for(lambda ev: any(e["type"] == "turn_done" for e in ev))
    engine.new_conversation()
    rec.wait_for(lambda ev: any(e["type"] == "cleared" for e in ev))
    assert engine.transcript == [] and engine.conversation.history == []
