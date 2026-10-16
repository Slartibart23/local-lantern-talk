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
    precision = "fp16"

    def synthesize(self, text: str, prompt: object, language: str | None = None) -> FakeSynth:
        return FakeSynth(np.zeros(2400, dtype=np.float32))


class FakeVoices:
    def __init__(self) -> None:
        self.builtins = load_builtin_voices(ROOT / "config" / "builtin_voices.yaml")
        self.prepared: list[str] = []
        self.texts: dict[str, tuple[str, str]] = {}
        self.languages: dict[str, str] = {}

    def prompt_for(self, voice: str) -> object:
        self.prepared.append(voice)
        return object()

    def matched(self, voice: str, language: str | None) -> str:
        from lantern_talk.tts import match_builtin_voice

        return match_builtin_voice(voice, language, self.builtins)

    def language_of(self, voice: str) -> str | None:
        b = self.builtins.get(voice.removeprefix("builtin:"))
        return b.language if b else None

    def reference_text(self, source: Path) -> tuple[str, str] | None:
        return self.texts.get(source.name)

    def clip_report(self, source: Path) -> dict[str, Any] | None:
        import soundfile as sf

        from lantern_talk.audio_prep import analyze

        try:
            audio, _ = sf.read(str(source), dtype="float32")
        except Exception:  # noqa: BLE001
            return None
        return analyze(audio, audio).to_dict()

    def custom_language(self, source: Path) -> str | None:
        return self.languages.get(source.name, "de")

    def user_language(self, source: Path) -> str | None:
        return self.languages.get(source.name)

    def set_language(self, source: Path, code: str) -> None:
        if code:
            self.languages[source.name] = code
        else:
            self.languages.pop(source.name, None)

    def forget(self, source: Path) -> None:
        self.forgotten = [*getattr(self, "forgotten", []), source.name]

    def cleanup_cache(self, voices_dir: Path) -> None:
        pass

    def regenerate_builtin(self, voice_id: str) -> None:
        self.regenerated = [*getattr(self, "regenerated", []), voice_id]

    def set_reference_text(self, source: Path, text: str) -> None:
        self.texts[source.name] = (text, "user")


class FakeServer:
    def __init__(self) -> None:
        self.running = False
        self.started_with: list[str] = []

    def start(self, config: Any) -> None:
        self.running = True
        self.started_with.append(config.model.name)
        self.last_config = config

    def stop(self) -> None:
        self.running = False


class FakeLLM:
    reply = "*lächelt* Mir geht es gut. Und dir?"

    def __init__(self) -> None:
        self.requests: list[list[Any]] = []
        self.side_replies: list[str] = []  # answers for memory upkeep requests (complete)

    def stream_chat(self, messages: list[Any], sampling: Any = None):  # noqa: ANN201
        self.requests.append(messages)
        for i in range(0, len(self.reply), 4):
            yield self.reply[i: i + 4]

    def complete(self, messages: list[Any], sampling: Any = None) -> str:
        self.requests.append(messages)
        return self.side_replies.pop(0) if self.side_replies else ""


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
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    import shutil

    from lantern_talk import engine as engine_module

    # A fixed fake GPU (24 GB, 6 GB in use) keeps the tests independent of this machine.
    monkeypatch.setattr(engine_module, "query_gpu",
                        lambda: {"name": "Test GPU", "used_mb": 6000, "total_mb": 24000})
    root = tmp_path / "root"
    (root / "app").mkdir(parents=True)
    (root / "config").mkdir()
    shutil.copy(ROOT / "config" / "model_catalog.yaml", root / "config" / "model_catalog.yaml")
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


def wav_bytes(seconds: float) -> bytes:
    import io

    import soundfile as sf

    # syllable-like tone bursts: a clean "recording" without background noise
    t = np.arange(int(24_000 * seconds)) / 24_000
    audio = (0.3 * (np.sin(2 * np.pi * 3 * t) > 0) * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, 24_000, format="WAV")
    return buf.getvalue()


def test_add_voice_stores_clip_and_lists_it(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    result = engine.add_voice("Meine Stimme.wav", wav_bytes(5))
    assert result["id"] == "custom:Meine Stimme.wav" and result["warnings"] == []
    assert (engine.paths.voices / "Meine Stimme.wav").exists()
    assert {"id": "custom:Meine Stimme.wav", "name": "Meine Stimme"} in engine.list_voices()["custom"]
    assert any(e["type"] == "voices" for e in rec.events)
    # Same name again: never overwrite an existing voice.
    assert engine.add_voice("Meine Stimme.wav", wav_bytes(5))["id"] == "custom:Meine Stimme (2).wav"


def test_add_voice_sanitizes_path(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    result = engine.add_voice(r"..\..\evil<>name.wav", wav_bytes(5))
    assert result["id"] == "custom:evil_name.wav"
    assert (engine.paths.voices / "evil_name.wav").exists()


@pytest.mark.parametrize("name, data, message", [
    ("clip.exe", b"MZ", "MP3, WAV"),
    ("clip.mp3", b"", "empty"),
    ("clip.wav", b"not audio at all", "Could not read"),
])
def test_add_voice_rejects_bad_files(setup, name: str, data: bytes, message: str) -> None:  # noqa: ANN001
    from lantern_talk.engine import UserError

    engine, comps, rec = setup
    with pytest.raises(UserError, match=message):
        engine.add_voice(name, data)
    assert not list(engine.paths.voices.glob("clip*"))


def test_add_voice_warns_about_length(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    assert "Only 1.5 s" in engine.add_voice("short.wav", wav_bytes(1.5))["warnings"][0]
    assert "first 15 s" in engine.add_voice("long.wav", wav_bytes(25))["warnings"][0]


def test_add_voice_accepts_messenger_formats(setup) -> None:  # noqa: ANN001
    from lantern_talk.engine import UserError

    engine, comps, rec = setup
    with pytest.raises(UserError, match="Could not read"):  # accepted extension, broken content
        engine.add_voice("voice-note.opus", b"not really opus")
    with pytest.raises(UserError, match="OPUS, M4A"):
        engine.add_voice("clip.exe", b"MZ")


def test_rename_voice_keeps_selection(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    voice = engine.add_voice("anna.wav", wav_bytes(5))["id"]
    (engine.paths.voices / "anna.txt").write_text("Hallo", encoding="utf-8")
    engine.update_settings({"voice": voice})
    rec.wait_for(lambda ev: any(e["type"] == "settings" and e["settings"]["voice"] == voice for e in ev))
    start = len(rec.events)
    engine.rename_voice(voice, "Anna <ruhig>")
    rec.wait_for(lambda ev: any(e["type"] == "settings" and e["settings"]["voice"] != voice for e in ev[start:]))
    assert engine.store.settings.voice == "custom:Anna _ruhig.wav"
    assert (engine.paths.voices / "Anna _ruhig.wav").exists() and (engine.paths.voices / "Anna _ruhig.txt").exists()
    assert not (engine.paths.voices / "anna.wav").exists()


def test_delete_selected_voice_falls_back_to_builtin(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    voice = engine.add_voice("anna.wav", wav_bytes(5))["id"]
    engine.update_settings({"voice": voice})
    rec.wait_for(lambda ev: any(e["type"] == "settings" and e["settings"]["voice"] == voice for e in ev))
    start = len(rec.events)
    engine.delete_voice(voice)
    rec.wait_for(lambda ev: any(e["type"] == "voices" for e in ev[start:]))
    rec.wait_for(lambda ev: any(e["type"] == "settings" and e["settings"]["voice"] == "builtin:en_female"
                                for e in ev[start:]))
    assert not (engine.paths.voices / "anna.wav").exists()


def test_builtin_voices_cannot_be_deleted(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    start = len(rec.events)
    engine.delete_voice("builtin:de_male")
    rec.wait_for(lambda ev: any(e["type"] == "error" and "cannot" in e["message"] for e in ev[start:]))


def test_preview_speaks_sample_in_voice_language(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    start = len(rec.events)
    engine.preview_voice("builtin:fr_male")
    rec.wait_for(lambda ev: any(e["type"] == "preview_done" for e in ev[start:]))
    audio = [e for e in rec.events[start:] if e["type"] == "audio"]
    assert audio and audio[0]["text"].startswith("Bonjour")
    assert engine.state is State.SPEAKING  # mic stays muted until the browser played it
    engine.playback_ended()
    rec.wait_state("listening", start)
    assert engine.store.settings.voice == "builtin:en_female"  # preview does not change the selection


def test_regenerate_builtin_voices(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    start = len(rec.events)
    engine.regenerate_builtin_voices()
    rec.wait_for(lambda ev: any(e["type"] == "notice" for e in ev[start:]))
    assert len(comps.voices.regenerated) == 8


def test_reference_text_rebuilds_active_custom_voice(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    voice = engine.add_voice("me.wav", wav_bytes(5))["id"]
    engine.update_settings({"voice": voice})
    rec.wait_for(lambda ev: any(e["type"] == "voice_info" and e["voice"] == voice for e in ev))
    prepared_before = len(comps.voices.prepared)
    start = len(rec.events)
    engine.set_reference_text(voice, "Das ist mein Text.")
    rec.wait_for(lambda ev: any(e["type"] == "voice_info" and e["reference_text"] for e in ev[start:]))
    info = [e for e in rec.events[start:] if e["type"] == "voice_info"][-1]
    assert info["reference_text"] == "Das ist mein Text." and info["origin"] == "user"
    assert info["text_file"].endswith("me.txt")
    assert len(comps.voices.prepared) == prepared_before + 1  # voice prepared again
    rec.wait_state("listening", start)


def test_reference_text_only_for_custom_voices(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    start = len(rec.events)
    engine.set_reference_text("builtin:de_male", "x")
    rec.wait_for(lambda ev: any(e["type"] == "error" for e in ev[start:]))


def test_precision_change_reloads_omnivoice(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    loads = comps.tts.load_calls
    engine.update_settings({"tts_precision": "bf16"})
    rec.wait_for(lambda ev: any(e["type"] == "settings" and e["settings"]["tts_precision"] == "bf16" for e in ev))
    assert comps.tts.precision == "bf16" and comps.tts.load_calls == loads + 1
    start = len(rec.events)
    engine.update_settings({"tts_precision": "fp8"})
    rec.wait_for(lambda ev: any(e["type"] == "error" for e in ev[start:]))
    assert engine.store.settings.tts_precision == "bf16"  # rolled back


# ---------------------------------------------------------------- model catalog (M4)

def fake_download(engine, size: int = 10):  # noqa: ANN001, ANN201
    """Replace the network download with one that writes the file immediately."""
    def start(key, url, dest, expected_size=0, on_done=None):  # noqa: ANN001, ANN202
        from lantern_talk.downloads import DownloadJob

        dest.write_bytes(b"x" * size)
        job = DownloadJob(key=key, file_name=dest.name, url=url, dest=str(dest), state="done")
        on_done(job)
        return job

    engine.models.downloader.start = start


def test_snapshot_lists_catalog_and_local_model(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    models = engine.snapshot()["models"]
    ids = {m["id"] for m in models}
    assert {"gemma4-12b", "cydonia-24b-v4.3"} <= ids
    local = [m for m in models if m["source"] == "local"]
    assert local and local[0]["state"] == "loaded" and local[0]["variant"] == "unknown"


def test_llm_runs_without_thinking_and_keeps_room_for_voice(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    args = comps.server.last_config.extra_args
    assert args[args.index("--reasoning-budget") + 1] == "0"
    # llama-fit-params is unavailable in the test: llama.cpp fits itself, keeping room for voice
    assert args[args.index("--fit-target") + 1] == str(1024 + 2560)
    assert comps.server.last_config.gpu_layers == "auto"



def test_download_then_load(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    fake_download(engine)
    start = len(rec.events)
    engine.download_model("gemma4-12b")
    rec.wait_for(lambda ev: any(e["type"] == "settings" and e["settings"]["model"].startswith("gemma-4-12b")
                                for e in ev[start:]))
    file = engine.store.settings.model
    assert file == "gemma-4-12b-it-Q5_K_M.gguf"  # 24 GB fake GPU: profile quant Q5 fits
    assert comps.server.started_with[-1] == file
    view = {m["id"]: m for m in engine.list_models()}
    assert view["gemma4-12b"]["state"] == "loaded"


def test_loaded_model_cannot_be_deleted(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    start = len(rec.events)
    engine.delete_model(engine.store.settings.model)
    rec.wait_for(lambda ev: any(e["type"] == "error" and "loaded" in e["message"] for e in ev[start:]))
    assert (engine.paths.llm_models / engine.store.settings.model).exists()


def test_delete_other_model(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    other = engine.paths.llm_models / "other-Q4_K_M.gguf"
    other.write_bytes(b"x")
    engine.delete_model(other.name)
    rec.wait_for(lambda ev: not other.exists())


def test_model_too_big_is_refused(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    (engine.paths.llm_models / "huge-Q8_0.gguf").write_bytes(b"x")
    from lantern_talk.models import FitResult

    engine.models.fit = lambda name, ctx, margin, free=None: FitResult(["-ngl", "0"], 0, 64, 0)
    start = len(rec.events)
    engine.update_settings({"model": "huge-Q8_0.gguf"})
    rec.wait_for(lambda ev: any(e["type"] == "error" and "too big" in e["message"] for e in ev[start:]))


def test_fitted_split_is_applied_exactly(setup) -> None:  # noqa: ANN001
    from lantern_talk.models import parse_fit_output

    engine, comps, rec = setup
    ot = r"blk\.24\.ffn_(gate|down).*=CPU,blk\.25\.ffn_(up)_(ch|)exps=CPU"
    out = f'noise\n-c 8192 -ngl 31 -ot "{ot}"\n'
    engine.models.fit = lambda name, ctx, margin, free=None: parse_fit_output(out, 30)
    (engine.paths.llm_models / "moe-Q4_K_M.gguf").write_bytes(b"x")
    start = len(rec.events)
    engine.update_settings({"model": "moe-Q4_K_M.gguf"})
    rec.wait_for(lambda ev: any(e["type"] == "model_status" for e in ev[start:]))
    cfg = comps.server.last_config
    assert cfg.gpu_layers == "31" and cfg.extra_args[:2] == ["-ot", ot]
    assert cfg.extra_args[cfg.extra_args.index("--fit") + 1] == "off"
    status = [e for e in rec.events[start:] if e["type"] == "model_status"][-1]
    assert status["partial"] and status["cpu_blocks"] == 2 and "experts of 2 of 30" in status["notice"]


def test_partial_offload_notice(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    (engine.paths.llm_models / "big-Q4_K_M.gguf").write_bytes(b"x")
    from lantern_talk.models import FitResult

    engine.models.fit = lambda name, ctx, margin, free=None: FitResult(["-ngl", "30"], 30, 48, 0)
    start = len(rec.events)
    engine.update_settings({"model": "big-Q4_K_M.gguf"})
    rec.wait_for(lambda ev: any(e["type"] == "model_status" for e in ev[start:]))
    status = [e for e in rec.events[start:] if e["type"] == "model_status"][-1]
    assert status["partial"] and "30 of 48" in status["notice"]


def test_label_for_hand_copied_model(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    file = engine.store.settings.model
    engine.set_model_label(file, "uncensored")
    rec.wait_for(lambda ev: any(e["type"] == "models" for e in ev))
    local = [m for m in engine.list_models() if m["source"] == "local"][0]
    assert local["variant"] == "uncensored"


def test_crashed_language_model_is_restarted_before_the_next_turn(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    comps.server.running = False  # llama-server died on its own
    start = len(rec.events)
    engine.submit_text("Hallo!")
    rec.wait_for(lambda ev: any(e["type"] == "turn_done" for e in ev[start:]))
    assert comps.server.started_with == ["test-model-Q4_K_M.gguf"] * 2
    assert any(e["type"] == "notice" and "started again" in e["message"] for e in rec.events[start:])
    engine.playback_ended()


def test_error_messages_are_friendly() -> None:
    import httpx

    from lantern_talk.errors import GPU_MEMORY, LLM_CONNECTION, describe

    assert describe(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")) == GPU_MEMORY
    assert describe(httpx.ConnectError("refused")) == LLM_CONNECTION
    msg = describe(ValueError("bad value\nTraceback line"), "Speech synthesis")
    assert msg.startswith("Speech synthesis failed (bad value).") and "logs\\server.log" in msg


def test_custom_voice_language_can_be_chosen(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    voice = engine.add_voice("anna.wav", wav_bytes(4))["id"]
    start = len(rec.events)
    engine.update_settings({"voice": voice})
    rec.wait_for(lambda ev: any(e["type"] == "voice_info" and e["voice"] == voice for e in ev[start:]))
    start = len(rec.events)
    engine.set_voice_language(voice, "fr")
    rec.wait_for(lambda ev: any(e["type"] == "voice_info" and e.get("language") == "fr" for e in ev[start:]))
    info = engine.voice_info()
    assert info["language_set"] is True
    start = len(rec.events)
    engine.set_voice_language(voice, "xx")
    rec.wait_for(lambda ev: any(e["type"] == "error" for e in ev[start:]))
    start = len(rec.events)
    engine.set_voice_language("builtin:de_female", "fr")
    rec.wait_for(lambda ev: any(e["type"] == "error" and "fixed language" in e["message"] for e in ev[start:]))


def test_automatic_context_fills_the_free_gpu_memory(setup) -> None:  # noqa: ANN001
    from lantern_talk.models import ContextProfile

    engine, comps, rec = setup
    # 24 GB fake GPU, 6 GB in use, 3.5 GB kept for speech: about 14.1 GB for the language model.
    profile = ContextProfile(((4096, 10_000), (16384, 12_000), (65536, 20_000)), 131072)
    engine._context_profiles["test-model-Q4_K_M.gguf"] = profile
    assert engine._target_context("test-model-Q4_K_M.gguf") == 27648
    engine.store.settings.hardware_profile_override = "mid"   # the profile caps "Automatic"
    assert engine._target_context("test-model-Q4_K_M.gguf") == 16384
    engine.store.settings.context_auto = False                # a fixed value is used as it is
    engine.store.settings.context = 40960
    assert engine._target_context("test-model-Q4_K_M.gguf") == 40960
    info = engine.context_info()
    assert info["profile"]["points"][0] == [4096, 10_000] and info["max"] == 131072
    assert info["available_mib"] == 14416 and info["auto_max"] == 16384


def test_context_setting_reloads_the_model(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup
    assert comps.server.last_config.context == 8192   # llama-fit-params unavailable: the old default
    start = len(rec.events)
    engine.update_settings({"context_auto": False, "context": 12288})
    rec.wait_for(lambda ev: any(e["type"] == "model_status" for e in ev[start:]))
    assert comps.server.last_config.context == 12288 and engine.context == 12288
    status = [e for e in rec.events[start:] if e["type"] == "model_status"][-1]
    assert status["context"] == 12288 and status["context_auto"] is False
    start = len(rec.events)
    engine.update_settings({"context": 5000})
    rec.wait_for(lambda ev: any(e["type"] == "error" for e in ev[start:]))
    assert engine.store.settings.context == 12288   # rolled back
