"""Session recording in the engine (spec §12), with fake speech components and the real ffmpeg."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from test_engine import loud, quiet
from test_engine_characters import _turn, _wait, setup  # noqa: F401 - the fixture is used by name
from test_recorder import FFMPEG, needs_ffmpeg

from lantern_talk.recorder import load_session, mp3_duration
from lantern_talk.vad import FRAME_SAMPLES, SegmenterConfig, Utterance, UtteranceSegmenter

pytestmark = needs_ffmpeg


def _start(setup):  # noqa: ANN001, ANN202
    engine, comps, rec = setup()
    engine.recorder.ffmpeg = FFMPEG
    return engine, comps, rec


def _session(engine) -> tuple[Path, dict]:  # noqa: ANN001
    engine.recorder.wait_idle()
    name = engine.memory.get_meta("lena", "session")
    return engine.paths.sessions / f"{name}.mp3", load_session(engine.paths.sessions, name)


def test_typed_turn_records_greeting_and_reply(setup) -> None:  # noqa: F811
    engine, comps, rec = _start(setup)
    start = len(rec.events)
    _turn(engine, rec, "Hallo Lena!")
    mp3, session = _session(engine)
    assert session["title"].startswith("Lena — ")
    roles = [(e["role"], e["tags"][0]) for e in session["entries"]]
    assert roles == [("assistant", "greeting"), ("user", "typed"), ("assistant", "German female")]
    reply = session["entries"][2]
    assert reply["duration"] and reply["duration"] > 0
    assert mp3_duration(mp3.read_bytes()) == pytest.approx(reply["offset"] + reply["duration"], abs=0.002)
    # the replay view still shows the actions; the Markdown has only the spoken words
    assert reply["segments"][0] == {"kind": "action", "text": "lächelt"}
    md = mp3.with_suffix(".md").read_text(encoding="utf-8").split("\n\n", 1)[1]
    assert md == ("Lena: Hi, I'm Lena! Great to have you here. What's on your mind today?\n\n"
                  "You: Hallo Lena!\n\nLena: Mir geht es gut. Und dir?\n\n")
    # the browser learns where each message is
    recorded = [e for e in rec.events[start:] if e["type"] == "recorded"]
    assert {e["id"] for e in recorded} >= {engine.transcript[-1]["id"]}
    sessions = [e for e in rec.events[start:] if e["type"] == "sessions"]
    assert sessions and sessions[-1]["recording"]["mp3"] == mp3.name


def test_spoken_turn_records_full_quality_user_audio(setup) -> None:  # noqa: F811
    engine, comps, rec = _start(setup)
    start = len(rec.events)
    engine.update_settings({"session_gap_ms": 200})
    _wait(rec, "settings", start, lambda e: e["settings"]["session_gap_ms"] == 200)
    rec.wait_state("listening", start)
    engine.on_audio(quiet(), 48_000)          # lets the audio thread finish its reset (it drops queued audio)
    time.sleep(0.2)
    start = len(rec.events)
    for _ in range(5):
        engine.on_audio(loud(), 48_000)       # 0.5 s speech at 48 kHz
    for _ in range(12):
        engine.on_audio(quiet(), 48_000)
    rec.wait_for(lambda ev: any(e["type"] == "turn_done" for e in ev[start:]))
    engine.playback_ended()
    rec.wait_for(lambda ev: any(e["type"] == "editable" for e in ev[start:]))
    _, session = _session(engine)
    user = session["entries"][1]
    assert user["role"] == "user" and user["source"] == "voice" and "de" in user["tags"]
    assert 0.5 <= user["duration"] < 1.6       # speech + kept tail, not the 16 kHz copy's silence
    exact = {e["id"]: e for e in rec.events[start:] if e["type"] == "recorded"}
    user_id, reply_id = engine.transcript[-2]["id"], engine.transcript[-1]["id"]
    assert exact[reply_id]["offset"] == pytest.approx(
        exact[user_id]["offset"] + exact[user_id]["duration"] + 0.2, abs=0.001)


def test_replay_info_survives_a_restart(setup) -> None:  # noqa: F811
    engine, comps, rec = _start(setup)
    _turn(engine, rec, "Hallo!")
    engine.recorder.wait_idle()
    name = engine.memory.get_meta("lena", "session")
    engine.shutdown()
    engine2, _, rec2 = _start(setup)
    reply = engine2.transcript[-1]
    assert reply["session"] == name and reply["offset"] is not None and reply["duration"] > 0
    # continuing the chat appends to the same recording
    _turn(engine2, rec2, "Noch da?")
    _, session = _session(engine2)
    assert len(session["entries"]) == 5


def test_end_chat_starts_a_new_recording(setup) -> None:  # noqa: F811
    engine, comps, rec = _start(setup)
    _turn(engine, rec, "Hallo!")
    first = engine.memory.get_meta("lena", "session")
    start = len(rec.events)
    engine.new_conversation()
    _wait(rec, "character", start)
    assert engine.recorder.name is None
    _turn(engine, rec, "Neuer Chat.")
    second = engine.memory.get_meta("lena", "session")
    assert second and second != first
    assert len(engine.snapshot()["sessions"]) == 2


def test_recording_can_be_switched_off(setup) -> None:  # noqa: F811
    engine, comps, rec = _start(setup)
    start = len(rec.events)
    engine.update_settings({"record_sessions": False})
    _wait(rec, "settings", start, lambda e: not e["settings"]["record_sessions"])
    _turn(engine, rec, "Hallo!")
    assert engine.recorder.name is None
    assert not engine.paths.sessions.exists() or not list(engine.paths.sessions.glob("*.md"))


def test_replay_mutes_the_microphone(setup) -> None:  # noqa: F811
    engine, comps, rec = _start(setup)
    start = len(rec.events)
    engine.set_replaying(True)
    rec.wait_for(lambda ev: any(e["type"] == "state" and e["state"] == "paused" for e in ev[start:]))
    assert any(e.get("detail") == "Replay is playing" for e in rec.events[start:] if e["type"] == "state")
    engine.set_replaying(False)
    rec.wait_state("listening", start)


def test_segmenter_reports_stream_positions() -> None:
    class LoudVAD:
        def reset(self) -> None:
            pass

        def speech_probability(self, frame: np.ndarray) -> float:
            return 0.9 if float(np.abs(frame).mean()) > 0.1 else 0.0

    seg = UtteranceSegmenter(LoudVAD(), SegmenterConfig(end_silence_ms=300, pre_roll_ms=0))  # type: ignore[arg-type]
    silence = np.zeros(FRAME_SAMPLES * 10, dtype=np.float32)
    speech = np.full(FRAME_SAMPLES * 20, 0.5, dtype=np.float32)
    out: list[Utterance] = []
    for block in (silence, speech, silence, silence):
        out += seg.feed(block)
    assert len(out) == 1
    utt = out[0]
    assert utt.end - utt.start == len(utt.audio)
    assert utt.start == FRAME_SAMPLES * 10  # speech began after 10 frames of silence



def _wait_transcript(engine, rec, start: int) -> dict:  # noqa: ANN001
    return _wait(rec, "transcript", start)


def test_deleting_the_end_of_a_chat_cuts_memory_history_and_recording(setup) -> None:  # noqa: F811
    engine, comps, rec = _start(setup)
    for text in ("Erste Frage.", "Zweite Frage.", "Dritte Frage."):
        _turn(engine, rec, text)
    mp3, session = _session(engine)
    assert len(session["entries"]) == 7  # greeting + 3 x (question, reply)
    second = next(t for t in engine.transcript if t.get("text") == "Zweite Frage.")
    start = len(rec.events)
    engine.delete_from(second["id"])
    ev = _wait_transcript(engine, rec, start)
    assert [t.get("text") for t in ev["transcript"] if t["role"] == "user"] == ["Erste Frage."]
    assert ev["deletable_from"] == ev["transcript"][1]["id"]  # the greeting stays
    assert [m.content for m in engine.memory.active_messages("lena")][1:] == ["Erste Frage.", comps.llm.reply]
    assert len(engine.conversation.history) == 3
    mp3, session = _session(engine)
    assert len(session["entries"]) == 3
    last = session["entries"][-1]
    assert mp3_duration(mp3.read_bytes()) == pytest.approx(last["offset"] + last["duration"], abs=0.05)
    md = mp3.with_suffix(".md").read_text(encoding="utf-8")
    assert "Erste Frage." in md and "Zweite Frage." not in md and md.startswith("# Lena")
    # Talking on appends right after the cut.
    _turn(engine, rec, "Neue Frage.")
    mp3, session = _session(engine)
    assert [e.get("text") for e in session["entries"] if e["role"] == "user"] == ["Erste Frage.", "Neue Frage."]
    assert session["entries"][-1]["offset"] > last["offset"] + last["duration"]


def test_regenerate_and_edit_the_last_message(setup) -> None:  # noqa: F811
    engine, comps, rec = _start(setup)
    _turn(engine, rec, "Wie heißt du?")
    comps.llm.reply = "Ich bin Lena, schön dich zu sehen."
    start = len(rec.events)
    engine.regenerate()
    rec.wait_for(lambda ev: any(e["type"] == "turn_done" for e in ev[start:]))
    engine.playback_ended()
    _wait(rec, "editable", start)
    assert not any(e["type"] == "user_message" for e in rec.events[start:])  # the question stays as it was
    assert [m.content for m in engine.memory.active_messages("lena")][1:] == [
        "Wie heißt du?", "Ich bin Lena, schön dich zu sehen."]
    mp3, session = _session(engine)
    assert [e["role"] for e in session["entries"]] == ["assistant", "user", "assistant"]
    start = len(rec.events)
    engine.edit_last("Wie alt bist du?")
    rec.wait_for(lambda ev: any(e["type"] == "turn_done" for e in ev[start:]))
    engine.playback_ended()
    _wait(rec, "editable", start)
    users = [t["text"] for t in engine.transcript if t["role"] == "user"]
    assert users == ["Wie alt bist du?"]
    assert [m.role for m in engine.conversation.history] == ["assistant", "user", "assistant"]


def test_summarized_messages_cannot_be_deleted(setup) -> None:  # noqa: F811
    engine, comps, rec = _start(setup)
    engine._loaded_context = 256  # tiny budget: old turns are summarized quickly
    for i in range(6):
        _turn(engine, rec, f"Frage {i}.")
    first_user = next(t for t in engine.transcript if t["role"] == "user")
    assert engine._deletable_from() > first_user["id"]
    start = len(rec.events)
    engine.delete_from(first_user["id"])
    assert "summary" in _wait(rec, "error", start)["message"]
