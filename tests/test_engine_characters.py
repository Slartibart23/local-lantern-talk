"""Characters and memory in the engine (spec §10 / §11), with fake components."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from test_engine import ROOT, FakeLLM, FakeServer, FakeSTT, FakeTTS, FakeVAD, FakeVoices, Recorder

from lantern_talk.engine import Components, Engine
from lantern_talk.paths import Paths
from lantern_talk.settings import SettingsStore


class MemoryLLM(FakeLLM):
    """Answers memory upkeep requests: summaries and fact updates."""

    summary_reply = "Tom hat Lena von seinem Hund Bello erzählt."
    facts_reply = '{"add": ["The user\'s name is Tom."], "remove": []}'

    def complete(self, messages: list[Any], sampling: Any = None) -> str:
        self.requests.append(messages)
        self.sampling_used = sampling
        return self.summary_reply if "long-term memory" in messages[0].content else self.facts_reply


def _paths(tmp_path: Path) -> Paths:
    root = tmp_path / "root"
    if not root.exists():
        (root / "app").mkdir(parents=True)
        (root / "config").mkdir()
        shutil.copy(ROOT / "config" / "model_catalog.yaml", root / "config" / "model_catalog.yaml")
        shutil.copytree(ROOT / "config" / "default_characters", root / "config" / "default_characters")
        paths = Paths(root)
        paths.llm_models.mkdir(parents=True)
        (paths.llm_models / "test-model-Q4_K_M.gguf").write_bytes(b"x")
        (paths.llm_models / "other-model-Q4_K_M.gguf").write_bytes(b"x")
    return Paths(root)


def _start(paths: Paths) -> tuple[Engine, Components, Recorder]:
    store = SettingsStore(paths.settings_file)
    comps = Components(stt=FakeSTT(), vad=FakeVAD(), tts=FakeTTS(), voices=FakeVoices(),
                       server=FakeServer(), llm=MemoryLLM())
    rec = Recorder()
    engine = Engine(paths, store, comps, rec)
    engine.start()
    rec.wait_state("listening")
    return engine, comps, rec


@pytest.fixture()
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    from lantern_talk import engine as engine_module

    monkeypatch.setattr(engine_module, "query_gpu",
                        lambda: {"name": "Test GPU", "used_mb": 6000, "total_mb": 24000})
    started: list[Engine] = []

    def start() -> tuple[Engine, Components, Recorder]:
        engine, comps, rec = _start(_paths(tmp_path))
        started.append(engine)
        return engine, comps, rec

    yield start
    for engine in started:
        engine.shutdown()


def _turn(engine: Engine, rec: Recorder, text: str) -> None:
    """One typed turn including the memory upkeep that follows it ("editable" comes last)."""
    start = len(rec.events)
    engine.submit_text(text)
    rec.wait_for(lambda ev: any(e["type"] == "turn_done" for e in ev[start:]))
    engine.playback_ended()
    rec.wait_for(lambda ev: any(e["type"] == "editable" for e in ev[start:]))


def _wait(rec: Recorder, kind: str, start: int, check=lambda e: True) -> dict[str, Any]:  # noqa: ANN001
    rec.wait_for(lambda ev: any(e["type"] == kind and check(e) for e in ev[start:]))
    return next(e for e in reversed(rec.events) if e["type"] == kind and check(e))


def test_startup_activates_first_assistant_with_greeting(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup()
    snap = engine.snapshot()
    assert snap["character"]["id"] == "lena"  # assistants first, then by name
    assert [c["id"] for c in snap["characters"]][:2] == ["lena", "max"]
    assert engine.store.settings.character == "lena"
    assert engine.store.settings.voice == "builtin:en_female"
    first = snap["transcript"][0]
    assert first["role"] == "assistant" and "Lena" in first["segments"][0]["text"]
    assert "You are Lena" in engine.conversation.config.persona
    assert engine.conversation.config.roleplay is False


def test_greeting_never_starts_the_llm_history(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup()
    _turn(engine, rec, "Hallo Lena!")
    messages = comps.llm.requests[0]
    assert [m.role for m in messages] == ["system", "user"]
    assert "You opened the conversation by saying" in messages[0].content


def test_switching_character_switches_persona_voice_and_history(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup()
    _turn(engine, rec, "Hallo Lena!")
    start = len(rec.events)
    engine.select_character("captain-vargas")
    ev = _wait(rec, "character", start)
    assert ev["character"]["id"] == "captain-vargas"
    assert engine.store.settings.voice == "builtin:en_male"
    assert engine.conversation.config.roleplay is True
    assert "Vargas" in engine.conversation.config.persona
    assert [t["role"] for t in ev["transcript"]] == ["assistant"]  # only Vargas' greeting
    # back to Lena: her conversation is still there
    start = len(rec.events)
    engine.select_character("lena")
    ev = _wait(rec, "character", start)
    assert [t["role"] for t in ev["transcript"]] == ["assistant", "user", "assistant"]
    assert ev["transcript"][1]["text"] == "Hallo Lena!"


def test_character_model_is_loaded_on_switch(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup()
    data = engine.characters.get("max").to_dict()
    data["model"] = "other-model-Q4_K_M.gguf"
    engine.characters.save(data, "max")
    start = len(rec.events)
    engine.select_character("max")
    _wait(rec, "character", start)
    assert comps.server.started_with[-1] == "other-model-Q4_K_M.gguf"


def test_history_survives_a_restart(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup()
    _turn(engine, rec, "Ich heiße Tom.")
    engine.shutdown()
    engine2, _, _ = setup()
    snap = engine2.snapshot()
    roles = [t["role"] for t in snap["transcript"]]
    assert roles == ["assistant", "user", "assistant"]
    assert snap["resume"]["messages"] == 3 and snap["resume"]["last"] > 0  # UI offers Continue / New chat
    assert [m.role for m in engine2.conversation.history] == ["assistant", "user", "assistant"]


def test_facts_are_updated_every_few_turns_and_reach_the_prompt(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup()
    for i in range(4):
        _turn(engine, rec, f"Nachricht {i}")
    facts = engine.memory_view()["facts"]
    assert [f["text"] for f in facts] == ["The user's name is Tom."]
    assert facts[0]["source"] == "llm"
    assert comps.llm.sampling_used.temperature < 0.5
    _turn(engine, rec, "Weißt du noch, wie ich heiße?")
    assert "The user's name is Tom." in comps.llm.requests[-1][0].content


def test_long_history_is_summarized(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup()
    engine._loaded_context = 256  # tiny budget: a few turns overflow it
    for i in range(6):
        _turn(engine, rec, f"Erzähl mir etwas, Nummer {i}.")
    view = engine.memory_view()
    assert view["summary"] == MemoryLLM.summary_reply
    active = engine.memory.active_messages("lena")
    assert len(active) < 13 and active[0].role == "user"
    assert len(engine.conversation.history) == len(active)
    assert len(engine.transcript) == 13  # the visible conversation stays complete
    _turn(engine, rec, "Und weiter?")
    assert MemoryLLM.summary_reply in comps.llm.requests[-1][0].content


def test_new_conversation_moves_the_talk_into_memory(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup()
    _turn(engine, rec, "Mein Hund heißt Bello.")
    start = len(rec.events)
    engine.new_conversation()
    ev = _wait(rec, "character", start)
    assert [t["role"] for t in ev["transcript"]] == ["assistant"]  # fresh greeting, before the summary is written
    assert ev["resume"] is None
    memory = _wait(rec, "memory", start, lambda e: bool(e["summary"]))  # written afterwards
    assert memory["summary"] == MemoryLLM.summary_reply
    rec.wait_state("listening", start)
    assert engine.memory.summary("lena") in engine.conversation.config.memory


def test_fact_editing_and_reset(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup()
    start = len(rec.events)
    engine.add_fact("The user likes jazz.")
    ev = _wait(rec, "memory", start, lambda e: e["facts"])
    fact = ev["facts"][0]
    assert fact["source"] == "user"
    assert "likes jazz" in engine.conversation.config.memory
    start = len(rec.events)
    engine.update_fact(fact["id"], "The user loves jazz.")
    ev = _wait(rec, "memory", start)
    assert ev["facts"][0]["text"] == "The user loves jazz."
    start = len(rec.events)
    engine.delete_fact(fact["id"])
    ev = _wait(rec, "memory", start)
    assert ev["facts"] == []
    _turn(engine, rec, "Hallo!")
    engine.add_fact("Something")
    start = len(rec.events)
    engine.reset_memory()
    ev = _wait(rec, "character", start)
    assert ev["memory"]["facts"] == [] and ev["memory"]["summary"] == ""
    assert [t["role"] for t in ev["transcript"]] == ["assistant"]


def test_memory_disabled_keeps_nothing(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup()
    data = engine.characters.get("lena").to_dict()
    start = len(rec.events)
    engine.save_character({**data, "memory_enabled": False}, "lena")
    _wait(rec, "character", start, lambda e: not e["character"]["memory_enabled"])
    start = len(rec.events)
    engine.submit_text("Hallo!")
    _wait(rec, "turn_done", start)
    engine.playback_ended()
    rec.wait_state("listening", after=start)
    assert [m.role for m in engine.memory.active_messages("lena")] == ["assistant"]  # stored earlier
    assert engine.conversation.config.memory == ""


def test_create_new_character_activates_it(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup()
    start = len(rec.events)
    engine.save_character({"name": "Großmutter Hilde", "kind": "roleplay",
                           "system_prompt": "You are Hilde, a grandmother who bakes.",
                           "greeting": "Komm rein, Kind!", "voice": "builtin:de_female"})
    ev = _wait(rec, "character", start, lambda e: e["character"]["id"] == "grossmutter-hilde")
    assert ev["character"]["avatar_initials"] == "GH"
    assert ev["transcript"][0]["segments"][0]["text"] == "Komm rein, Kind!"
    saved = json.loads((engine.paths.characters / "grossmutter-hilde.json").read_text(encoding="utf-8"))
    assert saved["system_prompt"].startswith("You are Hilde")


def test_invalid_character_is_rejected(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup()
    start = len(rec.events)
    engine.save_character({"name": "", "kind": "assistant"})
    ev = _wait(rec, "error", start)
    assert "name" in ev["message"].lower()
    start = len(rec.events)
    engine.save_character({"name": "X", "voice": "custom:missing.mp3"})
    _wait(rec, "error", start)
    assert engine.characters.get("x") is None


def test_duplicate_and_delete_active_character(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup()
    _turn(engine, rec, "Hallo!")
    start = len(rec.events)
    engine.duplicate_character("lena")
    ev = _wait(rec, "character", start, lambda e: len(e["characters"]) == 7)
    assert any(c["name"] == "Lena (copy)" for c in ev["characters"])
    start = len(rec.events)
    engine.delete_character("lena")
    ev = _wait(rec, "character", start, lambda e: e["character"]["id"] != "lena")
    assert all(c["id"] != "lena" for c in ev["characters"])
    assert engine.memory.active_messages("lena") == []


def test_side_panel_voice_and_actions_belong_to_the_character(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup()
    start = len(rec.events)
    engine.update_settings({"voice": "builtin:en_male", "show_actions": False})
    _wait(rec, "characters", start)
    lena = engine.characters.get("lena")
    assert lena.voice == "builtin:en_male" and lena.show_actions_as_text is False


def test_character_without_voice_output_replies_as_text(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup()
    data = engine.characters.get("lena").to_dict()
    start = len(rec.events)
    engine.save_character({**data, "voice_output": False}, "lena")
    _wait(rec, "character", start, lambda e: not e["character"]["voice_output"])
    start = len(rec.events)
    engine.submit_text("Hallo!")
    ev = _wait(rec, "turn_done", start)
    assert ev["audio_chunks"] == 0


def test_a_new_turn_continues_the_resumed_chat(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup()
    _turn(engine, rec, "Hallo!")
    engine.shutdown()
    engine2, _, rec2 = setup()
    assert engine2.resume is not None
    _turn(engine2, rec2, "Weiter geht's.")
    assert engine2.resume is None
    assert engine2.snapshot()["resume"] is None


@pytest.mark.parametrize("values", [{"vad_threshold": 1.5}, {"context": 5000}, {"session_bitrate_kbps": 50},
                                    {"session_gap_ms": 9000}, {"hardware_profile_override": "huge"}])
def test_invalid_settings_are_rejected_and_rolled_back(setup, values: dict) -> None:  # noqa: ANN001
    engine, comps, rec = setup()
    before = dict(vars(engine.store.settings))
    start = len(rec.events)
    engine.update_settings(values)
    _wait(rec, "error", start)
    assert vars(engine.store.settings) == before


def test_settings_screen_values_apply(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup()
    start = len(rec.events)
    engine.update_settings({"vad_threshold": 0.6, "session_bitrate_kbps": 96, "session_gap_ms": 800,
                            "hardware_profile_override": "mid"})
    _wait(rec, "settings", start, lambda e: e["settings"]["session_bitrate_kbps"] == 96)
    assert engine.segmenter.config.threshold == 0.6
    assert engine.segmenter.config.neg_threshold == pytest.approx(0.45)
    assert engine.gpu_summary()["profile"] == "mid"
    assert {p["name"] for p in engine.snapshot()["profiles"]} == {"ultra", "high", "mid", "low", "unsupported"}


def test_pictures_belong_to_their_character(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup()
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    avatar = engine.add_image("avatar", png)["name"]
    background = engine.add_image("background", png)["name"]
    with pytest.raises(Exception, match="JPG, PNG or WebP"):
        engine.add_image("avatar", b"GIF89a")
    data = {**engine.characters.get("max").to_dict(), "avatar": avatar, "background": background,
            "background_fade": 120, "text_color": "mint"}
    start = len(rec.events)
    engine.save_character(data, "max")
    _wait(rec, "character", start)
    saved = engine.characters.get("max")
    assert (saved.avatar, saved.background, saved.background_fade, saved.text_color) == (avatar, background, 95, "mint")
    # A duplicate gets copies, so deleting one character never breaks the other.
    start = len(rec.events)
    engine.duplicate_character("max")
    _wait(rec, "character", start)
    copy = next(c for c in engine.characters.list() if c.name == "Max (copy)")
    assert copy.avatar and copy.avatar != avatar and engine.images.path(copy.avatar)
    # Replaced pictures disappear once they are older than an hour.
    engine._fresh_images.clear()
    start = len(rec.events)
    engine.save_character({**saved.to_dict(), "avatar": ""}, "max")
    _wait(rec, "character", start)
    assert engine.images.path(avatar) is None and engine.images.path(background) is not None


def test_your_profile_reaches_every_character(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup()
    start = len(rec.events)
    engine.update_settings({"user_name": "  Alex   Example ", "user_about": "I love sci-fi."})
    rec.wait_for(lambda ev: any(e["type"] == "settings" for e in ev[start:]))
    assert engine.store.settings.user_name == "Alex Example"
    _turn(engine, rec, "Hi!")
    system = comps.llm.requests[-1][0].content
    assert "The user's name is Alex Example." in system and "I love sci-fi." in system
    assert engine._user_label() == "Alex Example"



def test_director_notes(setup) -> None:  # noqa: ANN001
    engine, comps, rec = setup()
    start = len(rec.events)
    engine.pin_note("It is a stormy night.")
    engine.set_pending_note("  Lena gets suspicious.  ")
    _wait(rec, "director", start, lambda e: e["pending"] == "Lena gets suspicious.")
    _turn(engine, rec, "Hello!")
    chat = [r for r in comps.llm.requests if r[-1].content == "Hello!"][0]
    system = chat[0].content
    assert "Always:\n- It is a stormy night." in system and "For your next reply: Lena gets suspicious." in system
    assert "Lena gets suspicious." not in " ".join(m.content for m in chat[1:])  # never part of the history
    assert engine.pending_note == "" and engine.transcript[-2]["note"] == "Lena gets suspicious."
    # The next reply only has the pinned note.
    _turn(engine, rec, "And now?")
    chat = [r for r in comps.llm.requests if r[-1].content == "And now?"][0]
    assert "stormy night" in chat[0].content and "For your next reply" not in chat[0].content
    # A note alone lets the character go on.
    engine.set_pending_note("A stranger walks in.")
    start = len(rec.events)
    engine.continue_story()
    rec.wait_for(lambda ev: any(e["type"] == "turn_done" for e in ev[start:]))
    engine.playback_ended()
    _wait(rec, "editable", start)
    chat = [r for r in comps.llm.requests if r[-1].content == "(continue)"][0]
    assert "For your next reply: A stranger walks in." in chat[0].content
    assert engine.transcript[-2]["source"] == "director" and engine.transcript[-2]["text"] == ""
    # Pinned notes belong to the character and survive a restart, the note of a message too.
    assert engine.characters.get("lena").director_notes == ["It is a stormy night."]
    engine.shutdown()
    engine2, comps2, rec2 = setup()
    notes = [t.get("note") for t in engine2.transcript if t["role"] == "user"]
    assert notes == ["Lena gets suspicious.", None, "A stranger walks in."]
    assert engine2.snapshot()["director"]["pinned"] == ["It is a stormy night."]
    start = len(rec2.events)
    engine2.unpin_note(0)
    _wait(rec2, "director", start)
    assert engine2.characters.get("lena").director_notes == []



def test_export_and_import_a_character(setup) -> None:  # noqa: ANN001
    import json

    engine, comps, rec = setup()
    png = b"\x89PNG\r\n\x1a\n" + b"\x01" * 32
    avatar = engine.add_image("avatar", png)["name"]
    max_char = engine.characters.get("max")
    start = len(rec.events)
    engine.save_character({**max_char.to_dict(), "avatar": avatar, "voice": "builtin:en_male",
                           "director_notes": ["Speaks like a pirate."], "text_color": "lime"}, "max")
    _wait(rec, "character", start)
    _turn(engine, rec, "Remember: my dog is called Bello.")  # memory is never exported
    name, data = engine.export_character("max")
    assert name == "Max.llt-character.json"
    payload = json.loads(data)
    assert payload["format"] == "local-lantern-talk-character" and "id" not in payload["character"]
    assert payload["character"]["avatar"] == "" and payload["pictures"]["avatar"]["file"] == "png"
    assert "Bello" not in data.decode("utf-8")
    start = len(rec.events)
    assert engine.import_character(data) == {"name": "Max (imported)"}
    _wait(rec, "notice", start)
    copy = next(c for c in engine.characters.list() if c.name == "Max (imported)")
    assert copy.id != "max" and copy.director_notes == ["Speaks like a pirate."] and copy.text_color == "lime"
    assert copy.avatar and copy.avatar != avatar and engine.images.path(copy.avatar).read_bytes() == png
    assert engine.memory.active_messages(copy.id) == []
    with pytest.raises(Exception, match="not a Local Lantern Talk character file"):
        engine.import_character(b'{"hello": 1}')
