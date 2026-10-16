"""Group chats and story mode: the group module and the engine flow with fake components."""

from __future__ import annotations

from typing import Any

import pytest
from test_engine_characters import MemoryLLM, _paths, _wait
from test_engine import FakeServer, FakeSTT, FakeTTS, FakeVAD, FakeVoices, Recorder

from lantern_talk.characters import Character
from lantern_talk.engine import Components, Engine
from lantern_talk.group import (
    NamePrefixFilter, build_group_prompt, cast_title, group_key, group_messages, intro, mentioned, next_speaker,
    stop_sequences,
)
from lantern_talk.llm import ChatMessage
from lantern_talk.settings import SettingsStore


def _char(cid: str, name: str, **kw: Any) -> Character:
    return Character(id=cid, name=name, **kw)


LENA, MAX, HALE = _char("lena", "Lena"), _char("max", "Max"), _char("professor-hale", "Professor Edmund Hale")


# -- the group module -------------------------------------------------------------------------------

def test_group_key_and_title() -> None:
    assert group_key(["max", "lena"]) == group_key(["lena", "max"]) == "group:lena+max"
    assert cast_title(["Lena"]) == "Lena"
    assert cast_title(["Lena", "Max", "Elara"]) == "Lena, Max and Elara"


def test_mentioned_finds_capitalized_names_in_order() -> None:
    members = [LENA, MAX, HALE]
    assert mentioned("What do you think, Professor? And you, Max?", members) == ["professor-hale", "max"]
    assert mentioned("Turn it up to the max.", members) == []           # not a name
    assert mentioned("Hale is right, Lena.", members, exclude=["lena"]) == ["professor-hale"]


def test_next_speaker_prefers_the_addressed_then_the_quietest() -> None:
    members = [LENA, MAX, HALE]
    history = [ChatMessage("assistant", "Hello there.", name="lena")]
    assert next_speaker(members, history) == "max"          # never Lena twice; Max is next in line
    history.append(ChatMessage("assistant", "Hi. Professor, your turn.", name="max"))
    assert next_speaker(members, history) == "professor-hale"  # addressed
    history.append(ChatMessage("assistant", "Indeed.", name="professor-hale"))
    assert next_speaker(members, history) == "lena"         # quiet the longest
    assert next_speaker(members, []) == "lena"


def test_history_from_each_speakers_point_of_view() -> None:
    history = [ChatMessage("user", "Hi both!"), ChatMessage("assistant", "Hi!", name="lena"),
               ChatMessage("assistant", "Hey.", name="max")]
    names = {"lena": "Lena", "max": "Max"}
    as_max = group_messages("max", "SYS", history, names, "Tom")
    assert [(m.role, m.content) for m in as_max] == [
        ("system", "SYS"), ("user", "Tom: Hi both!\n\nLena: Hi!"), ("assistant", "Hey."), ("user", "(continue)")]
    as_lena = group_messages("lena", "SYS", history, names)
    assert [(m.role, m.content) for m in as_lena[1:]] == [("user", "User: Hi both!"), ("assistant", "Hi!"),
                                                         ("user", "Max: Hey.")]
    first = group_messages("lena", "SYS", [], names)
    assert first[-1].content.startswith("(The scene begins")


def test_prompt_is_tailored_to_the_speaker() -> None:
    vivienne = _char("vivienne", "Vivienne", system_prompt="You are Vivienne, owner of a bar.",
                     speaking_style="Slow and smoky.", group_style="Owns the room.")
    max_ = _char("max", "Max", profile="Age 33. Laid-back and quick-witted. " + "More. " * 80)
    prompt = build_group_prompt(vivienne, [max_], default_persona="X", story=True, premise="A storm at night.",
                                user_name="Tom", director="Director: be bold.", reply_language="en")
    assert prompt.startswith("You are Vivienne")
    assert "How you speak: Slow and smoky." in prompt and "In a group: Owns the room." in prompt
    assert "With you in the scene:\n- Max: Age 33. Laid-back and quick-witted." in prompt
    assert "Never repeat or paraphrase" in prompt and "A storm at night." in prompt
    assert "Speak only as Vivienne" in prompt and "for Tom" in prompt and "Reply in English." in prompt
    chat = build_group_prompt(vivienne, [max_], default_persona="X", story=False, story_rules="CUSTOM")
    assert "group conversation" in chat and "CUSTOM" not in chat
    assert "CUSTOM" in build_group_prompt(vivienne, [max_], default_persona="X", story=True, story_rules="CUSTOM")
    assert len(intro(max_)) <= 240


@pytest.mark.parametrize("chunks, text", [(["Max: Well, ", "hello."], "Well, hello."),
                                          (["**Max:** Sure", " thing."], "Sure thing."),
                                          (["Maximum effort!"], "Maximum effort!"),
                                          (["Hi"], "Hi")])
def test_name_prefix_is_removed(chunks: list[str], text: str) -> None:
    f = NamePrefixFilter("Max")
    assert "".join(f.feed(c) for c in chunks) + f.finish() == text


def test_stop_sequences_cover_the_others_and_the_user() -> None:
    assert stop_sequences("max", {"lena": "Lena", "max": "Max"}, "Tom") == ["\nLena:", "\nTom:"]


# -- the engine ---------------------------------------------------------------------------------------

class GroupLLM(MemoryLLM):
    """Each character answers with its own line; Max always passes the word to Lena."""

    lines = {"Lena": "I'm not so sure about that.", "Max": "Max: Ha! Lena will disagree.",
             "Elara": "*wipes a glass* Rumors, rumors.", "Captain Vargas": "Stay sharp."}

    def __init__(self) -> None:
        super().__init__()
        self.samplings: list[Any] = []

    def stream_chat(self, messages: list[Any], sampling: Any = None):  # noqa: ANN201
        self.requests.append(messages)
        self.samplings.append(sampling)
        who = next((n for n in self.lines if messages[0].content.startswith(f"You are {n.split()[0]}")), "Lena")
        text = self.lines[who]
        for i in range(0, len(text), 5):
            yield text[i: i + 5]


class AutoPlay(Recorder):
    """The 'browser' reports the end of playback for every reply right away."""

    engine: Engine | None = None

    def __call__(self, msg: dict[str, Any], binary: bytes | None) -> None:
        super().__call__(msg, binary)
        if msg["type"] == "turn_done" and self.engine is not None:
            self.engine.playback_ended()


@pytest.fixture()
def group(tmp_path, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN001, ANN201
    from lantern_talk import engine as engine_module

    monkeypatch.setattr(engine_module, "query_gpu", lambda: {"name": "Test GPU", "used_mb": 6000, "total_mb": 24000})
    started: list[Engine] = []

    def start() -> tuple[Engine, Components, AutoPlay]:
        paths = _paths(tmp_path)
        comps = Components(stt=FakeSTT(), vad=FakeVAD(), tts=FakeTTS(), voices=FakeVoices(),
                           server=FakeServer(), llm=GroupLLM())
        rec = AutoPlay()
        engine = Engine(paths, SettingsStore(paths.settings_file), comps, rec)
        rec.engine = engine
        engine.start()
        rec.wait_state("listening")
        started.append(engine)
        return engine, comps, rec

    yield start
    for engine in started:
        engine.shutdown()


def _join(engine: Engine, rec: AutoPlay, ids: list[str]) -> None:
    start = len(rec.events)
    engine.set_group(ids)
    _wait(rec, "character", start, lambda e: len(e["group"]["members"]) == len(ids) + 1)


def _say(engine: Engine, rec: AutoPlay, text: str) -> list[dict[str, Any]]:
    start = len(rec.events)
    engine.submit_text(text)
    _wait(rec, "editable", start)
    return rec.events[start:]


def test_group_chat_answers_the_addressed_character_and_chains(group) -> None:  # noqa: ANN001
    engine, comps, rec = group()
    _join(engine, rec, ["max"])
    assert [m.id for m in engine.members] == ["lena", "max"]
    assert engine.transcript == []  # a group starts with you (no greeting)
    events = _say(engine, rec, "Hello Max!")
    starts = [e for e in events if e["type"] == "assistant_start"]
    assert [e["speaker"] for e in starts] == ["max", "lena"]  # Max answers, and his reply addresses Lena
    assert starts[0]["voice"] == "builtin:en_male" and starts[1]["voice"] == "builtin:en_female"
    assert engine.transcript[1]["segments"][0]["text"] == "Ha! Lena will disagree."  # no "Max:" in front
    max_request, lena_request = comps.llm.requests[-2:]
    assert max_request[0].content.startswith("You are Max") and "How you speak:" in max_request[0].content
    assert "With you in the scene:\n- Lena:" in max_request[0].content
    assert [m.content for m in lena_request[1:]] == ["User: Hello Max!\n\nMax: Ha! Lena will disagree."]
    assert "\nMax:" in comps.llm.samplings[-1].stop and comps.llm.samplings[-1].dry_multiplier > 0
    # Everything is in the group's memory, with the speaker of every line.
    stored = engine.memory.active_messages("group:lena+max")
    assert [(m.role, m.speaker) for m in stored] == [("user", None), ("assistant", "max"), ("assistant", "lena")]
    assert engine.memory.active_messages("lena") == [] or all(m.role == "assistant"
                                                              for m in engine.memory.active_messages("lena"))


def test_group_chat_survives_a_restart_and_reopens(group) -> None:  # noqa: ANN001
    engine, comps, rec = group()
    _join(engine, rec, ["max"])
    _say(engine, rec, "Hi there.")
    engine.shutdown()
    engine2, comps2, rec2 = group()
    assert [m.id for m in engine2.members] == ["lena", "max"]
    assert [t.get("speaker") for t in engine2.transcript] == [None, "lena"]
    assert engine2.resume is not None
    assert engine2.conversation.history[1].name == "lena"
    # Picking Lena on the left again: a chat with her alone; the group is in the recent groups.
    start = len(rec2.events)
    engine2.select_character("lena")
    _wait(rec2, "character", start, lambda e: e["group"]["members"] == ["lena"])
    assert engine2.snapshot()["group"]["recent"] == [["lena", "max"]]
    start = len(rec2.events)
    engine2.open_group(["lena", "max"])
    _wait(rec2, "character", start, lambda e: e["group"]["members"] == ["lena", "max"])
    assert len(engine2.transcript) == 2


def test_story_mode_plays_until_paused(group) -> None:  # noqa: ANN001
    engine, comps, rec = group()
    _join(engine, rec, ["elara"])
    start = len(rec.events)
    engine.set_premise("A storm keeps everyone in the tavern.")
    engine.story_play()
    rec.wait_for(lambda ev: sum(e["type"] == "assistant_start" for e in ev[start:]) >= 4)
    engine.story_pause()
    _wait(rec, "group", start, lambda e: not e["playing"])
    rec.wait_state("listening", len(rec.events) - 1)
    speakers = [e["speaker"] for e in rec.events[start:] if e["type"] == "assistant_start"]
    assert all(a != b for a, b in zip(speakers, speakers[1:]))  # they take turns
    story_request = [r for r in comps.llm.requests if r[0].content.startswith("You are Elara")][-1]
    assert "A storm keeps everyone in the tavern." in story_request[0].content
    assert "Never repeat or paraphrase" in story_request[0].content
    assert engine.store.settings.group_mode == "story"
    count = len(engine.transcript)
    import time

    time.sleep(0.3)
    assert len(engine.transcript) == count  # paused: nobody goes on


def test_director_notes_for_one_character_and_step(group) -> None:  # noqa: ANN001
    engine, comps, rec = group()
    _join(engine, rec, ["max"])
    start = len(rec.events)
    engine.set_pending_note("Max confesses a secret.", "max")
    engine.pin_note("It is late at night.")
    _wait(rec, "director", start, lambda e: e["group"] and len(e["pinned"]) == 1 and e["pending"])
    start = len(rec.events)
    engine.continue_story("lena")  # Lena speaks first: Max's note waits for him
    _wait(rec, "editable", start)
    lena = comps.llm.requests[-1][0].content
    assert "It is late at night." in lena and "Max confesses" not in lena
    start = len(rec.events)
    engine.continue_story()
    _wait(rec, "editable", start)
    max_ = comps.llm.requests[-1][0].content
    assert max_.startswith("You are Max") and "For your next reply: Max confesses a secret." in max_
    assert engine.transcript[-1]["note"] == "Max confesses a secret."
    # Write the last line again: the same character, with the same note.
    start = len(rec.events)
    engine.regenerate()
    _wait(rec, "editable", start)
    assert [t["speaker"] for t in engine.transcript] == ["lena", "max"]
    assert "Max confesses a secret." in comps.llm.requests[-1][0].content


def test_end_chat_in_a_group_clears_at_once_and_summarizes(group) -> None:  # noqa: ANN001
    engine, comps, rec = group()
    _join(engine, rec, ["max"])
    _say(engine, rec, "Hello Max!")
    start = len(rec.events)
    engine.new_conversation()
    ev = _wait(rec, "character", start)
    assert ev["transcript"] == []
    _wait(rec, "memory", start, lambda e: bool(e["summary"]))
    assert engine.memory.summary("group:lena+max") == MemoryLLM.summary_reply
    assert engine.memory.summary("lena") == ""


def test_write_instructions_from_a_profile(group) -> None:  # noqa: ANN001
    engine, comps, rec = group()
    comps.llm.side_replies = []
    comps.llm.complete = lambda messages, sampling=None: (  # type: ignore[method-assign]
        'Sure: {"system_prompt": "You are Hilde, 72, a baker.", "speaking_style": "Warm.", '
        '"group_style": "Mothers everyone.", "persona": "Baker from the Black Forest.", "greeting": "Hello, dear!"}')
    start = len(rec.events)
    engine.draft_character({"name": "Hilde", "kind": "roleplay", "profile": "Hilde, 72, Bäckerin, warmherzig."})
    draft = _wait(rec, "character_draft", start)["draft"]
    assert draft["system_prompt"].startswith("You are Hilde") and draft["greeting"] == "Hello, dear!"
    start = len(rec.events)
    engine.draft_character({"name": "Hilde", "kind": "roleplay", "profile": ""})
    assert "profile" in _wait(rec, "error", start)["message"]


def test_settings_cannot_change_the_cast(group) -> None:  # noqa: ANN001
    engine, comps, rec = group()
    start = len(rec.events)
    engine.update_settings({"group": ["max"], "story_rules": "Short lines only."})
    _wait(rec, "settings", start, lambda e: e["settings"]["story_rules"] == "Short lines only.")
    assert engine.store.settings.group == []
