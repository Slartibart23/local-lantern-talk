"""Memory store and policy (spec §11)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lantern_talk.memory import (
    KEEP_RECENT_MESSAGES, MAX_FACTS, MemoryStore, Message, memory_context, messages_to_summarize,
    parse_fact_update,
)


@pytest.fixture()
def mem(tmp_path: Path):  # noqa: ANN201
    store = MemoryStore(tmp_path / "memory" / "memory.db")
    yield store
    store.close()


def test_history_per_character(mem: MemoryStore) -> None:
    mem.add_message("max", "user", "Hallo Max", "de", "typed")
    mem.add_message("max", "assistant", "Hallo!", "de", None, "builtin:de_male")
    mem.add_message("elara", "user", "Ein Bier, bitte")
    assert [m.content for m in mem.active_messages("max")] == ["Hallo Max", "Hallo!"]
    assert [m.content for m in mem.active_messages("elara")] == ["Ein Bier, bitte"]
    first = mem.active_messages("max")[0]
    mem.mark_summarized("max", first.id)
    assert [m.content for m in mem.active_messages("max")] == ["Hallo!"]


def test_persists_across_instances(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    a = MemoryStore(db)
    a.add_message("max", "user", "Ich heiße Tom")
    a.set_summary("max", "Tom hat sich vorgestellt.")
    a.add_fact("max", "Der Nutzer heißt Tom.")
    a.close()
    b = MemoryStore(db)
    assert b.active_messages("max")[0].content == "Ich heiße Tom"
    assert b.summary("max") == "Tom hat sich vorgestellt."
    assert [f.text for f in b.facts("max")] == ["Der Nutzer heißt Tom."]
    b.close()


def test_facts(mem: MemoryStore) -> None:
    fid = mem.add_fact("max", "  Der Nutzer   heißt Tom. ")
    assert mem.facts("max")[0].text == "Der Nutzer heißt Tom."
    assert mem.add_fact("max", "der nutzer heißt tom.") is None  # duplicate
    mem.update_fact("max", fid, "Der Nutzer heißt Thomas.")
    assert mem.facts("max")[0].text == "Der Nutzer heißt Thomas." and mem.facts("max")[0].source == "user"
    llm_fact = mem.add_fact("max", "Mag Katzen.", source="llm")
    mem.remove_facts_by_text("max", ["Mag Katzen.", "Der Nutzer heißt Thomas."])
    assert [f.text for f in mem.facts("max")] == ["Der Nutzer heißt Thomas."]  # user facts are kept
    assert llm_fact is not None
    mem.delete_fact("max", fid)
    assert mem.facts("max") == []


def test_fact_limit(mem: MemoryStore) -> None:
    for i in range(MAX_FACTS + 5):
        mem.add_fact("x", f"Fact {i}")
    assert len(mem.facts("x")) == MAX_FACTS


def test_reset(mem: MemoryStore) -> None:
    mem.add_message("max", "user", "a")
    mem.add_fact("max", "b")
    mem.set_summary("max", "c")
    mem.set_meta("max", "turns", "3")
    mem.add_message("lena", "user", "keep me")
    mem.reset("max")
    assert mem.active_messages("max") == [] and mem.facts("max") == [] and mem.summary("max") == ""
    assert mem.get_meta("max", "turns", "0") == "0"
    assert mem.active_messages("lena")[0].content == "keep me"


def msgs(n: int, size: int = 400) -> list[Message]:
    return [Message(i, "user" if i % 2 == 0 else "assistant", "x" * size, None, None, None, 0.0) for i in range(n)]


def test_nothing_to_summarize_while_it_fits() -> None:
    assert messages_to_summarize(msgs(10), 8192) == []


def test_summarize_oldest_but_keep_recent() -> None:
    history = msgs(60, 400)  # ~ 6900 tokens > 4096 budget
    old = messages_to_summarize(history, 8192)
    assert old and old[0].id == 0
    assert old[-1].role == "assistant"  # a user message stays with its reply
    assert len(history) - len(old) >= KEEP_RECENT_MESSAGES


@pytest.mark.parametrize("reply, add, remove", [
    ('{"add": ["Der Nutzer heißt Tom."], "remove": []}', ["Der Nutzer heißt Tom."], []),
    ('Sure! ```json\n{"add": [], "remove": ["Mag Katzen."]}\n```', [], ["Mag Katzen."]),
    ("I could not find anything.", [], []),
    ('{"add": "not a list"}', [], []),
    ('{"add": [1, " ", "ok"], "remove": [null]}', ["ok"], []),
])
def test_parse_fact_update(reply: str, add: list[str], remove: list[str]) -> None:
    assert parse_fact_update(reply) == (add, remove)


def test_memory_context() -> None:
    from lantern_talk.memory import Fact

    text = memory_context("We met in the tavern.", [Fact(1, "The user is called Tom.", "llm")])
    assert "- The user is called Tom." in text and "We met in the tavern." in text
    assert memory_context("", []) == ""
