"""Characters: shipped examples, create / update / duplicate / delete (spec §10)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lantern_talk.characters import CharacterError, CharacterStore, initials_for, slugify

ROOT = Path(__file__).parents[1]
DEFAULTS = ROOT / "config" / "default_characters"


@pytest.fixture()
def store(tmp_path: Path) -> CharacterStore:
    s = CharacterStore(tmp_path / "characters", DEFAULTS)
    s.load()
    return s


def test_shipped_examples(store: CharacterStore) -> None:
    chars = {c.id: c for c in store.list()}
    assert set(chars) == {"max", "lena", "elara", "captain-vargas"}
    assert chars["max"].kind == chars["lena"].kind == "assistant"
    assert chars["elara"].kind == chars["captain-vargas"].kind == "roleplay"
    assert chars["max"].voice == "builtin:de_male" and chars["lena"].voice == "builtin:de_female"
    assert [c.kind for c in store.list()][:2] == ["assistant", "assistant"]  # assistants first
    for c in chars.values():
        assert c.greeting and c.system_prompt and len(c.avatar_initials) == 2


def test_deleted_examples_are_not_copied_again(store: CharacterStore) -> None:
    store.delete("elara")
    store.load()
    assert "elara" not in {c.id for c in store.list()}


def test_create_update_duplicate(store: CharacterStore) -> None:
    c = store.save({"name": "Großmutter Hilde", "kind": "roleplay", "system_prompt": "You are Hilde."})
    assert c.id == "grossmutter-hilde"
    assert c.avatar_initials == "GH"
    on_disk = json.loads((store.folder / f"{c.id}.json").read_text(encoding="utf-8"))
    assert on_disk["name"] == "Großmutter Hilde"
    updated = store.save({**c.to_dict(), "greeting": "Na, mein Kind?"}, c.id)
    assert updated.greeting == "Na, mein Kind?"
    copy = store.duplicate(c.id)
    assert copy.id != c.id and copy.name.endswith("(copy)") and copy.system_prompt == "You are Hilde."


def test_unique_ids(store: CharacterStore) -> None:
    assert store.save({"name": "Max"}).id == "max-2"


@pytest.mark.parametrize("data, message", [
    ({"name": ""}, "needs a name"),
    ({"name": "x" * 61}, "too long"),
    ({"name": "X", "kind": "villain"}, "Kind must be"),
    ({"name": "X", "system_prompt": "a" * 9000}, "too long"),
])
def test_validation(store: CharacterStore, data: dict, message: str) -> None:
    with pytest.raises(CharacterError, match=message):
        store.save(data)


def test_cannot_delete_last(store: CharacterStore) -> None:
    for c in store.list()[1:]:
        store.delete(c.id)
    with pytest.raises(CharacterError, match="last character"):
        store.delete(store.list()[0].id)


def test_broken_file_is_skipped(store: CharacterStore) -> None:
    (store.folder / "broken.json").write_text("{nope", encoding="utf-8")
    store.load()
    assert "broken" not in {c.id for c in store.list()}


def test_helpers() -> None:
    assert initials_for("Captain Vargas") == "CV" and initials_for("max") == "MA"
    assert slugify("Élan Über!") == "elan-ueber" and slugify("") == "character"
