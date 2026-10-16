"""Characters: shipped examples, create / update / duplicate / delete (spec §10)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lantern_talk.catalog import load_catalog
from lantern_talk.characters import CharacterError, CharacterStore, initials_for, slugify

ROOT = Path(__file__).parents[1]
DEFAULTS = ROOT / "config" / "default_characters"
CATALOG_IDS = {e.id for e in load_catalog(ROOT / "config" / "model_catalog.yaml")}


@pytest.fixture()
def store(tmp_path: Path) -> CharacterStore:
    s = CharacterStore(tmp_path / "characters", DEFAULTS)
    s.load()
    return s


def test_shipped_examples(store: CharacterStore) -> None:
    chars = {c.id: c for c in store.list()}
    assert set(chars) == {"max", "lena", "elara", "captain-vargas", "professor-hale", "vivienne"}
    assert chars["max"].kind == chars["lena"].kind == "assistant"
    assert chars["elara"].kind == chars["captain-vargas"].kind == "roleplay"
    assert chars["max"].voice == "builtin:en_male" and chars["lena"].voice == "builtin:en_female"
    assert [c.kind for c in store.list()][:2] == ["assistant", "assistant"]  # assistants first
    for c in chars.values():
        assert c.greeting and c.system_prompt and len(c.avatar_initials) == 2
        assert c.profile and len(c.recommended_models) == 2 and (DEFAULTS / f"{c.id}.webp").exists()
        assert all(m in CATALOG_IDS for m in c.recommended_models)  # every recommendation is in the catalog
    assert chars["vivienne"].adult and not any(c.adult for c in chars.values() if c.id != "vivienne")
    assert len({c.scene for c in chars.values()}) == 5  # every scene is used


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


def test_scene_is_validated_and_taken_from_the_examples(tmp_path: Path) -> None:
    import json

    from lantern_talk.characters import CharacterError, CharacterStore, validate

    assert validate({"name": "A"}, "a").scene == "plain"
    assert validate({"name": "A", "scene": "space"}, "a").scene == "space"
    with pytest.raises(CharacterError):
        validate({"name": "A", "scene": "disco"}, "a")
    defaults = tmp_path / "defaults"
    defaults.mkdir()
    (defaults / "elara.json").write_text(json.dumps({"name": "Elara", "scene": "tavern"}), encoding="utf-8")
    folder = tmp_path / "chars"
    folder.mkdir()
    (folder / ".defaults-copied").write_text("x", encoding="utf-8")
    (folder / "elara.json").write_text(json.dumps({"name": "Elara"}), encoding="utf-8")  # copied before scenes
    store = CharacterStore(folder, defaults)
    store.load()
    assert store.get("elara").scene == "tavern"


def test_older_installations_get_new_examples_and_portraits(tmp_path: Path) -> None:
    from lantern_talk.images import ImageStore

    folder = tmp_path / "chars"
    folder.mkdir()
    # An installation from before: the first four examples were copied, then Max was deleted.
    (folder / ".defaults-copied").write_text("x", encoding="utf-8")
    for cid in ("lena", "elara", "captain-vargas"):
        old = json.loads((DEFAULTS / f"{cid}.json").read_text(encoding="utf-8"))
        for key in ("profile", "recommended_models"):
            old.pop(key, None)
        (folder / f"{cid}.json").write_text(json.dumps(old), encoding="utf-8")
    edited = json.loads((folder / "elara.json").read_text(encoding="utf-8"))
    edited["avatar"] = "avatar-0123456789ab.webp"  # a picture of your own
    (folder / "elara.json").write_text(json.dumps(edited), encoding="utf-8")
    images = ImageStore(tmp_path / "images")
    store = CharacterStore(folder, DEFAULTS, images)
    store.load()
    ids = {c.id for c in store.list()}
    assert ids == {"lena", "elara", "captain-vargas", "professor-hale", "vivienne"}  # Max stays deleted
    assert store.get("lena").profile and store.get("lena").recommended_models  # new fields taken over
    assert images.path(store.get("lena").avatar) and images.path(store.get("vivienne").avatar)
    assert store.get("elara").avatar == "avatar-0123456789ab.webp"  # your own picture is kept
    # The portrait is given once: removing it is respected on the next start.
    store.save({**store.get("lena").to_dict(), "avatar": ""}, "lena")
    store.load()
    assert store.get("lena").avatar == ""
