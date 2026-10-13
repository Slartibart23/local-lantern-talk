"""Characters (spec §10): one JSON file per character in ``user_data\\characters``."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

KINDS = ("assistant", "roleplay")
MAX_TEXT = 8000


class CharacterError(ValueError):
    """Invalid character data, with a message meant for the user."""


@dataclass
class Character:
    id: str
    name: str
    kind: str = "assistant"
    avatar_initials: str = ""
    persona: str = ""                 # short description shown in the list
    greeting: str = ""
    system_prompt: str = ""           # who the character is; voice rules are added automatically
    voice: str = "builtin:en_female"  # custom:<file> or builtin:<id>
    model: str = ""                   # GGUF file name; "" = the globally selected model
    voice_output: bool = True
    show_actions_as_text: bool = True
    memory_enabled: bool = True
    recording_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def initials_for(name: str) -> str:
    words = [w for w in re.split(r"\s+", name.strip()) if w]
    if not words:
        return "?"
    if len(words) == 1:
        return words[0][:2].upper()
    return (words[0][0] + words[-1][0]).upper()


_GERMAN = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})


def slugify(name: str) -> str:
    """File-system-safe id from a name: "Großmutter Hilde" -> "grossmutter-hilde"."""
    import unicodedata

    text = unicodedata.normalize("NFKD", name.lower().translate(_GERMAN))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-") or "character"


def validate(data: dict[str, Any], char_id: str) -> Character:
    """Build a :class:`Character` from user input, with clear errors."""
    known = {f.name for f in fields(Character)}
    values = {k: v for k, v in data.items() if k in known}
    values["id"] = char_id
    name = str(values.get("name", "")).strip()
    if not name:
        raise CharacterError("A character needs a name.")
    if len(name) > 60:
        raise CharacterError("The name is too long (60 characters at most).")
    values["name"] = name
    kind = values.get("kind", "assistant")
    if kind not in KINDS:
        raise CharacterError("Kind must be assistant or roleplay.")
    for key in ("persona", "greeting", "system_prompt"):
        text = str(values.get(key, "") or "").strip()
        if len(text) > MAX_TEXT:
            raise CharacterError(f"{key.replace('_', ' ').capitalize()} is too long.")
        values[key] = text
    values["avatar_initials"] = (str(values.get("avatar_initials") or "").strip()[:3] or initials_for(name)).upper()
    for key in ("voice_output", "show_actions_as_text", "memory_enabled", "recording_enabled"):
        if key in values:
            values[key] = bool(values[key])
    for key in ("voice", "model"):
        if key in values:
            values[key] = str(values[key] or "").strip()
    return Character(**values)


class CharacterStore:
    """Loads, saves, creates and deletes character files."""

    def __init__(self, folder: Path, defaults: Path) -> None:
        self.folder = folder
        self.defaults = defaults
        self.characters: dict[str, Character] = {}

    def load(self) -> None:
        """Read all characters; copy the shipped examples on the very first start (spec §10)."""
        self.folder.mkdir(parents=True, exist_ok=True)
        marker = self.folder / ".defaults-copied"
        if not marker.exists():
            if self.defaults.is_dir():
                for f in sorted(self.defaults.glob("*.json")):
                    if not (self.folder / f.name).exists():
                        shutil.copy(f, self.folder / f.name)
            marker.write_text("The example characters were copied once; deleted ones stay deleted.\n")
        self.characters = {}
        for f in sorted(self.folder.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                char = validate(data, f.stem)
                self.characters[char.id] = char
            except (OSError, ValueError) as exc:
                log.warning("Skipping character file %s: %s", f.name, exc)

    def list(self) -> list[Character]:
        return sorted(self.characters.values(), key=lambda c: (c.kind != "assistant", c.name.lower()))

    def get(self, char_id: str) -> Character | None:
        return self.characters.get(char_id)

    def first(self) -> Character | None:
        chars = self.list()
        return chars[0] if chars else None

    def save(self, data: dict[str, Any], char_id: str | None = None) -> Character:
        """Create (no ``char_id``) or update a character."""
        if char_id is None:
            char_id = self._free_id(slugify(str(data.get("name", ""))))
        elif char_id not in self.characters:
            raise CharacterError("This character no longer exists.")
        char = validate(data, char_id)
        self._write(char)
        self.characters[char.id] = char
        return char

    def duplicate(self, char_id: str) -> Character:
        original = self.characters.get(char_id)
        if original is None:
            raise CharacterError("This character no longer exists.")
        data = original.to_dict()
        data["name"] = f"{original.name} (copy)"[:60]
        return self.save(data)

    def delete(self, char_id: str) -> None:
        if char_id not in self.characters:
            raise CharacterError("This character no longer exists.")
        if len(self.characters) == 1:
            raise CharacterError("You cannot delete the last character.")
        (self.folder / f"{char_id}.json").unlink(missing_ok=True)
        del self.characters[char_id]

    def _free_id(self, base: str) -> str:
        char_id, n = base, 2
        while char_id in self.characters or (self.folder / f"{char_id}.json").exists():
            char_id = f"{base}-{n}"
            n += 1
        return char_id

    def _write(self, char: Character) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        target = self.folder / f"{char.id}.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(char.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, target)
