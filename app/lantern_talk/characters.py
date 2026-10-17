"""Characters (spec §10): one JSON file per character in ``user_data\\characters``."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .images import valid_name as valid_image_name

log = logging.getLogger(__name__)

KINDS = ("assistant", "roleplay")
SCENES = ("plain", "tavern", "space", "red-lantern", "study")  # look of the chat (web UI decorations)
MAX_TEXT = 8000
MAX_NOTE = 300     # characters per director's note
MAX_NOTES = 8      # pinned notes per character
MAX_PROFILE = 2000
MAX_STYLE = 1500   # "How they speak" and "In a group"
# The examples shipped before new ones were added; installations that copied them have the old marker file.
FIRST_DEFAULTS = ("lena", "max", "elara", "captain-vargas")
DEFAULTS_STATE = ".defaults.json"
# Chat text colours (web UI): 16 clearly different colours that are all readable on the dark chat.
TEXT_COLORS = ("white", "silver", "yellow", "gold", "orange", "coral", "red", "pink", "magenta", "lavender",
               "violet", "sky", "blue", "turquoise", "mint", "lime")


# Starting points for a new character (+ New character); the parts in [brackets] are for you to fill in.
TEMPLATES: dict[str, dict[str, str]] = {
    "assistant": {
        "system_prompt": (
            "You are [name], [age], a [role, e.g. patient language teacher / friendly cooking coach]. "
            "You help the user with [topics]. You know a lot about [expertise]; when you are not sure, "
            "you say so honestly. Your manner is [three or four traits, e.g. warm, calm, witty]. "
            "[Anything else: what you never do, a habit, a catchphrase.] You run fully locally on the user's "
            "computer and cannot look things up online or know today's news."),
        "speaking_style": "[e.g. short, clear sentences; everyday words; a light joke now and then]",
        "group_style": "[e.g. listens first, then sums up and suggests what to do next]",
        "greeting": "Hello! I'm [name]. What can I do for you?",
    },
    "roleplay": {
        "system_prompt": (
            "You are [name], [age], [role / job]. Background: [where you come from, two or three experiences "
            "that shaped you]. Personality: [three or four traits]. What you know: [your expertise], and what "
            "you don't. The setting: [where and when the conversation takes place]. The user is [who the user "
            "is in this story]. Stay in character at all times, react to what the user says and does, and "
            "drive the story forward with [hooks, secrets, small surprises]."),
        "speaking_style": "[e.g. slow and dry; old-fashioned words; calls the user \"stranger\"; never swears]",
        "group_style": "[e.g. takes the lead / teases whoever spoke last / trusts nobody at first]",
        "greeting": "[What you say when the user arrives, with a little action in *asterisks*.]",
    },
}


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
    scene: str = "plain"              # one of SCENES
    avatar: str = ""                  # picture in user_data\images; "" = the initials
    background: str = ""              # chat background picture in user_data\images; "" = the scene's
    background_fade: int = 80         # how washed out the background is shown, 0-95 %
    text_color: str = ""              # one of TEXT_COLORS for the character's messages; "" = default
    director_notes: list[str] = field(default_factory=list)   # pinned director's notes (always active)
    adult: bool = False               # for adults only: hidden in family mode
    profile: str = ""                 # profile card for you (who the character is); not sent to the model
    recommended_models: list[str] = field(default_factory=list)  # catalog ids: big GPUs first, then small ones
    speaking_style: str = ""          # how the character speaks (added to the instructions)
    group_style: str = ""             # how the character behaves in a group chat or a story

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
    for key in ("voice_output", "show_actions_as_text", "memory_enabled", "recording_enabled", "adult"):
        if key in values:
            values[key] = bool(values[key])
    for key in ("voice", "model"):
        if key in values:
            values[key] = str(values[key] or "").strip()
    scene = str(values.get("scene") or "plain")
    if scene not in SCENES:
        raise CharacterError(f"Unknown scene {scene}.")
    values["scene"] = scene
    for key in ("avatar", "background"):
        name = str(values.get(key) or "")
        values[key] = name if valid_image_name(name) else ""
    try:
        values["background_fade"] = max(0, min(95, int(values.get("background_fade", 80))))
    except (TypeError, ValueError):
        values["background_fade"] = 80
    color = str(values.get("text_color") or "")
    values["text_color"] = color if color in TEXT_COLORS else ""
    notes = values.get("director_notes") or []
    if not isinstance(notes, list):
        notes = []
    values["director_notes"] = [" ".join(str(n).split())[:MAX_NOTE] for n in notes if str(n).strip()][:MAX_NOTES]
    for key, label in (("speaking_style", "How they speak"), ("group_style", "In a group")):
        text = str(values.get(key, "") or "").strip()
        if len(text) > MAX_STYLE:
            raise CharacterError(f"{label} is too long ({MAX_STYLE} characters at most).")
        values[key] = text
    profile = str(values.get("profile", "") or "").strip()
    if len(profile) > MAX_PROFILE:
        raise CharacterError(f"The profile is too long ({MAX_PROFILE} characters at most).")
    values["profile"] = profile
    models = values.get("recommended_models") or []
    values["recommended_models"] = [str(m) for m in models if re.fullmatch(r"[\w.\-]{1,80}", str(m))][:4] \
        if isinstance(models, list) else []
    return Character(**values)


class CharacterStore:
    """Loads, saves, creates and deletes character files."""

    def __init__(self, folder: Path, defaults: Path, images: Any = None) -> None:
        self.folder = folder
        self.defaults = defaults
        self.images = images   # ImageStore for the examples' portraits (None in tests)
        self.characters: dict[str, Character] = {}

    def _defaults_state(self) -> dict[str, list[str]]:
        state_file = self.folder / DEFAULTS_STATE
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # Before this file existed, the first four examples were copied once (marker file).
            state = {"characters": list(FIRST_DEFAULTS) if (self.folder / ".defaults-copied").exists() else []}
        return {"characters": list(state.get("characters") or []), "avatars": list(state.get("avatars") or [])}

    def load(self) -> None:
        """Read all characters. Each shipped example is copied once (spec §10): new examples also reach
        older installations, and examples you changed or deleted are never brought back."""
        self.folder.mkdir(parents=True, exist_ok=True)
        state = self._defaults_state()
        shipped_files = sorted(self.defaults.glob("*.json")) if self.defaults.is_dir() else []
        for f in shipped_files:
            if f.stem not in state["characters"]:
                if not (self.folder / f.name).exists():
                    shutil.copy(f, self.folder / f.name)
                state["characters"].append(f.stem)
        self.characters = {}
        for f in sorted(self.folder.glob("*.json")):
            if f.name.startswith("."):
                continue  # the app's own state (.defaults.json), not a character
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                shipped_path = self.defaults / f.name
                shipped = json.loads(shipped_path.read_text(encoding="utf-8")) if shipped_path.exists() else None
                if shipped is not None:
                    # Copied from an older version: take over what the example has gained since.
                    for key in ("scene", "profile", "recommended_models", "speaking_style", "group_style"):
                        if key not in data and key in shipped:
                            data[key] = shipped[key]
                char = validate(data, f.stem)
                if shipped is not None and char.id not in state["avatars"]:
                    char = self._give_portrait(char)
                    state["avatars"].append(char.id)
                self.characters[char.id] = char
            except (OSError, ValueError) as exc:
                log.warning("Skipping character file %s: %s", f.name, exc)
        state_file = self.folder / DEFAULTS_STATE
        state_file.write_text(json.dumps(state, indent=1) + "\n", encoding="utf-8")

    def _give_portrait(self, char: Character) -> Character:
        """An example without a picture gets its shipped portrait, once (you can replace or remove it)."""
        picture = self.defaults / f"{char.id}.webp"
        if self.images is None or char.avatar or not picture.exists():
            return char
        try:
            char.avatar = self.images.save("avatar", picture.read_bytes())
            self._write(char)
        except (OSError, ValueError) as exc:
            log.warning("Could not add the portrait of %s: %s", char.id, exc)
        return char

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
