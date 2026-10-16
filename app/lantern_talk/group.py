"""Group chats and story mode: two to four characters in one conversation.

All characters share the one loaded language model; each speaks with its own voice and
gets its own system prompt (its persona, how it speaks, how it behaves in a group and who
else is there). The shared history is shown to each speaker from its own point of view:
its own lines are the assistant's turns, everything the others said arrives as user turns
with the speaker's name in front.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from .llm import ChatMessage
from .prompts import VOICE_RULES, language_name

MAX_CAST = 4          # characters in one chat, including the one picked on the left
MAX_RULES = 3000
GROUP_PREFIX = "group:"
INTRO_CHARS = 240     # how much of a profile introduces a character to the others

# Story mode: what every character is told about telling a story together (Settings → Story mode).
DEFAULT_STORY_RULES = """\
This is a story that you and the other characters tell together, line by line, while the user \
listens and directs.
- Play the scene of the story setup. React to the line before yours, then move the scene forward. \
Each line adds something new: something you do, feel, want or decide, a discovery, a joke, a \
change of mood.
- Speak plainly and directly, like a real person in this situation: say what you feel, want and \
do. No flowery metaphors, no riddles, no vague hints, no small talk that leads nowhere.
- Never repeat or paraphrase what was already said, and never summarize. Vary how you begin your \
sentences.
- Stay true to your character: tease, hesitate, disagree or surprise when it fits you and the scene.
- Ask a question only when it matters, not in every line.
- Keep each line to one to four spoken sentences.
- When you speak to someone, use their name."""

GROUP_CHAT_RULES = """\
This is a group conversation: the user and several characters talk together.
- React to what was just said, by the user or by the others. You may address the others by name, \
agree, disagree or add something new, but never repeat what someone already said.
- Keep it short, one to three spoken sentences, so that everyone gets a turn."""

# Story mode: now and then a speaker gets a nudge, so the story never runs in circles.
IMPULSES = (
    "Bring something unexpected into the scene.",
    "Reveal something about yourself that the others do not know yet.",
    "Say plainly what you feel or want right now.",
    "Make a decision and act on it.",
    "Let a quiet, personal moment happen.",
    "Do something in the scene instead of only talking.",
)
SETUP_REMINDER_CHARS = 400   # the story setup is repeated at the end of the prompt, where models follow best
IMPULSE_EVERY = 5     # lines without a director's note before a nudge

OPENING_TEXT = "(The scene begins. You speak first.)"
CONTINUE_LINE = "(continue)"


def group_key(ids: Sequence[str]) -> str:
    """Memory key of a cast: the same characters always share one memory, in any order."""
    return GROUP_PREFIX + "+".join(sorted(ids))


def session_stem(ids: Sequence[str]) -> str:
    """Characters of a cast for a recording's file name ("lena-max")."""
    return "-".join(sorted(ids))[:80]


def cast_title(names: Sequence[str]) -> str:
    """"Lena", "Lena and Max", "Lena, Max and Elara"."""
    names = list(names)
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def call_names(name: str) -> list[str]:
    """The ways a character is addressed: the full name and each longer word of it."""
    words = [w for w in re.findall(r"[^\W\d_][\w'\-]*", name) if len(w) >= 3]
    return [name.strip(), *words]


def mentioned(text: str, members: Sequence[Any], exclude: Sequence[str] = ()) -> list[str]:
    """Ids of the members addressed or named in ``text``, in the order they first appear.

    Names are matched as capitalized words ("Max", "Professor"), so "the max" does not count.
    """
    found: list[tuple[int, str]] = []
    for m in members:
        if m.id in exclude:
            continue
        positions = []
        for word in call_names(m.name):
            hit = re.search(rf"(?<![\w]){re.escape(word)}(?![\w])", text)
            if hit and word[:1].isupper():
                positions.append(hit.start())
        if positions:
            found.append((min(positions), m.id))
    return [cid for _, cid in sorted(found)]


def next_speaker(members: Sequence[Any], history: Sequence[ChatMessage]) -> str:
    """Who speaks next when nobody was asked: someone the last line addressed, else whoever
    has been quiet the longest (never the same character twice in a row)."""
    ids = [m.id for m in members]
    last = history[-1] if history else None
    if last is not None:
        exclude = [last.name] if last.role == "assistant" and last.name else []
        named = mentioned(last.content, members, exclude)
        if named:
            return named[0]
    last_spoke = {cid: -1 for cid in ids}
    for i, m in enumerate(history):
        if m.role == "assistant" and m.name in last_spoke:
            last_spoke[m.name] = i
    candidates = [cid for cid in ids if not (last is not None and last.role == "assistant" and last.name == cid)]
    candidates = candidates or ids
    return min(candidates, key=lambda cid: (last_spoke[cid], ids.index(cid)))


def intro(member: Any) -> str:
    """One or two sentences about a character for the others (from its profile)."""
    text = " ".join((member.profile or member.persona or "").split())
    if len(text) > INTRO_CHARS:
        cut = text[:INTRO_CHARS]
        end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        text = cut[: end + 1] if end > 60 else cut.rsplit(" ", 1)[0] + " …"
    return text


def persona_text(member: Any, default: str) -> str:
    """Who the character is and how it speaks (used alone and in groups)."""
    parts = [(member.system_prompt or member.persona or default).strip()]
    if member.speaking_style.strip():
        parts.append("How you speak: " + member.speaking_style.strip())
    return "\n\n".join(parts)


def build_group_prompt(speaker: Any, others: Sequence[Any], *, default_persona: str, story: bool,
                       story_rules: str = "", premise: str = "", user: str = "", user_name: str = "",
                       memory: str = "", director: str = "", reply_language: str | None = None) -> str:
    """The system prompt for ``speaker``'s next line in a group chat or a story."""
    parts = [persona_text(speaker, default_persona)]
    setup = premise.strip() if story else ""
    if setup:  # the scene comes first: it replaces the character's usual setting
        parts.append("THE SCENE OF THIS STORY. It takes priority over the setting, the situation and the role of the "
                     "user in your description above; keep your personality, your past and your way of speaking:\n"
                     + setup)
    if speaker.group_style.strip():
        parts.append("In a group: " + speaker.group_style.strip())
    with_you = [f"- {o.name}" + (f": {intro(o)}" if intro(o) else "") for o in others]
    parts.append("With you in the scene:\n" + "\n".join(with_you))
    if user:
        parts.append(user.strip())
    parts.append((story_rules.strip() or DEFAULT_STORY_RULES) if story else GROUP_CHAT_RULES)
    if memory:
        parts.append(memory.strip())
    if director:
        parts.append(director.strip())
    who = user_name.strip() or "the user"
    if setup:
        short = setup if len(setup) <= SETUP_REMINDER_CHARS else setup[:SETUP_REMINDER_CHARS].rsplit(" ", 1)[0] + " …"
        parts.append(f"Stay in the scene: {short}")
    parts.append(f"Each message from someone else starts with their name. Speak only as {speaker.name}: "
                 f"never write lines for the others or for {who}, and never put your own name in front of "
                 "your reply. Stay in character.")
    parts.append(VOICE_RULES)
    name = language_name(reply_language)
    parts.append(f"Everyone is speaking {name}. Reply in {name}." if name
                 else "Reply in the language the conversation is held in.")
    return "\n\n".join(p for p in parts if p)


def group_messages(speaker_id: str, system: str, history: Sequence[ChatMessage], names: dict[str, str],
                   user_name: str = "") -> list[ChatMessage]:
    """The shared history from ``speaker_id``'s point of view."""
    user_label = user_name.strip() or "User"
    turns: list[ChatMessage] = []
    for m in history:
        if m.role == "assistant" and m.name == speaker_id:
            turn = ChatMessage("assistant", m.content)
        elif m.role == "assistant":
            turn = ChatMessage("user", f"{names.get(m.name or '', m.name or 'Someone')}: {m.content}")
        else:
            turn = ChatMessage("user", f"{user_label}: {m.content}")
        if turns and turns[-1].role == turn.role:
            turns[-1] = ChatMessage(turn.role, turns[-1].content + "\n\n" + turn.content)
        else:
            turns.append(turn)
    # Chat templates want the first turn from the user: the speaker's opening goes into the prompt.
    opening = []
    while turns and turns[0].role == "assistant":
        opening.append(turns.pop(0).content)
    if opening:
        system += "\n\nYou opened the conversation by saying: " + " ".join(opening)
    if not turns:
        turns.append(ChatMessage("user", OPENING_TEXT))
    elif turns[-1].role == "assistant":
        turns.append(ChatMessage("user", CONTINUE_LINE))  # the same character goes on
    return [ChatMessage("system", system), *turns]


def stop_sequences(speaker_id: str, names: dict[str, str], user_name: str = "") -> list[str]:
    """Where a reply ends if the model starts writing someone else's line."""
    labels = [n for cid, n in names.items() if cid != speaker_id] + [user_name.strip() or "User"]
    return [f"\n{label}:" for label in labels if label][:8]


class NamePrefixFilter:
    """Removes the speaker's own name from the start of a streamed reply ("Max: Well…")."""

    def __init__(self, name: str) -> None:
        words = [re.escape(w) for w in call_names(name)]
        self._pattern = re.compile(r"^\s*(?:\*\*)?(?:" + "|".join(words) + r")(?:\*\*)?\s*:\s*(?:\*\*)?\s*",
                                   re.IGNORECASE)
        self._limit = len(name) + 8
        self._buffer = ""
        self._done = False

    def feed(self, delta: str) -> str:
        if self._done:
            return delta
        self._buffer += delta
        if len(self._buffer) < self._limit and "\n" not in self._buffer.lstrip():
            return ""
        return self._release()

    def finish(self) -> str:
        return "" if self._done else self._release()

    def _release(self) -> str:
        self._done = True
        text, self._buffer = self._pattern.sub("", self._buffer, count=1), ""
        return text


def director_lines(pinned: Sequence[dict[str, str]], speaker_id: str) -> list[str]:
    """Pinned group notes that apply to ``speaker_id`` (for everyone or for them)."""
    return [p["text"] for p in pinned if p.get("target", "") in ("", speaker_id) and p.get("text")]
