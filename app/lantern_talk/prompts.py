"""System prompt construction (spec §8 layer 1 and §9 step 3)."""

from __future__ import annotations

LANGUAGE_NAMES: dict[str, str] = {
    "de": "German", "en": "English", "es": "Spanish", "fr": "French", "it": "Italian",
    "nl": "Dutch", "pt": "Portuguese", "pl": "Polish", "ru": "Russian", "tr": "Turkish",
    "ja": "Japanese", "zh": "Chinese", "ko": "Korean", "sv": "Swedish", "da": "Danish",
    "no": "Norwegian", "fi": "Finnish", "cs": "Czech", "uk": "Ukrainian", "ar": "Arabic",
}

VOICE_RULES = """\
Your replies are converted to speech and played aloud, so write for the ear:
- Write only words that are meant to be spoken. Never narrate actions, gestures or \
stage directions (no *asterisks*, no _underscores_, no [brackets]). If an action \
matters, weave it into what you say ("Let me pour you a drink.").
- No markdown: no headings, lists, bullet points, tables, bold, code blocks or links.
- No emojis, no special symbols. Write numbers the way they are naturally spoken.
- Keep replies conversational and fairly short: usually one to four sentences, \
longer only when the user explicitly asks for detail."""


def language_name(code: str | None) -> str | None:
    if not code:
        return None
    return LANGUAGE_NAMES.get(code, code)


ROLEPLAY_RULES = """\
This is an interactive roleplay with the user.
- Stay in character at all times. Do not mention being an AI or a language model unless the \
user explicitly asks out of character (OOC).
- Speak only as yourself: never write the user's lines, thoughts or decisions.
- Keep each reply to a few spoken sentences and leave the user something to react to."""


DIRECTOR_RULES = """\
Director's notes from the user follow. They are stage directions for you, not words the user says in \
the conversation: follow them in what you say and do, but never mention, quote or answer them, and \
stay in character."""
CONTINUE_TEXT = "(continue)"   # the user's turn when they only gave a director's note


def director_context(pinned: list[str], note: str = "") -> str:
    """Pinned notes (always) and a one-time note for the next reply (Director field in the chat)."""
    lines = [f"- {n}" for n in pinned if n.strip()]
    parts = []
    if lines:
        parts.append("Always:\n" + "\n".join(lines))
    if note.strip():
        parts.append(f"For your next reply: {note.strip()}")
    if not parts:
        return ""
    return DIRECTOR_RULES + "\n" + "\n".join(parts)


def user_context(name: str, about: str) -> str:
    """What the user wrote about themselves in their profile (Settings → Your profile)."""
    lines = []
    if name.strip():
        lines.append(f"The user's name is {name.strip()}.")
    if about.strip():
        lines.append(f"About the user, in their own words: {about.strip()}")
    return "\n".join(lines)


def build_system_prompt(persona: str, reply_language: str | None = None, *,
                        roleplay: bool = False, memory: str = "", user: str = "", director: str = "") -> str:
    """Combine the character persona, roleplay rules, memory and the voice-friendly output rules."""
    parts = [persona.strip()]
    if user:
        parts.append(user.strip())
    if roleplay:
        parts.append(ROLEPLAY_RULES)
    if memory:
        parts.append(memory.strip())
    if director:
        parts.append(director.strip())
    parts.append(VOICE_RULES)
    name = language_name(reply_language)
    if name:
        parts.append(f"The user is speaking {name}. Reply in {name}.")
    else:
        parts.append("Always reply in the same language the user writes or speaks in.")
    return "\n\n".join(p for p in parts if p)


DEFAULT_ASSISTANT_PERSONA = (
    "You are Lantern, a friendly and knowledgeable voice assistant running fully "
    "locally on the user's computer. You are helpful, honest and concise. "
    "You can talk, answer questions, explain, brainstorm and help with writing. "
    "You cannot access the internet, play music, set reminders or timers, control "
    "devices or apps, or know the current date and news; say so honestly when asked."
)
