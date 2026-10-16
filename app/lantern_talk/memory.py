"""Per-character memory (spec §11): history, rolling summary and a list of facts.

Stored in SQLite (``user_data\\memory\\memory.db``). Simple and transparent: no
vector database. The LLM prompts used to maintain the memory are defined here
too; the engine runs them between turns.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

MAX_FACTS = 30
MAX_FACT_CHARS = 200
KEEP_RECENT_MESSAGES = 6          # never summarized away: the immediate context
HISTORY_SHARE_OF_CONTEXT = 0.5    # history may use half the context; the rest is prompt and reply
FACT_UPDATE_EVERY = 4             # user turns between fact updates
CHARS_PER_TOKEN = 3.5             # rough average for German / English text
SUMMARY_WORDS_PER_TOKEN = 0.025   # summary length per token of context: 8k -> 200, 16k -> 400, 32k -> 800 words
SUMMARY_MIN_WORDS = 150
SUMMARY_MAX_WORDS = 1000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    character TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    language TEXT,
    source TEXT,
    voice TEXT,
    created REAL NOT NULL,
    summarized INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS messages_character ON messages (character, summarized, id);
CREATE TABLE IF NOT EXISTS summaries (character TEXT PRIMARY KEY, text TEXT NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    character TEXT NOT NULL,
    text TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'llm',
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (character TEXT NOT NULL, key TEXT NOT NULL, value TEXT, PRIMARY KEY (character, key));
"""


@dataclass(frozen=True)
class Message:
    id: int
    role: str
    content: str
    language: str | None
    source: str | None
    voice: str | None
    created: float
    replay_session: str | None = None   # session recording that contains this message (spec §12.3)
    replay_offset: float | None = None
    replay_duration: float | None = None


@dataclass(frozen=True)
class Fact:
    id: int
    text: str
    source: str   # "llm" or "user"


class MemoryStore:
    """Thread-safe access to the memory database."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.executescript(_SCHEMA)
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(messages)")}
        for name, kind in (("replay_session", "TEXT"), ("replay_offset", "REAL"), ("replay_duration", "REAL")):
            if name not in columns:  # databases from before session recording
                self._db.execute(f"ALTER TABLE messages ADD COLUMN {name} {kind}")
        self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._db.execute(sql, params)
            self._db.commit()
            return cur

    def _query(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    # -- history ----------------------------------------------------------------------
    def add_message(self, character: str, role: str, content: str, language: str | None = None,
                    source: str | None = None, voice: str | None = None) -> int:
        cur = self._execute(
            "INSERT INTO messages (character, role, content, language, source, voice, created) VALUES (?,?,?,?,?,?,?)",
            (character, role, content, language, source, voice, time.time()))
        return int(cur.lastrowid)

    def active_messages(self, character: str) -> list[Message]:
        """Messages not yet folded into the summary, oldest first."""
        rows = self._query(
            "SELECT id, role, content, language, source, voice, created, replay_session, replay_offset, "
            "replay_duration FROM messages WHERE character = ? AND summarized = 0 ORDER BY id", (character,))
        return [Message(*r) for r in rows]

    def set_replay(self, message_id: int, session: str, offset: float, duration: float | None) -> None:
        self._execute("UPDATE messages SET replay_session = ?, replay_offset = ?, replay_duration = ? WHERE id = ?",
                      (session, offset, duration, message_id))

    def mark_summarized(self, character: str, up_to_id: int) -> None:
        self._execute("UPDATE messages SET summarized = 1 WHERE character = ? AND id <= ?", (character, up_to_id))

    def message_count(self, character: str) -> int:
        return int(self._query("SELECT COUNT(*) FROM messages WHERE character = ?", (character,))[0][0])

    # -- summary ----------------------------------------------------------------------
    def summary(self, character: str) -> str:
        rows = self._query("SELECT text FROM summaries WHERE character = ?", (character,))
        return rows[0][0] if rows else ""

    def set_summary(self, character: str, text: str) -> None:
        self._execute("INSERT INTO summaries (character, text, updated) VALUES (?,?,?) "
                      "ON CONFLICT(character) DO UPDATE SET text = excluded.text, updated = excluded.updated",
                      (character, text.strip(), time.time()))

    # -- facts --------------------------------------------------------------------------
    def facts(self, character: str) -> list[Fact]:
        rows = self._query("SELECT id, text, source FROM facts WHERE character = ? ORDER BY id", (character,))
        return [Fact(*r) for r in rows]

    def add_fact(self, character: str, text: str, source: str = "user") -> int | None:
        text = " ".join(text.split())[:MAX_FACT_CHARS]
        if not text or any(f.text.lower() == text.lower() for f in self.facts(character)):
            return None
        if len(self.facts(character)) >= MAX_FACTS:
            return None
        return int(self._execute("INSERT INTO facts (character, text, source, created) VALUES (?,?,?,?)",
                                 (character, text, source, time.time())).lastrowid)

    def update_fact(self, character: str, fact_id: int, text: str) -> None:
        text = " ".join(text.split())[:MAX_FACT_CHARS]
        if text:
            self._execute("UPDATE facts SET text = ?, source = 'user' WHERE character = ? AND id = ?",
                          (text, character, fact_id))
        else:
            self.delete_fact(character, fact_id)

    def delete_fact(self, character: str, fact_id: int) -> None:
        self._execute("DELETE FROM facts WHERE character = ? AND id = ?", (character, fact_id))

    def remove_facts_by_text(self, character: str, texts: list[str]) -> None:
        wanted = {t.strip().lower() for t in texts}
        for f in self.facts(character):
            if f.text.lower() in wanted and f.source != "user":  # never drop what the user wrote
                self.delete_fact(character, f.id)

    # -- meta / reset -------------------------------------------------------------------------
    def get_meta(self, character: str, key: str, default: str = "") -> str:
        rows = self._query("SELECT value FROM meta WHERE character = ? AND key = ?", (character, key))
        return rows[0][0] if rows else default

    def set_meta(self, character: str, key: str, value: str) -> None:
        self._execute("INSERT INTO meta (character, key, value) VALUES (?,?,?) "
                      "ON CONFLICT(character, key) DO UPDATE SET value = excluded.value", (character, key, value))

    def reset(self, character: str) -> None:
        """Forget everything about this character (spec §11 'reset a character's memory')."""
        for table in ("messages", "summaries", "facts", "meta"):
            self._execute(f"DELETE FROM {table} WHERE character = ?", (character,))  # noqa: S608 - fixed names


# ---------------------------------------------------------------------------- policy

def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 4


def history_budget(context: int) -> int:
    return int(context * HISTORY_SHARE_OF_CONTEXT)


def summary_words(context: int) -> int:
    """Length of the summary: a larger context keeps more of a long story (200 words at 8k)."""
    return min(SUMMARY_MAX_WORDS, max(SUMMARY_MIN_WORDS, round(context * SUMMARY_WORDS_PER_TOKEN / 50) * 50))


def messages_to_summarize(messages: list[Message], context: int) -> list[Message]:
    """Oldest messages to fold into the summary, or ``[]`` while the history fits."""
    total = sum(estimate_tokens(m.content) for m in messages)
    if total <= history_budget(context) or len(messages) <= KEEP_RECENT_MESSAGES:
        return []
    # Summarize until the rest uses at most half the budget, but keep the recent messages.
    target = history_budget(context) // 2
    out: list[Message] = []
    for m in messages[:-KEEP_RECENT_MESSAGES]:
        if total <= target:
            break
        out.append(m)
        total -= estimate_tokens(m.content)
    # never split a user message from the reply that follows it
    if out and out[-1].role == "user":
        out.pop()
    return out


def transcript_text(messages: list[Message], character_name: str, user_name: str = "User") -> str:
    who = {"user": user_name or "User", "assistant": character_name}
    return "\n".join(f"{who.get(m.role, m.role)}: {m.content}" for m in messages)


SUMMARY_SYSTEM = ("You keep the long-term memory of an ongoing conversation between a user and "
                  "{name}. Write compact, factual notes.")
SUMMARY_PROMPT = """Earlier summary (may be empty):
{summary}

Conversation to add:
{transcript}

Write an updated summary of the whole conversation so far in at most {words} words, in the language
the conversation is held in. Keep names, decisions, promises, open questions and story events.
Plain text only, no lists or markdown."""

FACTS_SYSTEM = ("You maintain a short list of durable facts that {name} should remember about the user "
                "and their shared story.")
FACTS_PROMPT = """Known facts:
{facts}

Recent conversation:
{transcript}

Which new durable facts did the recent conversation reveal (names, preferences, relationships,
important events, things the user asked to remember)? Which known facts are now wrong?
Answer with JSON only: {{"add": ["short fact", ...], "remove": ["exact text of a wrong known fact", ...]}}
At most 5 additions, each one short sentence in the language of the conversation. Use empty lists
if nothing changed. Do not repeat known facts."""


def parse_fact_update(text: str) -> tuple[list[str], list[str]]:
    """Parse the facts reply; anything unusable yields no change."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return [], []
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return [], []
    if not isinstance(data, dict):
        return [], []

    def strings(key: str) -> list[str]:
        value = data.get(key, [])
        if not isinstance(value, list):
            return []
        return [x.strip() for x in value if isinstance(x, str) and x.strip()]

    return strings("add")[:5], strings("remove")


def memory_context(summary: str, facts: list[Fact]) -> str:
    """What the system prompt tells the model about earlier conversations."""
    parts = []
    if facts:
        parts.append("Things you know and remember:\n" + "\n".join(f"- {f.text}" for f in facts))
    if summary:
        parts.append("Summary of your earlier conversations with the user:\n" + summary)
    return "\n\n".join(parts)
