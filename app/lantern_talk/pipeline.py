"""One conversation turn: LLM stream → sanitizer → sentences → TTS.

The LLM stream and TTS synthesis run in parallel worker threads so the first
sentence can play while later sentences are still being generated. Callers
consume a single stream of :class:`TurnEvent` objects; this keeps the pipeline
usable from the console (M1) and from the web backend (M2).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from .llm import ChatMessage, LLMClient
from .prompts import build_system_prompt
from .sanitizer import ActionFilter, Segment, clean_for_speech
from .sentences import SentenceSplitter
from .tts import OmniVoiceTTS

log = logging.getLogger(__name__)


# -- events ------------------------------------------------------------------------

@dataclass
class TextDelta:
    """Raw token text from the LLM (for live display)."""
    text: str


@dataclass
class SegmentEvent:
    """A finalized segment: speech text or a filtered action / OOC / code part."""
    segment: Segment


@dataclass
class SpeechAudio:
    """Synthesized audio for one sentence, in order."""
    index: int
    text: str
    audio: np.ndarray
    sample_rate: int
    synth_seconds: float


@dataclass
class TurnDone:
    reply: str
    metrics: TurnMetrics


@dataclass
class TurnError:
    message: str


TurnEvent = TextDelta | SegmentEvent | SpeechAudio | TurnDone | TurnError


@dataclass
class TurnMetrics:
    """Latency measurements for one turn (seconds since the turn started)."""
    start: float = field(default_factory=time.perf_counter)
    first_token: float | None = None
    first_sentence: float | None = None
    first_audio: float | None = None
    llm_done: float | None = None
    tts_done: float | None = None
    sentences: int = 0
    chunks: int = 0               # TTS calls (sentences are batched)
    audio_seconds: float = 0.0
    synth_seconds: float = 0.0

    def mark(self, name: str) -> None:
        if getattr(self, name) is None:
            setattr(self, name, time.perf_counter() - self.start)

    def summary(self) -> str:
        def fmt(v: float | None) -> str:
            return "-" if v is None else f"{v * 1000:.0f} ms"

        text = f"first token {fmt(self.first_token)} | LLM done {fmt(self.llm_done)}"
        if not self.sentences:
            return text
        rtf = self.synth_seconds / self.audio_seconds if self.audio_seconds else 0.0
        return (
            f"{text} | first sentence {fmt(self.first_sentence)} | first audio {fmt(self.first_audio)} | "
            f"{self.sentences} sentences in {self.chunks} TTS calls, {self.audio_seconds:.1f}s audio, TTS RTF {rtf:.2f}"
        )


_END = object()

# Upper bound for text batched into one TTS call (OmniVoice splits internally above ~30 s).
MAX_TTS_CHARS = 300


@dataclass
class ConversationConfig:
    persona: str
    voice: str | None             # voice id for VoiceLibrary; None = text-only replies
    max_history_messages: int = 200  # the memory module keeps the history within the context
    match_language: bool = True   # swap built-in voices to the spoken language (spec §7.3)
    roleplay: bool = False        # add the roleplay rules to the system prompt
    memory: str = ""              # remembered facts and summary of earlier conversations


class Conversation:
    """Holds history and runs turns against the LLM and (optionally) TTS."""

    def __init__(self, llm: LLMClient, tts: OmniVoiceTTS | None, voices, config: ConversationConfig) -> None:  # noqa: ANN001
        self.llm = llm
        self.tts = tts
        self.voices = voices
        self.config = config
        self.history: list[ChatMessage] = []
        self.active_voice: str | None = config.voice  # voice used by the last turn

    @property
    def voice_enabled(self) -> bool:
        return self.tts is not None and self.config.voice is not None

    def voice_for(self, language: str | None) -> str | None:
        """The voice to speak a reply in ``language`` with (None when voice output is off)."""
        if not self.voice_enabled:
            return None
        assert self.config.voice is not None
        if self.config.match_language:
            return self.voices.matched(self.config.voice, language)
        return self.config.voice

    def build_messages(self, user_text: str, language: str | None) -> list[ChatMessage]:
        prompt = build_system_prompt(
            self.config.persona, language, roleplay=self.config.roleplay, memory=self.config.memory)
        recent = list(self.history[-self.config.max_history_messages:])
        # Many chat templates (Mistral, Gemma) require user / assistant turns to alternate,
        # starting with the user: an opening greeting goes into the system prompt instead.
        opening: list[str] = []
        while recent and recent[0].role != "user":
            opening.append(recent.pop(0).content)
        if opening:
            prompt += "\n\nYou opened the conversation by saying: " + " ".join(opening)
        messages = [ChatMessage("system", prompt)]
        for m in [*recent, ChatMessage("user", user_text)]:
            if messages[-1].role == m.role and m.role != "system":
                messages[-1] = ChatMessage(m.role, messages[-1].content + "\n\n" + m.content)
            else:
                messages.append(m)
        return messages

    def run_turn(self, user_text: str, language: str | None = None) -> Iterator[TurnEvent]:
        """Run one turn and yield events until :class:`TurnDone` or :class:`TurnError`."""
        metrics = TurnMetrics()
        events: queue.Queue[object] = queue.Queue()
        sentences: queue.Queue[object] = queue.Queue()
        messages = self.build_messages(user_text, language)
        reply_parts: list[str] = []
        errors: list[str] = []
        voice = self.voice_for(language)
        self.active_voice = voice
        tts_language = language or (self.voices.language_of(voice) if voice else None)

        def llm_worker() -> None:
            action_filter = ActionFilter()
            splitter = SentenceSplitter()

            def handle(segments: list[Segment]) -> None:
                for seg in segments:
                    events.put(SegmentEvent(seg))
                    speech = seg.text if seg.spoken else " "
                    for sentence in splitter.feed(speech):
                        push_sentence(sentence)

            def push_sentence(sentence: str) -> None:
                cleaned = clean_for_speech(sentence)
                if cleaned:
                    metrics.mark("first_sentence")
                    sentences.put(cleaned)

            try:
                for delta in self.llm.stream_chat(messages):
                    metrics.mark("first_token")
                    reply_parts.append(delta)
                    events.put(TextDelta(delta))
                    handle(action_filter.feed(delta))
                handle(action_filter.finish())
                for sentence in splitter.finish():
                    push_sentence(sentence)
            except Exception as exc:  # noqa: BLE001 - reported to the user as a turn error
                log.exception("LLM stream failed")
                errors.append(f"The language model failed: {exc}")
            finally:
                metrics.mark("llm_done")
                sentences.put(_END)

        def tts_worker() -> None:
            index = 0
            try:
                prompt = self.voices.prompt_for(voice) if voice else None
                finished = False
                while not finished:
                    item = sentences.get()
                    if item is _END:
                        break
                    # Each OmniVoice call costs about the same regardless of text length,
                    # so everything the LLM wrote while the previous chunk was being
                    # synthesized is spoken in one call. The first sentence still goes
                    # out alone, which keeps the time to first audio low.
                    batch = [str(item)]
                    while sum(len(t) + 1 for t in batch) < MAX_TTS_CHARS:
                        try:
                            nxt = sentences.get_nowait()
                        except queue.Empty:
                            break
                        if nxt is _END:
                            finished = True
                            break
                        batch.append(str(nxt))
                    if prompt is None or self.tts is None:
                        continue
                    text = " ".join(batch)
                    result = self.tts.synthesize(text, prompt, tts_language)
                    metrics.mark("first_audio")
                    metrics.sentences += len(batch)
                    metrics.chunks += 1
                    metrics.audio_seconds += result.duration_s
                    metrics.synth_seconds += result.seconds
                    events.put(SpeechAudio(index, text, result.audio, 24_000, result.seconds))
                    index += 1
            except Exception as exc:  # noqa: BLE001
                log.exception("TTS failed")
                errors.append(f"Speech synthesis failed: {exc}")
                while sentences.get() is not _END:  # drain so the LLM worker never blocks
                    pass
            finally:
                metrics.mark("tts_done")
                events.put(_END)

        threads = [threading.Thread(target=llm_worker, daemon=True), threading.Thread(target=tts_worker, daemon=True)]
        for t in threads:
            t.start()
        while True:
            item = events.get()
            if item is _END:
                break
            yield item  # type: ignore[misc]
        for t in threads:
            t.join()

        reply = "".join(reply_parts).strip()
        if errors:
            yield TurnError("; ".join(errors))
        if reply:
            self.history.append(ChatMessage("user", user_text))
            self.history.append(ChatMessage("assistant", reply))
        yield TurnDone(reply, metrics)
