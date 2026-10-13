"""M1 console loop: mic → VAD → Whisper → llama-server → sanitizer → OmniVoice → playback.

Run via ``console.bat`` (sets up the portable environment). Options::

    console.bat                         talk with the default model and voice
    console.bat --voice builtin:de_male use another built-in voice
    console.bat --voice "C:\\clip.mp3"   clone a voice from a reference clip
    console.bat --text                  typed input only (Whisper / VAD not loaded)
    console.bat --no-voice              text replies only (OmniVoice not loaded)
    console.bat --list-devices          show audio devices

You can also type a message and press Enter at any time while in voice mode.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .llm import LlamaServer, LLMClient, ServerConfig
from .logging_setup import setup_logging
from .paths import Paths, apply_portable_env, get_paths, path_warnings
from .pipeline import Conversation, ConversationConfig, SegmentEvent, SpeechAudio, TurnDone, TurnError
from .prompts import DEFAULT_ASSISTANT_PERSONA, language_name
from .stt import WhisperSTT
from .tts import OUTPUT_SAMPLE_RATE, OmniVoiceTTS, VoiceLibrary
from .vad import FRAME_SAMPLES, SAMPLE_RATE, SegmenterConfig, SileroVAD, UtteranceSegmenter

log = logging.getLogger("lantern_talk.console")

DIM_ITALIC = "\x1b[2;3m"
DIM = "\x1b[2m"
AMBER = "\x1b[33m"
CYAN = "\x1b[36m"
RED = "\x1b[31m"
RESET = "\x1b[0m"

# Audio after playback ends that is still ignored (room echo / speaker tail).
POST_PLAYBACK_MUTE_S = 0.3


def say(text: str = "", end: str = "\n") -> None:
    print(text, end=end, flush=True)


def status(label: str, color: str = AMBER) -> None:
    say(f"{color}[{label}]{RESET}")


# -- audio output -------------------------------------------------------------------

class Player:
    """Plays queued audio chunks back-to-back on one output stream."""

    def __init__(self, device: int | str | None) -> None:
        import sounddevice as sd

        self._stream = sd.OutputStream(samplerate=OUTPUT_SAMPLE_RATE, channels=1, dtype="float32", device=device)
        self._stream.start()
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._idle = threading.Event()
        self._idle.set()
        self.first_play_time: float | None = None
        threading.Thread(target=self._run, daemon=True).start()

    def enqueue(self, audio: np.ndarray) -> None:
        self._idle.clear()
        self._queue.put(audio)

    def _run(self) -> None:
        while True:
            audio = self._queue.get()
            if audio is None:
                return
            if self.first_play_time is None:
                self.first_play_time = time.perf_counter()
            self._stream.write(audio.reshape(-1, 1).astype(np.float32))
            if self._queue.empty():
                # Let the device buffer drain before declaring playback finished.
                time.sleep(self._stream.latency)
                self._idle.set()

    def wait(self) -> None:
        self._idle.wait()

    def close(self) -> None:
        self._queue.put(None)
        self._stream.stop()
        self._stream.close()


# -- audio input ----------------------------------------------------------------------

class Microphone:
    """Captures 16 kHz mono audio; frames are dropped while muted."""

    def __init__(self, device: int | str | None) -> None:
        import sounddevice as sd

        self.frames: queue.Queue[np.ndarray] = queue.Queue()
        self.muted = threading.Event()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=FRAME_SAMPLES, device=device, callback=self._callback,
        )

    def _callback(self, indata: np.ndarray, frames: int, time_info: object, status_flags: object) -> None:
        if not self.muted.is_set():
            self.frames.put(indata[:, 0].copy())

    def start(self) -> None:
        self._stream.start()

    def mute(self) -> None:
        self.muted.set()

    def unmute(self) -> None:
        self.drain()
        self.muted.clear()

    def drain(self) -> None:
        while True:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


def stdin_reader(lines: queue.Queue[str]) -> None:
    for line in sys.stdin:
        lines.put(line.rstrip("\n"))
    lines.put("/quit")


# -- turn display ----------------------------------------------------------------------

def run_and_show_turn(
    conv: Conversation, player: Player | None, user_text: str, language: str | None, t_ref: float | None = None
) -> None:
    status("Thinking")
    if player is not None:
        player.first_play_time = None
    say(f"{CYAN}AI:{RESET} ", end="")
    spoke = False
    metrics = None
    for event in conv.run_turn(user_text, language):
        if isinstance(event, SegmentEvent):
            seg = event.segment
            if seg.spoken:
                say(seg.text, end="")
            else:
                say(f"{DIM_ITALIC}{seg.text}{RESET}{DIM} (not spoken){RESET} ", end="")
        elif isinstance(event, SpeechAudio) and player is not None:
            if not spoke:
                spoke = True
            player.enqueue(event.audio)
        elif isinstance(event, TurnError):
            say(f"\n{RED}Error: {event.message}{RESET}")
        elif isinstance(event, TurnDone):
            metrics = event.metrics
    say()
    if spoke and player is not None:
        status("Speaking")
        player.wait()
    if metrics is not None:
        line = metrics.summary()
        if t_ref is not None and player is not None and player.first_play_time is not None:
            line += f" | end of speech → playback {player.first_play_time - t_ref:.2f}s"
        say(f"{DIM}  ⏱ {line}{RESET}")
        log.info("Turn metrics: %s", line)


# -- setup ---------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="console.bat", description="Local Lantern Talk — M1 console loop")
    p.add_argument("--model", help="GGUF file (default: the M1 test model in models\\llm)")
    p.add_argument("--voice", default="builtin:en_female",
                   help="builtin:<id> from config/builtin_voices.yaml, or a path to an MP3/WAV reference clip")
    p.add_argument("--text", action="store_true", help="typed input only; the microphone is not used")
    p.add_argument("--no-voice", action="store_true", help="text replies only; OmniVoice is not loaded")
    p.add_argument("--persona", default=DEFAULT_ASSISTANT_PERSONA, help="persona / system prompt")
    p.add_argument("--context", type=int, default=8192, help="LLM context length")
    p.add_argument("--gpu-layers", default="all", help="llama.cpp --n-gpu-layers (all / auto / number)")
    p.add_argument("--num-step", type=int, default=32, help="OmniVoice diffusion steps (16 = faster)")
    p.add_argument("--end-silence-ms", type=int, default=700, help="silence that ends an utterance")
    p.add_argument("--vad-threshold", type=float, default=0.5, help="speech probability threshold")
    p.add_argument("--input-device", help="input device index or name (see --list-devices)")
    p.add_argument("--output-device", help="output device index or name")
    p.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    p.add_argument("--verbose", action="store_true", help="show info logs in the console")
    return p.parse_args(argv)


def _device(value: str | None) -> int | str | None:
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def default_model(paths: Paths) -> Path:
    pins = json.loads((paths.config / "runtime_pins.json").read_text(encoding="utf-8"))
    return paths.llm_models / pins["m1_llm"]["file"]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = get_paths()
    apply_portable_env(paths)
    log_file = setup_logging(paths.logs, "console", logging.INFO if args.verbose else logging.WARNING)
    os.system("")  # enables ANSI colors in the classic Windows console

    if args.list_devices:
        import sounddevice as sd

        say(str(sd.query_devices()))
        return 0

    for w in path_warnings(paths.root):
        say(f"{AMBER}Warning: {w}{RESET}")

    use_mic = not args.text
    use_voice = not args.no_voice
    server = LlamaServer(paths.llama_server, paths.logs, env=dict(os.environ))
    mic: Microphone | None = None
    player: Player | None = None
    try:
        say("Local Lantern Talk — M1 console")
        stt = WhisperSTT(paths.whisper_models / "large-v3-turbo")
        if use_mic:  # otherwise loaded lazily if a voice reference needs transcribing
            status("Loading Whisper")
            stt.load()

        tts = voices = None
        if use_voice:
            status("Loading OmniVoice")
            tts = OmniVoiceTTS(paths.omnivoice_model, num_step=args.num_step)
            tts.load()
            voices = VoiceLibrary(
                tts, paths.config / "builtin_voices.yaml", paths.builtin_voices, paths.voice_cache,
                transcribe=lambda audio, sr, lang: stt.transcribe_at(audio, sr, lang).text,
            )
            voice_id = args.voice if args.voice.startswith("builtin:") else str(Path(args.voice).resolve())
            status(f"Preparing voice {args.voice}")
            voices.prompt_for(voice_id)
            args.voice = voice_id

        model = Path(args.model) if args.model else default_model(paths)
        status(f"Loading model {model.name}")
        server.start(ServerConfig(model=model, context=args.context, gpu_layers=args.gpu_layers))

        conv = Conversation(
            LLMClient(server), tts, voices,
            ConversationConfig(persona=args.persona, voice=args.voice if use_voice else None),
        )
        if use_voice:
            player = Player(_device(args.output_device))

        if use_mic:
            vad = SileroVAD(paths.vad_models / "silero_vad.jit")
            vad.load()
            segmenter = UtteranceSegmenter(vad, SegmenterConfig(
                threshold=args.vad_threshold, end_silence_ms=args.end_silence_ms))
            mic = Microphone(_device(args.input_device))
            voice_loop(conv, stt, segmenter, mic, player)
        else:
            text_loop(conv, player)
        return 0
    except KeyboardInterrupt:
        say("\nBye.")
        return 0
    except Exception as exc:  # noqa: BLE001 - short message on screen, details in the log
        log.exception("Fatal error")
        say(f"\n{RED}Error: {exc}{RESET}\nDetails: {log_file}")
        return 1
    finally:
        if mic is not None:
            mic.close()
        if player is not None:
            player.close()
        server.stop()


def text_loop(conv: Conversation, player: Player | None) -> None:
    say(f"{DIM}Type a message and press Enter. /quit to exit.{RESET}")
    while True:
        try:
            text = input(f"{AMBER}You:{RESET} ").strip()
        except EOFError:
            return
        if not text:
            continue
        if text == "/quit":
            return
        run_and_show_turn(conv, player, text, None)


def voice_loop(conv: Conversation, stt: WhisperSTT, segmenter: UtteranceSegmenter,
               mic: Microphone, player: Player | None) -> None:
    typed: queue.Queue[str] = queue.Queue()
    threading.Thread(target=stdin_reader, args=(typed,), daemon=True).start()
    say(f"{DIM}Speak, or type a message and press Enter. /quit or Ctrl+C to exit.{RESET}")
    last_language: str | None = None
    mic.start()
    status("Listening", CYAN)
    while True:
        # Typed input is equal to voice input.
        try:
            text = typed.get_nowait().strip()
        except queue.Empty:
            text = ""
        if text == "/quit":
            return
        if text:
            mic.mute()
            run_and_show_turn(conv, player, text, last_language)
            _resume_listening(mic, segmenter)
            continue

        try:
            frames = mic.frames.get(timeout=0.1)
        except queue.Empty:
            continue
        for utt in segmenter.feed(frames):
            # The segmenter reports the utterance only after the end-of-speech silence
            # has elapsed; measure latency from when the user actually stopped talking.
            t_end = time.perf_counter() - segmenter.config.end_silence_ms / 1000
            mic.mute()  # ignore input until our reply has finished playing
            status("Transcribing")
            tr = stt.transcribe(utt.audio)
            if not tr.text or tr.text.strip(" .") == "":
                _resume_listening(mic, segmenter)
                break
            if tr.language_probability >= 0.6 or last_language is None:
                last_language = tr.language
            lang = language_name(last_language)
            say(f"{AMBER}You ({lang}, {utt.duration_s:.1f}s, STT {tr.seconds * 1000:.0f} ms):{RESET} {tr.text}")
            run_and_show_turn(conv, player, tr.text, last_language, t_ref=t_end)
            _resume_listening(mic, segmenter)
            break


def _resume_listening(mic: Microphone, segmenter: UtteranceSegmenter) -> None:
    time.sleep(POST_PLAYBACK_MUTE_S)
    segmenter.reset()
    mic.unmute()
    status("Listening", CYAN)


if __name__ == "__main__":
    sys.exit(main())
