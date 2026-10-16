# Architecture

Local Lantern Talk is a Python backend (FastAPI) with a plain HTML/JavaScript frontend.
Everything runs on the local computer; the backend only listens on `127.0.0.1`.

```
 Browser (app\web)                         Backend (app\lantern_talk)
 ┌──────────────────────────┐   WebSocket  ┌────────────────────────────────────────────┐
 │ microphone (AudioWorklet)├─────PCM─────►│ server.py   FastAPI + WebSocket, HTTP API  │
 │ text input, UI           ├────JSON─────►│   │                                        │
 │ playback (Web Audio)     │◄──PCM/JSON───┤ engine.py   worker thread + audio thread   │
 │ replay (<audio>)         │◄──HTTP MP3───┤   ├─ vad.py      Silero VAD → utterances   │
 └──────────────────────────┘              │   ├─ stt.py      faster-whisper            │
                                           │   ├─ pipeline.py LLM → sanitizer → TTS     │
                                           │   │   ├─ llm.py        llama-server (HTTP) │
                                           │   │   ├─ sanitizer.py  actions never spoken│
                                           │   │   ├─ sentences.py  streaming sentences │
                                           │   │   └─ tts.py        OmniVoice           │
                                           │   ├─ characters.py   personas             │
                                           │   ├─ turn.py         end of turn (Smart)  │
                                           │   ├─ memory.py       SQLite history/facts │
                                           │   ├─ images.py       avatars, backgrounds │
                                           │   ├─ recorder.py     session MP3 + .md    │
                                           │   └─ models.py       catalog, downloads   │
                                           └────────────────────────────────────────────┘
```

## One turn

1. **Microphone** — the browser captures audio with an AudioWorklet and streams 16-bit
   PCM at the device's sample rate over the WebSocket. The microphone is only "open"
   while the engine is in the *Listening* state; it is muted while the assistant thinks
   or speaks and while a recording is replayed, so the app never hears itself.
2. **Voice activity detection** — the audio thread resamples to 16 kHz and feeds Silero
   VAD in 32 ms frames. How an utterance ends depends on the talk mode: *Auto* after the
   configured silence (default 700 ms); *Smart* asks Smart Turn v3.2 (an 8 MB ONNX model
   on the CPU, `turn.py`) after 300 ms of silence whether the speaker sounds finished, and
   otherwise waits for a long pause; *Push to talk* and *Hold on* collect all audio between
   two button events (markers in the audio queue keep them in order with the sound). The
   full-quality audio of the same span is kept for session recording.
3. **Speech recognition** — faster-whisper (large-v3-turbo) transcribes the utterance
   and detects its language. Typed messages skip this step; their language is guessed
   from the text.
4. **Prompt** — the character's instructions, your profile, roleplay rules, remembered
   facts and summary, director's notes (pinned ones and the one for this reply) and rules
   for speakable output form the system prompt (`prompts.py`); the unsummarized history
   follows.
5. **Language model** — `llama-server` (llama.cpp) streams the reply. Reasoning is
   disabled in the chat template and filtered anyway.
6. **Sanitizer** — `sanitizer.py` splits the stream into *speech*, *action*, *OOC*,
   *code* and *thinking* segments. Only speech is ever spoken; actions are shown dimmed
   with a "not spoken" label (or hidden), thinking is never shown.
7. **Speech synthesis** — complete sentences go to OmniVoice as soon as they arrive;
   sentences that arrive while one is being synthesized are batched. Audio chunks are
   streamed to the browser and played gaplessly with Web Audio.
8. **After the turn** — the browser reports the end of playback, the engine stores the
   turn in memory, appends it to the session recording (in a background thread), and
   between turns updates facts / summarizes old history with short LLM requests.

## Threads

- **Engine worker** — runs commands one at a time from a queue: turns, settings
  changes, model loading, character switches. No locking games: state changes happen
  here.
- **Audio thread** — VAD on the incoming microphone stream; hands finished utterances to
  the worker.
- **Pipeline threads** — during a turn, one thread reads the LLM stream and one runs TTS,
  so speech starts before the reply is complete.
- **Recorder thread** — encodes and appends session audio with ffmpeg so recording never
  delays the conversation.
- **Download thread** — model downloads with progress, cancel and resume.

## Context length

When a model is loaded for the first time, `llama-fit-params` is run for three context
lengths (4k, 16k, 64k). Its memory breakdown (weights + context cache + compute buffers)
grows linearly with the context, so the need of any other length is interpolated
(`models.ContextProfile`). *Automatic* takes the largest multiple of 1024 that fits into
the free memory (minus a 512 MB reserve), capped by the hardware profile; the settings
slider colours the same data green / yellow / red.

## GPU memory

The language model gets what is left after Whisper, Silero VAD and OmniVoice (which grows
by about 2.4 GB while speaking). Before loading, `llama-fit-params` computes how many
layers (or, for mixture-of-experts models, which expert tensors) fit; Windows' CUDA
reports more free memory than is really available, so the margin is corrected with
`nvidia-smi`. The split is applied exactly with `--fit off`. Turning off the microphone
or voice output unloads Whisper / OmniVoice.

## Processes

`llama-server.exe` is a child process. A Windows job object ties it to the app, so it
ends together with the app even if the console window is closed or the app crashes.

## Storage

| What | Where | Format |
| --- | --- | --- |
| Settings | `user_data\settings.json` | JSON, written atomically |
| Characters | `user_data\characters\<id>.json` | one JSON per character |
| Memory | `user_data\memory\memory.db` | SQLite: messages, summaries, facts, meta |
| Voices | `user_data\voices\` | audio clips + `.txt` reference texts; prepared clips and voice prompts are cached in `.cache\` |
| Sessions | `user_data\sessions\` | `<session>.mp3`, `<session>.md`, offsets in `.index\<session>.jsonl` |
| Pictures | `user_data\images\` | avatars and backgrounds (WebP / PNG / JPEG, shrunk in the browser) |
| Models | `models\` | GGUF, CTranslate2, safetensors, TorchScript |

## Security

- The server binds to `127.0.0.1` only.
- Requests must address the app by its own host name (blocks DNS rebinding); the
  WebSocket and uploads require the app's own origin; recordings are not served to
  other web pages.
- Chat text, characters and facts are never written to the log files.
- Pictures and character exports are only served to the app's own page.

## Tests

`tests\` covers the sanitizer, sentence splitter, paths, catalog parsing, hardware
profiles, MP3 appending and offsets, memory, characters, the engine (with fake speech
components) and the web server. Run them with:

```
runtime\venv\Scripts\python.exe -m pytest
```
