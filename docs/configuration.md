# Configuration

Almost everything can be set in the app: the side panels, the character editor and
**Settings** (bottom of the left panel). This page lists where things are stored and
the few options that only exist on the command line.

## The program folder

Everything lives inside the program folder. The sub-folders are fixed:

```
<program folder>\
├── app\           Python backend and web frontend
├── config\        Shipped configuration: model catalog, built-in voices, example characters, pinned versions
├── docs\          This documentation
├── scripts\       Helper scripts used by the .bat files
├── runtime\       Python, llama.cpp, ffmpeg, caches          (downloaded by install.bat)
├── models\        Language, speech and voice models           (downloaded by install.bat / the app)
│   ├── llm\             GGUF language models
│   ├── whisper\         speech recognition
│   ├── omnivoice\       speech synthesis
│   ├── vad\             voice activity detection
│   └── builtin_voices\  the 8 generated built-in voices
├── user_data\     Everything personal
│   ├── characters\      one JSON file per character
│   ├── voices\          your voice clips (+ .txt transcripts)
│   ├── memory\          memory.db (history, summaries, facts)
│   ├── sessions\        recordings: <session>.mp3 + <session>.md
│   ├── custom_models.json
│   └── settings.json
└── logs\          server.log, llama-server.log, console / setup / self-test logs
```

Moving the whole folder to another drive keeps everything working. Nothing is written
outside of it: every cache (Python, PyTorch, Hugging Face, uv, temp files) is redirected
into `runtime\` by `scripts\env.bat`. No admin rights, registry entries or system-wide
installs are needed, and deleting the folder removes the app completely.

**Back up `user_data\`** to keep your characters, voices, memory and recordings. Updates
(`update.bat`) never touch `user_data\` or `models\`.

## Settings screen

| Section | Setting | Default |
| --- | --- | --- |
| Audio devices | Microphone, speaker / headphones (stored in the browser) | system default |
| Microphone | Microphone on | on |
| | Speech detection (0.3 more sensitive … 0.8 ignores more noise) | 0.50 |
| | End of speech after | 700 ms |
| Voice output | Voice output on | on |
| | Synthesis steps (16 / 32 / 48 / 64) | 32 |
| | Precision (fp16 / bf16 / fp32) | fp16 |
| | Regenerate built-in voices | — |
| Language model | Context length (4096 … 32768 tokens) | 8192 |
| | Hardware profile (automatic or high / mid / low / unsupported) | automatic |
| Session recording | Record sessions | on |
| | MP3 quality (32 … 128 kbit/s) | 64 kbit/s |
| | Pause between turns (0 … 2 s) | 0.5 s |

The side panels hold the settings you change most often: voice, *Match built-in voice to
spoken language*, language model, memory and session recording.

## `user_data\settings.json`

The app writes this file whenever a setting changes. You can edit it while the app is
closed; invalid values are ignored or rejected with a message.

| Key | Meaning |
| --- | --- |
| `character` | Id of the active character |
| `voice` | Active voice: `builtin:<id>` or `custom:<file name>` |
| `model` | Active GGUF file in `models\llm\` |
| `mic_enabled`, `voice_output` | Microphone and voice output on / off |
| `match_language` | Built-in voices follow the spoken language |
| `show_actions` | Show actions as text (the active character's setting) |
| `end_silence_ms`, `vad_threshold` | End of speech (200–3000 ms) and speech detection (0.2–0.9) |
| `num_step`, `tts_precision` | OmniVoice synthesis steps (8–96) and precision |
| `context` | LLM context length: 2048, 4096, 8192, 16384, 32768 or 65536 |
| `record_sessions`, `session_gap_ms`, `session_bitrate_kbps` | Session recording |
| `hardware_profile` | Detected by the installer (`high`, `mid`, `low`, `unsupported`) |
| `hardware_profile_override` | Your choice; empty = use the detected profile |
| `gpu_name`, `gpu_vram_mib` | Detected GPU (informational) |

## Command-line options

`start.bat` (the web app):

| Option | Meaning |
| --- | --- |
| `--port 7866` | Use another port (default 7865; the next free one is used if it is taken) |
| `--no-browser` | Do not open the browser |

`install.bat`: `-Yes` (no questions), `-NoStarterModel`, `-NoMicTest`.

`update.bat`: `-Yes`, `-NoSelfTest`.

`console.bat` (the conversation loop without a browser, for troubleshooting):
`--voice builtin:de_male` or `--voice "C:\path\clip.mp3"`, `--text` (typed input only),
`--no-voice` (text replies only), `--model <file>`, `--context`, `--input-device`,
`--output-device`, `--list-devices`, `--verbose`. `console.bat --help` lists all.

## Environment variables

| Variable | Meaning |
| --- | --- |
| `LLT_LLAMA_BUILD` | Force a llama.cpp build in `install.bat`, e.g. `set LLT_LLAMA_BUILD=cuda-12.4` (troubleshooting) |

All other environment variables (cache folders, `UV_*`, `HF_*`, `TORCH_HOME`, …) are set
by `scripts\env.bat` and should not be changed.

## Shipped configuration

| File | Content |
| --- | --- |
| `config\model_catalog.yaml` | The model catalog (see [models.md](models.md)) |
| `config\builtin_voices.yaml` | Descriptions of the 8 built-in voices (see [voices.md](voices.md)) |
| `config\default_characters\` | The example characters copied on first start (see [characters.md](characters.md)) |
| `config\runtime_pins.json` | Pinned versions and checksums of everything the installer downloads |
