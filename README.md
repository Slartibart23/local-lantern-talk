# Local Lantern Talk

Fully local, portable Windows app for spoken (or typed) conversations with a local
large language model that answers in a synthesized — optionally cloned — voice.
Assistant use and character roleplay, with per-character memory and session recording.

> **Status: M1 (core loop, console only).** The web UI, installer checks, model
> catalog, characters, memory and recording follow in later milestones (see `SPEC.md` §17).

## Requirements

- Windows 10 / 11, 64-bit
- NVIDIA RTX GPU with 10 GB+ VRAM (24 GB+ recommended) and a driver supporting CUDA 12.8
- ~30 GB free disk space on an NTFS or exFAT drive

## M1 quick start

1. Run `install.bat` (downloads ~12 GB: portable Python + PyTorch, llama.cpp,
   Whisper, OmniVoice, Silero VAD, and the Gemma 3 12B test model).
2. Run `console.bat` and talk. Wait for `[Listening]`, speak, pause briefly.
   You can also type a message and press Enter at any time.

Useful options (`console.bat --help` for all):

| Option | Effect |
|---|---|
| `--voice builtin:de_male` | Built-in voice: `de_/en_/es_/fr_` + `male`/`female` |
| `--voice "C:\path\clip.mp3"` | Clone a voice from a 3–10 s reference clip |
| `--text` | Typed input only (mic, Whisper and VAD are not loaded) |
| `--no-voice` | Text replies only (OmniVoice is not loaded) |
| `--num-step 16` | Faster, slightly lower-quality speech synthesis |
| `--end-silence-ms 900` | Wait longer before treating a pause as end of turn |
| `--input-device N` / `--output-device N` | Pick audio devices (`--list-devices`) |

Built-in voices are generated locally with OmniVoice voice design on first use and
stored in `models\builtin_voices\` — no audio is shipped with the repository.

Logs are written to `logs\`. Everything (Python, caches, models) stays inside the
program folder; deleting the folder removes the app completely.

## Disclaimers

- Only clone voices with the explicit consent of the person whose voice it is. Never use
  cloned voices for impersonation, fraud or deception.
- You are responsible for how you use uncensored models.
- Language models are not redistributed by this project; each has its own license
  (e.g. the Gemma terms of use), which you must accept yourself.

## License

MIT — see [LICENSE](LICENSE).
