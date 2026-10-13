# Local Lantern Talk

Fully local, portable Windows app for spoken (or typed) conversations with a local
large language model that answers in a synthesized — optionally cloned — voice.
Assistant use and character roleplay, with per-character memory and session recording.

> **Status: M5 (voices).** Characters, memory and session recording follow in
> later milestones (see `SPEC.md` §17).

## Requirements

- Windows 10 / 11, 64-bit
- NVIDIA RTX GPU with 10 GB+ VRAM (24 GB+ recommended)
- **NVIDIA driver 570.65 or newer** (the first Windows driver with CUDA 12.8 support);
  driver **581 or newer** is recommended
- ~30 GB free disk space on an NTFS or exFAT drive

You do **not** need to install the CUDA Toolkit: the installer downloads the CUDA
runtime libraries into `runtime\` (PyTorch ships its own, and llama.cpp gets the
matching cudart / cuBLAS DLLs next to `llama-server.exe`).

The installer picks the llama.cpp build from your driver version:

| Driver | llama.cpp build |
|---|---|
| 581 or newer | CUDA 13 (fastest) |
| 570.65 – 580.x | CUDA 12 (fallback) |
| older than 570.65 | not supported — the installer stops and links to the driver download |

On RTX 50-series GPUs the CUDA 12 build compiles its GPU kernels once on first
start (about 12 s extra on an RTX 5090) and caches them in `runtime\cache\`; it is
also somewhat slower than the CUDA 13 build. After a
driver update, re-run `install.bat` to switch to the CUDA 13 build.

## Quick start

1. Run `install.bat`. It checks your GPU, driver and disk, shows what it will download
   (about 9 GB, plus 7–8 GB for the optional starter language model) and asks before
   starting. At the end a self-test checks the GPU, speech output and microphone.
   Details and expected Windows warnings: [docs/installation.md](docs/installation.md).
2. Run `start.bat` (double-click it, or type `.\start.bat` in a PowerShell terminal). It starts the app and opens `http://127.0.0.1:7865` in your browser.
   Keep the console window open while you use the app; closing it quits the app.
3. Click **Start** and allow the microphone. When the status shows **Listening**,
   just talk — or type in the message box at any time.

The app only listens on `127.0.0.1` and accepts requests from its own page only.
One browser tab is active at a time; opening a second tab takes over.

## Using the app

- **Status button**: Listening / Thinking / Speaking. Click it to pause listening;
  in *Mic off* it turns the microphone back on. The mic is muted while the
  assistant speaks, so it never hears itself.
- **Voice**: 8 built-in voices (German, English, Spanish, French × male / female),
  generated locally with OmniVoice voice design on first use and stored in
  `models\builtin_voices\` — no audio is shipped with the repository. With *Match
  built-in voice to spoken language*, a built-in voice switches to the language you
  speak or type, so replies never have a foreign accent.
- **Your own voices**: *Add voice…* uploads an MP3 / WAV / FLAC / OGG / OPUS / M4A clip
  (3–10 s of clean speech works best) into `user_data\voices\`. Whisper transcribes
  it; check and correct the *Reference text* — an exact transcript clones better.
  It is saved as `clip.txt` next to `clip.mp3`. Clip length, what makes a good
  recording and how to provide the transcript without Whisper: [docs/voices.md](docs/voices.md).
- **Speech quality**: synthesis steps (16–64) and precision (fp16 / bf16 / fp32).
  Default: 32 steps, fp16. On an RTX 5090, 64 steps add about 1 s before each reply.
- **Language model**: *Change model…* opens the catalog — German-focused assistant and
  roleplay models (censored and uncensored) with one-click download, automatic choice of
  the quantization that fits your GPU, and your own models from Hugging Face or a link.
  See [docs/models.md](docs/models.md).
- **Microphone off / voice output off** unload Whisper + VAD / OmniVoice from VRAM;
  with both off the app is a pure text chat.
- Actions like `*smiles*` are never spoken; they are shown dimmed with a
  "not spoken" label, or hidden (*Session → Show actions as text*).

### Console mode

`console.bat` runs the same conversation loop without a browser (used for M1 and
for troubleshooting). Options: `console.bat --help`, e.g. `--voice builtin:de_male`,
`--voice "C:\path\clip.mp3"`, `--text`, `--no-voice`, `--list-devices`.

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
