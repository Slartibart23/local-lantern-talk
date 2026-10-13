# Local Lantern Talk — Project Specification

> This document is the single source of truth for building **Local Lantern Talk**.
> It was written before implementation started. Where it says *verify*, check the
> current state of the dependency before relying on the assumption.

---

## 1. Project overview

**Local Lantern Talk** is a fully local, portable Windows application that lets a
user hold spoken (or typed) conversations with a local large language model. The
model answers with a synthesized voice that can be cloned from any MP3 reference
clip. It supports assistant-style use and character roleplay, with per-character
memory and full session recording (text transcript + one MP3 of the whole dialogue).

| Item | Value |
|---|---|
| Name | Local Lantern Talk |
| Repository | `github.com/Slartibart23/local-lantern-talk` |
| Visibility | **Private** at first (owner decides when to make it public) |
| License | **MIT**, copyright holder `Slartibart23`, year 2026 |
| Language of code, comments, docs, UI, commits | **English only** |
| Target platform | Windows 10 / 11, 64-bit |
| GPU support | **NVIDIA only** (CUDA). No AMD / Intel GPU support. |
| Internet | Needed only for installation and model downloads. Runtime is offline. |

### 1.1 Primary use cases
1. **Assistant:** general questions and help by voice or text.
2. **Roleplay:** conversations with persistent characters that have their own voice,
   persona and memory.

Language practice is explicitly **not** a target use case.

### 1.2 Non-goals (do not build)
- Interrupting the AI while it speaks (barge-in). The mic is muted during playback.
- Wake words.
- AMD / Intel GPUs, macOS, Linux.
- Cloud LLM or cloud TTS providers.
- Multi-user / network access (the app binds to `127.0.0.1` only).
- A "narrator voice" for roleplay actions (see §8 — actions are never spoken).

---

## 2. Hardware requirements (to appear in README and docs)

| Tier | GPU | System RAM | Free disk (SSD recommended) |
|---|---|---|---|
| **Minimum** | NVIDIA RTX with **10 GB VRAM** (e.g. RTX 3080) | 16 GB | ~30 GB (install + one small model) |
| **Recommended** | NVIDIA RTX with **24 GB+ VRAM** (RTX 4090 / 5090) | 32 GB | ~100 GB (several models) |

Additional requirements:
- An NVIDIA driver that supports **CUDA 12.8** (required for RTX 50-series / Blackwell).
- Windows 10 or 11, 64-bit.
- Target drive formatted **NTFS or exFAT** (FAT32 cannot store files > 4 GB).

Notes for the docs:
- Text-only mode (mic and voice output disabled) needs much less VRAM.
- Values are initial estimates; **measure real VRAM usage during testing** on the
  RTX 5090 and update the docs. RTX 3080 / 4090 values remain estimates until
  someone tests on that hardware — say so in the docs.

---

## 3. Portability and folder structure

Everything lives inside **one program folder**. The user chooses only where the
program folder goes (wherever they clone / extract the repo). Sub-folder locations
are **fixed and not configurable**.

```
<ProgramFolder>\
├── app\                 Application source code (Python backend + web frontend)
├── config\              Shipped config: model catalog, defaults, built-in voice definitions
├── docs\                Documentation (English)
├── scripts\             Helper scripts used by the .bat files
├── runtime\             Downloaded by installer: portable Python (uv), llama.cpp, ffmpeg   [gitignored]
├── models\              Downloaded by installer / on demand                                 [gitignored]
│   ├── llm\             GGUF language models
│   ├── whisper\         faster-whisper models
│   ├── omnivoice\       OmniVoice TTS model
│   ├── vad\             Silero VAD
│   └── builtin_voices\  The 8 generated built-in voices
├── user_data\           Everything personal                                                 [gitignored]
│   ├── characters\
│   ├── voices\          User's MP3/WAV reference clips
│   ├── memory\
│   ├── sessions\        Transcripts + session MP3s
│   └── settings.json    User settings incl. hardware profile overrides
├── logs\                Application logs                                                    [gitignored]
├── install.bat
├── start.bat
├── update.bat
├── README.md
├── CHANGELOG.md
├── LICENSE
└── THIRD_PARTY_NOTICES.md
```

Rules:
- **All paths are resolved relative to the program folder.** Moving the whole folder
  to another drive must keep everything working.
- **No admin rights, no registry entries, no system-wide installs.** Deleting the
  folder removes the app completely.
- **Redirect every library cache into the program folder** so nothing lands in
  `C:\Users\...\.cache` or `%APPDATA%`. At minimum set, for every process the app starts:
  `HF_HOME`, `HF_HUB_CACHE`, `TORCH_HOME`, `XDG_CACHE_HOME`, `UV_CACHE_DIR`,
  `UV_PYTHON_INSTALL_DIR`, `PIP_CACHE_DIR`, `TMP`/`TEMP` (to a folder inside the
  program folder). Audit for any other library that writes caches.
- Paths **with spaces must work** (quote everything in .bat scripts). Warn (do not
  block) if the path contains non-ASCII characters or is very long.
- `user_data\` and `models\` are never touched by updates.

---

## 4. Installation (`install.bat`)

Goal: **one double-click**. The script must be idempotent (safe to re-run) and
resumable (interrupted downloads continue).

Steps:
1. **Pre-checks**
   - Detect GPU via `nvidia-smi` (name, VRAM, driver version). If no NVIDIA GPU is
     found, show a clear message and stop.
   - Check the driver supports CUDA 12.8; if not, explain and link to NVIDIA driver downloads.
   - Check free disk space on the target drive and the file system (warn on FAT32).
   - Check the install path (spaces OK; warn on non-ASCII / very long paths).
   - Show the **total download size** and ask for confirmation before downloading.
2. **Runtime** (into `runtime\`)
   - `uv` standalone binary → portable Python (version compatible with all deps;
     *verify* OmniVoice's supported Python versions, prefer 3.11 or 3.12).
   - Python dependencies into a venv inside `runtime\`, including **PyTorch built for
     CUDA 12.8 (`cu128`)** — this single build works on RTX 30/40/50 series.
   - **llama.cpp** prebuilt Windows CUDA binaries (pin a specific release tag; `llama-server.exe`).
   - **ffmpeg** portable binary.
3. **Core models** (into `models\`) — downloaded at install time:
   - **faster-whisper** model sized for the detected hardware profile (§5).
   - **OmniVoice** TTS model.
   - **Silero VAD**.
   - **No LLM** is downloaded at install time.
4. **Built-in voices:** generate the 8 built-in voices with OmniVoice voice design (§7.2)
   into `models\builtin_voices\`.
5. **Hardware profile:** write the detected profile to `user_data\settings.json`.
6. **Self-test:** mic capture test, TTS test (speak a short sentence with a built-in
   voice), GPU / CUDA availability check. Print a clear summary (pass / fail per item,
   with fix hints).
7. Print how to start the app (`start.bat`).

Also document expected warnings: Windows SmartScreen / antivirus prompts when running
a downloaded `.bat`, and that the first install downloads several GB.

### `start.bat`
Activates the portable environment (with all cache env vars), starts the backend,
opens the browser at `http://127.0.0.1:<port>`.

### `update.bat`
Updates the app code (git pull if it is a git checkout, otherwise download the latest
release) and updates dependencies. **Never touches `models\` or `user_data\`.**

---

## 5. Hardware profiles

Detected automatically from VRAM; user can override in Settings.

| Profile | VRAM | Whisper model | Default LLM quant | Notes |
|---|---|---|---|---|
| `high` | ≥ 24 GB | large-v3-turbo | Q5_K_M | All catalog models available |
| `mid` | 16–23 GB | large-v3-turbo (int8) | Q4_K_M | 24–30B models may need reduced context |
| `low` | 10–15 GB | small or turbo (int8) | Q4_K_M | Only ≤ 12B models enabled by default; Whisper may run on CPU |
| unsupported | < 10 GB | — | — | Warn; allow text-only / CPU mode as a best-effort fallback |

Runtime behaviour:
- Before loading a model, estimate VRAM need (model file size + context/KV cache) plus
  the currently loaded Whisper / OmniVoice footprint. Grey out models that cannot fit.
- If a model almost fits, allow **partial CPU offload** via llama.cpp `--n-gpu-layers`
  and show a notice that a smaller quant would be faster.
- When voice output is disabled, **unload OmniVoice** from VRAM. When mic input is
  disabled, **unload Whisper and VAD**. Reload on re-enable.
- Show current VRAM usage per component in the UI.

Treat the table values as starting points; tune during testing.

---

## 6. Architecture

```
Browser UI  ⇄  WebSocket / HTTP  ⇄  Python backend (FastAPI)
                                      ├── Audio in:  PCM from browser → Silero VAD → faster-whisper
                                      ├── LLM:       llama-server.exe subprocess (OpenAI-compatible API)
                                      ├── Text post: sentence splitter → speech sanitizer
                                      ├── TTS:       OmniVoice (in-process, optional)
                                      ├── Memory:    per-character history, summaries, facts
                                      └── Recorder:  session MP3 + Markdown transcript
```

Decisions:
- **Backend:** Python, FastAPI + WebSockets, bound to `127.0.0.1` only.
- **Frontend:** plain HTML / CSS / JavaScript served by the backend. **No Node.js build
  step** (keeps install simple). Small vendored libraries are fine if needed.
- **LLM:** `llama-server.exe` managed as a subprocess by the backend; restarted with the
  new model on model switch. Talk to it via its OpenAI-compatible HTTP API with streaming.
- **TTS:** OmniVoice used directly as a Python library in-process (pip package
  `omnivoice`). *Verify* its API for: reference audio input, whether a reference
  transcript is required (if so, auto-transcribe the reference with Whisper and cache it),
  language parameter, voice design attributes, streaming / chunked output, VRAM use.
- **STT:** faster-whisper with automatic language detection.
- **VAD:** Silero VAD.
- **Audio capture:** in the browser via `getUserMedia` (with `echoCancellation`,
  `noiseSuppression`), streamed as PCM over WebSocket. Keep the **full-quality** capture
  for recording; downsample a copy to 16 kHz for VAD / Whisper.
- **Audio playback:** in the browser, sentence by sentence as TTS chunks arrive.

---

## 7. Voices

### 7.1 Custom reference voices (most important feature)
- The user drops **MP3** (also accept WAV / FLAC / OGG) files into `user_data\voices\`,
  or uploads them through the UI (the UI copies the file into that folder).
- Every file there appears automatically as a selectable voice (name = file name,
  renamable in the UI).
- Preprocess with ffmpeg on first use: convert to the format OmniVoice expects (mono,
  required sample rate), trim leading/trailing silence, loudness-normalize. Cache the
  processed version and any derived data (e.g. reference transcript / embedding) so it
  is not recomputed every time; invalidate the cache when the source file changes.
- Recommend 3–10 s of clean speech in the docs. Warn in the UI if the clip is too short,
  too long, or very noisy (best effort).
- UI: list, preview (speak a sample sentence), rename, delete, add.

### 7.2 Built-in voices (8)
- **German, English, Spanish, French × male and female.**
- Generated at install time via **OmniVoice voice design** (attributes such as gender
  and accent / language). **No audio files are committed to the repository** (avoids
  any rights issues with recordings of real people).
- Definitions live in `config\builtin_voices.yaml`; generated assets in
  `models\builtin_voices\`. Provide a "Regenerate built-in voices" action.

### 7.3 Voice selection logic
- Each character has a voice (custom or built-in). The user can override the voice for
  the current session in the side panel.
- Option **"Match built-in voice to spoken language"**: if the active voice is a
  built-in voice, switch to the built-in voice of the same gender in the language the
  user is speaking (from Whisper's language detection).
- Always pass the reply language to OmniVoice if it accepts a language parameter.

### 7.4 Voice output toggle
- Voice output (OmniVoice) is **optional** and can be toggled globally and per character.
- OmniVoice is **always installed**, but when voice output is off the AI replies
  **as text only** (no fallback voice) and OmniVoice is unloaded from VRAM.

---

## 8. Speech sanitizer — actions are NEVER spoken

Roleplay models often write actions between asterisks (`*smiles and leans back*`).
These must **never** be sent to TTS — not quietly, not by a narrator voice.

Two layers:
1. **Prompting:** the system prompt instructs the model to write voice-friendly output:
   weave actions into spoken dialogue instead of narrating them, no markdown, no lists,
   no emojis.
2. **Strict filter before TTS** (unit-tested): remove
   - text between `*...*` and `_..._` used as action markers,
   - OOC text in `(( ... ))`, `[OOC: ...]`, and similar,
   - emojis,
   - markdown syntax (headers, bullets, code fences, bold / italic markers, links → link text),
   - anything left that is not pronounceable punctuation.
   Handle unclosed markers safely during streaming (buffer until the marker closes or
   the message ends; never speak a half-open action).

Display in the transcript, configurable per character:
- **Show actions as text** (dimmed, italic, with a small "not spoken" label), or
- **Hide actions completely.**

---

## 9. Conversation flow

1. **Listening:** VAD detects speech start / end. End of utterance = silence longer
   than a configurable threshold (default ~700 ms; tune during testing).
2. **Transcribing:** faster-whisper transcribes and detects the language.
3. **Thinking:** send to the LLM with system prompt, character persona, memory, recent
   history, and an instruction to reply in the user's language. Stream tokens.
4. **Speaking:** split the stream into sentences, sanitize, synthesize each sentence and
   play it while later sentences are still being generated.
5. **Mic is muted** (input ignored) from the start of playback until playback ends, so
   the AI never hears itself through speakers. Then return to Listening.

Text input:
- A **text input field is always available**, equal to voice. The user can switch
  between talking and typing at any time, even mid-conversation.
- Typed messages go through the same pipeline (memory, transcript, voice reply if
  enabled).
- **Text-only mode:** the microphone can be switched off completely (Whisper + VAD
  unloaded). Combined with voice output off, the app is a pure text chat.

UI states: `Listening`, `Thinking`, `Speaking`, plus `Mic off`, `Loading model`, `Error`.

---

## 10. Characters

Stored as one JSON (or YAML) file per character in `user_data\characters\`.

Fields (minimum):
- `id`, `name`, `kind` (`assistant` | `roleplay`), `avatar_initials`
- `persona` / description, `greeting`, `system_prompt`
- `voice` (custom file name or built-in id)
- `model` (optional; falls back to the globally selected model)
- `voice_output` (on / off), `show_actions_as_text` (on / off)
- `memory_enabled`, `recording_enabled`

Ship 4 example characters in `config\default_characters\`, copied to
`user_data\characters\` on first start:
- **Max** — assistant, built-in German male voice
- **Lena** — assistant, built-in German female voice
- **Elara** — roleplay, tavern keeper (fantasy)
- **Captain Vargas** — roleplay, starship captain (sci-fi)

Switching character switches persona, voice, model (if set), memory and history in one click.
UI: create, edit, duplicate, delete characters.

---

## 11. Memory

Characters remember previous conversations across sessions. Keep it simple and
transparent (no vector database in v1).

- **History:** the conversation history per character is persisted (SQLite recommended).
- **Rolling summary:** when history exceeds a token budget for the current model's
  context, older turns are summarized by the LLM; the summary replaces them in the prompt.
- **Facts list:** a short list of durable facts per character (e.g. "User's name is Tom")
  that the LLM updates after exchanges (e.g. every N turns, via a separate small prompt).
  Facts are injected into the system prompt.
- **UI:** view and edit facts, delete individual facts, reset a character's memory.
- Memory can be disabled per character.

---

## 12. Session recording

Recording is togglable globally and per character. Files go to `user_data\sessions\`,
named `YYYY-MM-DD_HHMM_<character-id>.mp3` and `.md`.

### 12.1 One MP3 for the whole dialogue
- **A single MP3 file per session** containing the complete dialogue in order: the
  user's voice and the AI's voice, alternating.
- **Write incrementally:** append each utterance as soon as it is complete (MP3 frames
  can be concatenated). After a crash, everything up to the last utterance must be intact.
- **Smooth pauses:** do not record waiting time (thinking latency). Insert a fixed short
  gap between turns (default 0.5 s, configurable).
- **Loudness normalization:** normalize user and AI audio to a common loudness.
- **User audio** comes from the full-quality browser capture, not the 16 kHz Whisper copy.
- Resample everything to one common sample rate before encoding.
- Default encoding: mono, **64 kbit/s** (≈ 29 MB per hour), configurable.
- Encoding runs on the CPU in the background; it must not delay the conversation.

### 12.2 Transcript
- Markdown file with timestamps for each utterance. Each timestamp is the **offset in
  the session MP3**.
- Typed turns and text-only replies appear in the transcript only (no audio).
- Filtered actions appear in the transcript according to the character's display
  setting, clearly marked as not spoken.

### 12.3 Replay
- In the UI each message has a replay button showing its timestamp; clicking it plays
  the session MP3 from that offset.
- Past sessions are listed in the sidebar and can be opened read-only with replay.

---

## 13. LLM model catalog

### 13.1 Catalog file
`config\model_catalog.yaml` — editable without touching code. Fields per entry:
`id`, `display_name`, `family`, `variant` (`censored` | `uncensored`), `purpose`,
`hf_repo`, `files` (per quantization: file name + approx size), `min_vram_gb` per quant,
`context_default`, `license_note`, `languages_note`.

### 13.2 Initial catalog (10 models)

| # | Model | Variant | Purpose | Target GPUs |
|---|---|---|---|---|
| 1 | Gemma 3 27B | censored | Assistant, very good German | 24 GB+ |
| 2 | Gemma 3 27B abliterated | uncensored | Assistant | 24 GB+ |
| 3 | Qwen3 30B-A3B | censored | Assistant, very fast (MoE) | 24 GB+ |
| 4 | Qwen3 30B-A3B abliterated | uncensored | Assistant, very fast | 24 GB+ |
| 5 | Mistral Small 3.2 24B | censored | Roleplay, good DE / FR / ES | 24 GB+ |
| 6 | Cydonia 24B (Mistral Small finetune) | uncensored | Roleplay | 24 GB+ |
| 7 | Gemma 3 12B | censored | Assistant + roleplay | all |
| 8 | Gemma 3 12B abliterated | uncensored | Assistant + roleplay | all |
| 9 | Rocinante 12B (Mistral Nemo finetune) | uncensored | Roleplay | all |
| 10 | Qwen3 8B | censored | Assistant, fastest | all |

Implementation notes:
- **Verify at implementation time** whether newer versions of these model families
  exist and propose updates to the owner before changing the list.
- Resolve exact Hugging Face GGUF repositories and file names at implementation time;
  prefer well-known GGUF publishers and verify that each file exists. Do not guess URLs.
- Document in the UI / docs that abliterated models can be slightly less precise than
  the originals, and that roleplay finetunes (Cydonia, Rocinante) are strongest in
  English and weaker in other languages.

### 13.3 Download behaviour
- **Nothing is downloaded until a model is selected for the first time.** Then download
  with a progress bar (resumable), then load automatically.
- Quantization chosen automatically from the hardware profile; user can override.
- Show per model: downloaded / not downloaded / downloading / loaded, and disk size.
- Allow deleting downloaded models.

### 13.4 Custom models
- Input field accepts a **Hugging Face repo id or a direct GGUF URL**, plus quantization
  (auto or explicit) and a censored / uncensored label.
- Download **only the single matching GGUF file** (e.g. `huggingface_hub.hf_hub_download`).
  **Do not** `git clone` model repositories (Git LFS, whole repos can be 100+ GB).
- GGUF files copied manually into `models\llm\` are detected automatically; their
  variant label defaults to "unknown" and is editable.
- Custom entries are stored in `user_data\` (not in the shipped catalog).

---

## 14. User interface

English only. Dark, calm theme (reference mockup: dark grey background, amber accent,
IBM Plex Sans / Plex Mono — or an equivalent font bundled locally, since the app runs
offline). Must remain usable in narrower windows.

### 14.1 Conversation screen (three columns)
- **Left:** character list (+ new character), recent sessions.
- **Center:** header with character name, role, active voice, recording indicator;
  message list (user right, AI left, replay button with timestamp per message, actions
  shown dimmed with "not spoken" label if enabled); large status button
  (Listening / Thinking / Speaking) with input level meter; pause-listening button;
  **text input field + Send button**.
- **Right:** Voice (custom reference MP3s, add MP3, 8 built-in voices, "match built-in
  voice to spoken language" toggle, voice output on/off); Language model (current model,
  variant badge, quant, VRAM, "Change model" → catalog); per-component VRAM bar; Memory
  (facts list, edit); Session (record toggle, show-actions toggle, current file names,
  open session folder).

### 14.2 Model catalog screen
Filter chips (All / Censored / Uncensored / Downloaded), table of models with purpose,
variant badge, size, status / action button (Download & load, Load, Loaded, progress);
"Add your own model" section (§13.4). Show detected GPU and recommended quant.

### 14.3 Settings screen
Audio input / output device, VAD sensitivity, end-of-speech silence threshold,
mic on/off, voice output on/off, hardware profile override, context length,
recording bitrate and turn gap, regenerate built-in voices.

### 14.4 Accessibility
Real buttons / inputs with labels, visible focus, sufficient contrast, keyboard usable.

---

## 15. Repository, documentation and quality

### 15.1 Repository hygiene
- `.gitignore`: `runtime/`, `models/`, `user_data/`, `logs/`, temp folders, Python caches.
  Personal data must never be committable by accident.
- No binaries or large files in the repo (GitHub limit 100 MB per file; keep the repo small).
- Pin dependency versions (lock file).

### 15.2 Documentation (English)
- `README.md`: what it is, feature list, screenshots, hardware requirements (§2),
  quick start (download → `install.bat` → `start.bat`), links to docs, disclaimers.
- `docs\`: `installation.md`, `voices.md` (custom MP3 voices, tips for good reference
  clips), `models.md` (catalog, custom models, quantization, VRAM), `characters.md`,
  `memory.md`, `recording.md`, `configuration.md`, `troubleshooting.md`
  (SmartScreen, drivers, CUDA, mic permissions, VRAM issues), `architecture.md`.
- `CHANGELOG.md` (Keep a Changelog style, semantic versioning).
- `LICENSE` — MIT, `Copyright (c) 2026 Slartibart23`.
- `THIRD_PARTY_NOTICES.md` — licenses of bundled / downloaded components
  (OmniVoice: Apache-2.0 — *verify* model weight license; llama.cpp, faster-whisper,
  Silero VAD: MIT — *verify*; ffmpeg build license; PyTorch).
- **Disclaimers** (README + docs):
  - Only clone voices with the explicit consent of the person whose voice it is;
    no impersonation, fraud or deception (mirrors OmniVoice's terms).
  - Users are responsible for how they use uncensored models.
  - LLMs are not redistributed by this project; each model has its own license
    (e.g. Gemma terms of use), which users must accept themselves.

### 15.3 Code quality
- Python with type hints, small focused modules, docstrings, clear user-facing error
  messages (no raw stack traces in the UI; full details in `logs\`).
- Unit tests at least for: speech sanitizer, sentence splitter, path resolution /
  portability, catalog parsing, hardware profile selection, MP3 append / timestamp logic.

---

## 16. Git and GitHub

- Initialize git in the program folder; commit in small, meaningful steps with English
  commit messages.
- Remote: `github.com/Slartibart23/local-lantern-talk`, **private**.
- Create the remote with the GitHub CLI (`gh`) if available and authenticated;
  otherwise give the owner step-by-step instructions.
- **Ask the owner for confirmation before creating the repository and before every
  push.** Never make the repository public; the owner does that manually.

---

## 17. Milestones (build in this order, test with the owner after each)

1. **M1 — Core loop (console):** mic → VAD → Whisper → llama-server → sanitizer →
   OmniVoice → playback, with one fixed model and voice. Validate latency and the
   "never speak actions" rule.
2. **M2 — Web UI:** conversation screen, text input, status states, voice / model
   selection with what exists so far.
3. **M3 — Portable installer:** `install.bat`, `start.bat`, cache redirection,
   GPU detection, hardware profiles, self-test. Test from a fresh folder.
4. **M4 — Model catalog:** on-demand download, custom models, VRAM checks, unloading.
5. **M5 — Voices:** custom MP3 voices end to end, built-in voice generation, language matching.
6. **M6 — Characters and memory.**
7. **M7 — Session recording:** single dialogue MP3, transcript, replay.
8. **M8 — Polish and docs:** settings screen, error handling, full documentation,
   screenshots, `update.bat`, final README.
9. **M9 — GitHub:** create private repo and push (with owner confirmation).

---

## 18. Items to verify before / during implementation

- OmniVoice: Python version support, API for cloning (reference text needed?),
  language parameter, voice design attributes, streaming, VRAM footprint, license of
  code and weights, Windows compatibility with CUDA 12.8 PyTorch.
- llama.cpp: current Windows CUDA release asset names; flags for GPU offload and context.
- faster-whisper on Windows with the chosen PyTorch / CTranslate2 CUDA versions
  (CTranslate2 may need matching cuDNN / cuBLAS DLLs — bundle them in `runtime\` if needed).
- Exact Hugging Face repos and file names for all 10 catalog models.
- Real VRAM usage on the RTX 5090 for each component; update §2 and §5 accordingly.
