# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- M1 core loop (console): microphone → Silero VAD → faster-whisper → llama-server →
  speech sanitizer → OmniVoice → playback, with typed input as an alternative.
- Speech sanitizer that never speaks actions (`*…*`, `_…_`, `[…]`), OOC text, emojis or
  markdown, including safe handling of unclosed markers while streaming.
- Streaming sentence splitter for low-latency, sentence-by-sentence speech.
- Eight built-in voices (DE / EN / ES / FR × male / female) generated locally via
  OmniVoice voice design and verified with Whisper.
- Portable runtime setup (`install.bat`): uv, Python 3.12, PyTorch cu128, llama.cpp,
  core models; all caches redirected into the program folder.
- Installer checks the NVIDIA driver before downloading (minimum 570.65 / CUDA 12.8)
  and selects the llama.cpp build from it: CUDA 13 for driver 581+, CUDA 12 otherwise.
  The matching CUDA runtime DLLs are installed into `runtime\llama.cpp\`, and the
  installer verifies that llama.cpp finds the GPU without any CUDA Toolkit on `PATH`.
- Built-in voices follow the spoken language (spec §7.3, on by default): an English
  built-in voice switches to the same-gender German / Spanish / French one, so replies
  are not spoken with a foreign accent. `console.bat --no-match-language` disables it.
- `LLT_LLAMA_BUILD` environment variable to force a llama.cpp build (troubleshooting).
- Web UI (`start.bat`): conversation screen with status button, level meter, pause,
  text input, actions shown as "not spoken", voice / model selection, microphone and
  voice-output toggles (unloading Whisper / VAD / OmniVoice), GPU memory display.
- "Add voice…" uploads an MP3 / WAV / FLAC / OGG reference clip into `user_data\voices`,
  with a warning for clips outside the recommended 3–10 s.
- Reference text for custom voices: the Whisper transcript is shown and can be corrected
  in the UI (saved as `clip.txt` next to `clip.mp3`); the voice is rebuilt with it.
- Speech quality settings: OmniVoice synthesis steps (16-64) and precision (fp16 / bf16 /
  fp32). There is only one official OmniVoice model; these are its quality levers.
- Typed messages get their language guessed (DE / EN / ES / FR) so the reply voice matches.
- Installer (M3): pre-checks (driver, VRAM, free space, FAT32, install path), download
  plan with confirmation, SHA-256-verified downloads, portable ffmpeg, hardware profiles
  (high / mid / low / unsupported) stored in settings, built-in voices generated at
  install time, optional starter model, and a self-test (GPU / CUDA, files, speech
  output checked by Whisper, microphone). Options `-Yes`, `-NoStarterModel`, `-NoMicTest`.
- `docs/installation.md` with requirements, hardware profiles, measured VRAM and the
  expected Windows warnings.
- `docs/voices.md`: clip length limits, what makes a good reference recording, and how
  to provide the transcript yourself instead of using Whisper.
- Git pre-commit hook and test that keep personal data and binaries out of the repository.
- Model catalog (M4): 9 German-focused models (Gemma 4, Mistral Small 3.2 and roleplay
  finetunes, censored and uncensored) in `config/model_catalog.yaml`; one-click download
  of a single GGUF file with progress, cancel and resume, loaded automatically when done;
  quantization chosen from the hardware profile and the free GPU memory; fits / partly on
  CPU / too big badges; custom models from a Hugging Face repo id, page/file link or direct
  URL; hand-copied GGUF files detected and labelable; delete downloaded models.
- Exact GPU/CPU split from llama-fit-params (incl. mixture-of-experts expert offload),
  with a notice when part of the model runs on the CPU; the free-memory estimate is
  corrected for Windows, where CUDA over-reports free VRAM.
- Per-component VRAM bar (language model, Whisper, OmniVoice, other).
- `docs/models.md`.
- Voices (M5): custom clips are prepared with ffmpeg (silence trimmed, loudness
  normalized, 24 kHz mono) and cached; quality hints for length, background noise and
  clipping; more formats (OPUS, M4A, AAC, WEBM); preview, rename and delete voices;
  "Regenerate built-in voices"; custom voices use the language of their transcript.
- Characters (M6): four example characters (Lena, Max, Elara, Captain Vargas) copied to
  `user_data\characters` on first start; create, edit, duplicate and delete characters
  in the UI; switching a character switches persona, voice, model (if set), memory and
  history in one click; roleplay characters get stay-in-character rules; per-character
  voice output and show-actions settings.
- Memory (M6): per-character history in SQLite (`user_data\memory`), a rolling summary
  when the history outgrows the model's context, a facts list updated by the model every
  4 replies, "End chat" folds the old talk into the summary; facts can be added,
  edited and deleted, and a character's memory reset, in the right panel.
- `docs/characters.md` and `docs/memory.md`.
- "End chat" button in the conversation header, and a "Continue / Start new chat" bar
  when a character's last chat is loaded after a restart or a switch.
- Session recording (M7): one MP3 per chat with the whole dialogue (full-quality user
  audio, common loudness, 0.5 s gaps instead of waiting time, 64 kbit/s mono), appended
  utterance by utterance in the background so it survives crashes; a screenplay-style
  Markdown transcript ("Elara: …", spoken words only) with the MP3 offsets kept in a
  hidden index for replay; replay buttons on every message (microphone muted while playing);
  recent sessions in the sidebar, opened read-only with replay; global and per-character
  switch, REC badge, "Open session folder". `docs/recording.md`.
- Settings screen (M8): audio input / output device, speech detection, end of speech,
  microphone and voice output, synthesis quality, context length, hardware profile
  override, recording quality and turn gap, regenerate built-in voices.
- `update.bat`: updates the app code (git pull, or the latest GitHub release) and the
  Python packages / runtime tools; never touches `models\` or `user_data\`.
- Clear error messages for GPU out-of-memory and lost language-model connections; a
  crashed llama-server is restarted automatically before the next turn.
- Documentation: `configuration.md`, `troubleshooting.md`, `architecture.md`,
  `THIRD_PARTY_NOTICES.md`, screenshots and a rewritten README.
- Loading screen: after start the page shows that the models are loading (up to about
  2 minutes) and offers *Start* only when everything is ready; side-panel and settings
  controls are locked while models load, so nothing is switched off by accident.
- Language of a custom voice clip can be chosen (or detected automatically); Whisper
  transcribes the clip with it.
- The 8 built-in voices are shipped as short FLAC clips in `config\builtin_voices\`, so
  every installation sounds the same and the installer no longer designs them.
- "Need help? Ask Claude" hints in installer, update, start-up and error messages and
  in the docs; installation guide covers ZIP download vs. `git clone` and running the
  installer in a command window.
- Decorations: a string of twinkling fairy lights and two swaying red paper lanterns
  under the chat header, a glowing lantern logo; can be switched off in
  Settings → Appearance.
- Scenes per character (character editor → Scene): Tavern, Space, Red lantern, Study or
  Plain change the chat background, accent colour and fairy-light colours, and add
  hand-drawn cliparts to the side panels; the reply being spoken (or replayed) glows.
- "Thinking pauses" slider back in the right panel; the end-of-speech pause now goes up to
  3 s so the character does not interrupt you while you think.
- Model catalog: Magidonia 24B v4.3 Heretic and Qwen 3.8 27B Abliterated; Cydonia 24B v4.3
  now uses bartowski's imatrix quantizations.
- "Quit app" button: closes the app from the browser, unloads all models and frees the
  GPU memory; conversations are saved.
- Talk modes under the chat, one click to switch: *Auto* (a pause ends the turn), *Smart*
  (Smart Turn v3.2 hears whether you sound finished and waits while you think; 8 MB ONNX
  model on the CPU, downloaded by the installer or once on first use) and *Push to talk*
  (Talk button or Space; hold, or click to start and again to send). *Hold on* collects
  everything you say, including the sentence in progress, until *Send*.
- *Stop* button (and Esc) ends the reply that is being written or spoken.
- Delete messages from the end of a chat (with everything after them): they are removed
  from the chat, the memory, the LLM history and the session recording (MP3, transcript
  and index are cut). *Write again* for the last reply, *Edit* for your last message.
- Director's notes under the message box: a note for the next reply, notes pinned to the
  character (kept across *End chat* and restarts), and *Continue* to let the character go
  on by itself. Never spoken and never in the transcript.
- Pictures and colours: an avatar per character (cropped to a circle in the browser,
  shown in the list and at the top left of the chat), a washed-out chat background per
  character, and 16 text colours for the character's and your own messages.
- Your profile (Settings): name, a few words about you (sent to every character), your
  own avatar at the top right of the chat and your text colour; your name replaces "You"
  in session transcripts.
- Character export / import: one file with description, look, pictures and pinned notes;
  memory and cloned voices are never included.
- Family mode (Settings): hides uncensored and explicit models and characters marked
  *For adults only*.
- Context length: *Automatic* (default) loads the largest context that fits completely on
  the GPU, capped by the hardware profile; otherwise a slider in 1024-token steps up to the
  model's maximum with green / yellow / red zones measured for the loaded model
  (llama-fit-params at three context lengths, interpolated). Explanations behind **i**.
- Hardware profile `ultra` for 32 GB+ GPUs (Q6_K, automatic context up to 32k); a new
  profile is picked up at the next start without reinstalling.
- Model catalog: a new label *explicit* (filter, badge, label for your own models) with
  Qwen 3.6 27B Uncensored Aggressive (HauhauCS, K_P quantizations), Qwen 3.6 27B
  Abliterated, Mistral Small 3.2 24B Abliterated v2, Mistral Small 24B (2501) Abliterated
  and Dolphin 3.0 Mistral 24B.
- English tooltips on every field and button.
- A notice on the start screen, in Settings and in the README: use the app at your own
  risk, responsibly and with respect for other people's privacy.

### Changed
- The example characters greet in English and use the English built-in voices; they still
  answer in the language you speak.
- The memory summary grows with the context (200 words at 8k, 400 at 16k, 800 at 32k).
- License: PolyForm Strict 1.0.0 instead of MIT — noncommercial use; changing the software,
  sharing changed versions or commercial use only with the author's permission. Own data and
  configuration files may be changed (additional permissions in `LICENSE`).
- The installer's starter model is Gemma 4 12B (Apache-2.0) instead of Gemma 3 12B.
- No language model is downloaded automatically any more; the installer offers the
  starter model instead (spec §4).

### Fixed
- After an update the browser could keep using an old cached `app.js`, so new controls (e.g.
  Decorations) did nothing; the page's scripts and styles are now always revalidated.
- `update.bat` on a ZIP installation: falls back to the main branch while there is no
  release, and uses the GitHub CLI login for a private repository.
- Switches were drawn on the checkbox itself, which Firefox does not render; they are now
  drawn next to it and show their state in every browser.
- The greeting no longer starts the chat history sent to the model: strict chat templates
  (Mistral, Gemma) require the conversation to start with the user.
- Model reasoning is never spoken: thinking is disabled in the chat template and
  `<think>` / `[THINK]` blocks are filtered by the sanitizer (Qwen3 spoke its reasoning).
- `llama-server.exe` kept running (holding ~9 GB VRAM) after the console window was
  closed; child processes are now tied to the app with a Windows job object.
- First reply after start was ~1.3 s slower: OmniVoice is now warmed up while loading.
- Short sentences were synthesized one by one; sentences that arrive while speech is
  being generated are now batched into one TTS call (each call has a fixed cost).
- Harmless `SyntaxWarning`s from pydub and weight-loading progress bars no longer
  clutter the console.
