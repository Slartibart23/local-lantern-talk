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
