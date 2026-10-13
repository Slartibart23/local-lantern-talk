# Third-party notices

Local Lantern Talk itself is MIT-licensed (see [LICENSE](LICENSE)). The repository
contains only its own code and configuration. All components below are **downloaded by
`install.bat` or by the app** from their official sources; they are not redistributed by
this project. Each is used under its own license.

## Models

| Component | Source | License |
| --- | --- | --- |
| **OmniVoice** (speech synthesis) — code | [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) | Apache-2.0 |
| **OmniVoice** — pre-trained model weights | [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | **CC BY-NC (non-commercial)**, because of its training data (e.g. Emilia) |
| Whisper large-v3-turbo (CTranslate2 conversion) | [dropbox-dash/faster-whisper-large-v3-turbo](https://huggingface.co/dropbox-dash/faster-whisper-large-v3-turbo), based on [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | MIT |
| Silero VAD | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | MIT |
| Language models (catalog, starter model, your own) | Hugging Face, see `config\model_catalog.yaml` | Each model's own license (e.g. Gemma terms of use, Apache-2.0, Mistral licenses), shown in the model catalog. You accept it when you download the model. |

> **Non-commercial use:** the OmniVoice model weights are licensed for non-commercial
> use only. Because every spoken reply (built-in and cloned voices) is produced with
> them, using Local Lantern Talk's voice output commercially is not permitted by that
> license.

## Programs

| Component | Source | License |
| --- | --- | --- |
| llama.cpp (`llama-server.exe`) | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT |
| NVIDIA CUDA runtime (cudart, cuBLAS DLLs shipped with llama.cpp) | [ggml-org/llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) | NVIDIA CUDA Toolkit EULA (redistributable runtime components) |
| ffmpeg 9.0.2 "essentials" build | [GyanD/codexffmpeg](https://github.com/GyanD/codexffmpeg) | GPL-3.0 (run as a separate program; not linked into the app) |
| uv | [astral-sh/uv](https://github.com/astral-sh/uv) | MIT or Apache-2.0 |
| Python 3.12 (python-build-standalone via uv) | [astral-sh/python-build-standalone](https://github.com/astral-sh/python-build-standalone) | PSF License |

## Python packages

Installed into `runtime\venv\` from PyPI / the PyTorch index (exact versions in `uv.lock`).

| Package | License |
| --- | --- |
| PyTorch, torchaudio (CUDA 12.8 builds) | BSD-3-Clause / BSD-2-Clause; the CUDA wheels include NVIDIA libraries under the NVIDIA EULA |
| omnivoice | Apache-2.0 |
| faster-whisper | MIT |
| CTranslate2 | MIT |
| huggingface-hub | Apache-2.0 |
| NumPy | BSD-3-Clause |
| soundfile | BSD-3-Clause (bundles libsndfile, LGPL-2.1) |
| sounddevice | MIT (bundles PortAudio, MIT) |
| httpx | BSD-3-Clause |
| PyYAML | MIT |
| FastAPI | MIT |
| Uvicorn | BSD-3-Clause |
| websockets | BSD-3-Clause |
| pytest (development only) | MIT |

Their own dependencies are listed in `uv.lock`; each package's license file is in
`runtime\venv\Lib\site-packages\<package>.dist-info\`.

## Fonts

| Component | Source | License |
| --- | --- | --- |
| IBM Plex Sans, IBM Plex Mono | [IBM/plex](https://github.com/IBM/plex) via [Fontsource](https://fontsource.org) | SIL Open Font License 1.1 |

The fonts are downloaded into `runtime\fonts\` so the app works offline.
