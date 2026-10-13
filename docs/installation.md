# Installation

Local Lantern Talk is portable: everything — Python, libraries, models, caches and
your personal data — lives inside one program folder. It needs no administrator
rights, writes nothing to the registry and nothing to your user profile. Deleting
the folder removes the app completely.

## Requirements

| | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA RTX with 10 GB VRAM (e.g. RTX 3080) | NVIDIA RTX with 24 GB+ VRAM (RTX 4090 / 5090) |
| System RAM | 16 GB | 32 GB |
| Free disk space | ~30 GB (SSD recommended) | ~100 GB for several models |
| NVIDIA driver | 570.65 (CUDA 12.8) | 581 or newer |
| Windows | 10 or 11, 64-bit | |

- The drive must be formatted **NTFS or exFAT**. FAT32 cannot store files larger
  than 4 GB, and language models are bigger than that.
- The CUDA Toolkit is **not** needed. The installer downloads the CUDA runtime
  libraries into the program folder.
- Only NVIDIA GPUs are supported (no AMD / Intel GPUs, no macOS / Linux).
- Internet access is needed only for installation and model downloads.

## Install

1. Put the program folder where you want it, e.g. `D:\Local Lantern Talk`.
   Spaces in the path are fine. Avoid non-ASCII characters (umlauts) and very long
   paths; the installer warns about both.
2. Double-click **`install.bat`**.
3. The installer checks the GPU, driver, free space and file system, then shows
   what it will download and asks before it starts.
4. When it asks about the **starter language model**, answer *Y* unless you want
   to add a model yourself. Without a language model the app cannot answer.
   (A model catalog with one-click downloads follows in a later version.)
5. At the end the **self-test** runs. It plays a short English test sentence on
   your speakers and asks you to say a sentence into the microphone.
6. Start the app with **`start.bat`**.

`install.bat` can be run again at any time. Finished steps are skipped, and
interrupted downloads continue where they stopped. Run it again after an NVIDIA
driver update so the fastest llama.cpp build for the new driver is used.

Options (for advanced users), e.g. `install.bat -Yes`:

| Option | Effect |
|---|---|
| `-Yes` | Do not ask; accept the download (and the starter model) |
| `-NoStarterModel` | Do not download the starter language model |
| `-NoMicTest` | Skip the interactive microphone test |

### What gets downloaded

| Component | Size | Where |
|---|---|---|
| uv + portable Python 3.12 | ~40 MB | `runtime\uv`, `runtime\python` |
| Python packages incl. PyTorch for CUDA 12.8 | ~3.6 GB (≈ 7.7 GB unpacked) | `runtime\venv` |
| llama.cpp with CUDA runtime | ~0.6 GB | `runtime\llama.cpp` |
| ffmpeg | ~0.1 GB | `runtime\ffmpeg` |
| Whisper large-v3-turbo | ~1.6 GB | `models\whisper` |
| OmniVoice | ~3.2 GB | `models\omnivoice` |
| Silero VAD, fonts | < 5 MB | `models\vad`, `runtime\fonts` |
| Starter model Gemma 3 12B (optional) | 7.3–8.4 GB | `models\llm` |

Every runtime download (uv, llama.cpp, ffmpeg) is verified with a pinned SHA-256
checksum. The 8 built-in voices are not downloaded: they are generated on your GPU
during installation (about a minute).

## Hardware profiles

The installer picks a profile from your GPU's memory and stores it in
`user_data\settings.json` (`hardware_profile`). You can override it with
`hardware_profile_override` (a settings screen follows in a later version).

| Profile | VRAM | Whisper | Default model quantization |
|---|---|---|---|
| `high` | 24 GB+ | GPU, float16 | Q5_K_M |
| `mid` | 16–23 GB | GPU, int8 | Q4_K_M |
| `low` | 10–15 GB | GPU, int8 | Q4_K_M, models up to 12B |
| `unsupported` | < 10 GB | CPU | best effort; text-only recommended |

Cards report slightly less memory than their nominal size (a "24 GB" RTX 5090
Laptop GPU reports 23.9 GB), so the limits are set just below the nominal values.

### Measured memory use

Measured on an **RTX 5090 Laptop GPU (24 GB)**, driver 616.92:

| Component | VRAM |
|---|---|
| Whisper large-v3-turbo (float16) | ~2.4 GB |
| OmniVoice (fp16) | ~2.3 GB, up to ~4.7 GB while speaking |
| Gemma 3 12B Q5_K_M, 8k context | ~9.4 GB |

Values for other GPUs (RTX 3080, RTX 4090, desktop RTX 5090 with 32 GB) are
estimates until someone measures them on that hardware.

## Expected warnings

- **Windows SmartScreen** may show *"Windows protected your PC"* when you run a
  downloaded `.bat` file. Click *More info* → *Run anyway*. The scripts are plain
  text; you can open them in Notepad to see what they do.
- **Antivirus software** may scan or briefly block the downloaded programs
  (`uv.exe`, `llama-server.exe`, `ffmpeg.exe`). They come from their official
  GitHub releases and are checksum-verified.
- **Windows Defender Firewall** should not ask: the app only listens on
  `127.0.0.1` (this computer). If it asks anyway, you can deny network access.
- The first install downloads **several GB** and can take a while on slow
  connections. It can be interrupted and resumed.

## Uninstall

Close the app and delete the program folder. Nothing else needs to be removed.
