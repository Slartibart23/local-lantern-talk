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

## Get the app

Create a new, empty folder with a short path, e.g. `D:\LocalLanternTalk`. Spaces are
fine; avoid umlauts, very long paths and folders synced by OneDrive (the Desktop and
*Documents* often are), because the models are many GB. Then either:

- **Download the ZIP:** on the project page on GitHub
  (<https://github.com/Slartibart23/local-lantern-talk>) click the green **Code** button
  → **Download ZIP**. Open the ZIP; it contains one folder, `local-lantern-talk-main`.
  Copy **its contents** (`install.bat`, `start.bat`, `app`, `config`, …) into your new
  folder. Optional: to avoid a Windows warning for every script, open PowerShell in the
  folder and run `Get-ChildItem -Recurse | Unblock-File`.
- **Or clone it with git** (makes updates easiest — `update.bat` then simply pulls the
  new version):

  ```
  git clone https://github.com/Slartibart23/local-lantern-talk D:\LocalLanternTalk
  ```

## Install

Run the installer in a **command window** so you can read all its messages:

1. Open your folder in Explorer, click into the address bar, type `cmd` and press
   **Enter**. A command window opens in that folder.
2. Type `install.bat` and press **Enter**. (Double-clicking `install.bat` works too; its
   window stays open at the end. In a PowerShell window type `.\install.bat`.)
3. The installer checks the GPU, driver, free space and file system, then shows
   what it will download and asks before it starts.
4. When it asks about the **starter language model**, answer *Y* unless you want
   to add a model yourself. Without a language model the app cannot answer. More
   models can be downloaded later in the app (**Change model…**).
5. At the end the **self-test** runs. It plays a short English test sentence on
   your speakers and asks you to say a sentence into the microphone.
6. Start the app: type `start.bat` in the same window (or double-click it). Loading
   the models takes up to 2 minutes; the browser page shows a loading screen and tells
   you when it is ready.

If something fails, copy the messages from the command window into Claude
(<https://claude.ai>) and ask what to do — see also [troubleshooting.md](troubleshooting.md).

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
| Silero VAD, Smart Turn v3.2 (talk mode Smart), fonts | ~12 MB | `models\vad`, `runtime\fonts` |
| Starter model Gemma 4 12B (optional) | 7.1–8.4 GB | `models\llm` |

Every runtime download (uv, llama.cpp, ffmpeg) and the Smart Turn model are verified with
a pinned SHA-256 checksum. Installations from before Smart Turn existed download it
(9 MB) the first time you choose the talk mode *Smart*. The 8 built-in voices come with the app (`config\builtin_voices\`, under 1 MB);
the installer only prepares them for OmniVoice (a few seconds).

## Hardware profiles

The app picks a profile from your GPU's memory and stores it in
`user_data\settings.json` (`hardware_profile`); a new GPU or a new profile is picked up
at the next start. You can choose another one in **Settings → Language model → Hardware
profile** (`hardware_profile_override`); the **i** next to it explains what it does.

| Profile | VRAM | Whisper | Default model quantization | Automatic context up to |
|---|---|---|---|---|
| `ultra` | 32 GB+ | GPU, float16 | Q6_K | 32 768 tokens |
| `high` | 24–31 GB | GPU, float16 | Q5_K_M | 32 768 tokens |
| `mid` | 16–23 GB | GPU, int8 | Q4_K_M | 16 384 tokens |
| `low` | 10–15 GB | GPU, int8 | Q4_K_M, models up to 12B | 8 192 tokens |
| `unsupported` | < 10 GB | CPU | best effort; text-only recommended | 4 096 tokens |

The automatic context only goes as high as it still fits completely on the GPU, so the
limit is a maximum, not a promise.

Cards report slightly less memory than their nominal size (a "24 GB" RTX 5090
Laptop GPU reports 23.9 GB, a desktop RTX 5090 31.8 GB), so the limits are set just
below the nominal values.

### Measured memory use

Measured on an **RTX 5090 Laptop GPU (24 GB)**, driver 616.92:

| Component | VRAM |
|---|---|
| Whisper large-v3-turbo (float16) | ~2.4 GB |
| OmniVoice (fp16) | ~2.3 GB, up to ~4.7 GB while speaking |
| Gemma 3 12B Q5_K_M, 8k context | ~9.4 GB |

Measured on a **desktop RTX 5090 (32 GB)**, driver 616.92, profile `ultra`:

| Component | VRAM |
|---|---|
| Whisper large-v3-turbo (float16) + Silero VAD | ~2.6 GB |
| OmniVoice (fp16) | ~2.1 GB, more while speaking |
| Gemma 4 12B Q5_K_M, 32k context (Automatic) | ~9.7 GB |
| Smart Turn v3.2 | runs on the CPU (about 15 ms per check) |

Loading everything took 34 s; the first spoken word followed a typed message after
1.3 s. Values for other GPUs (RTX 3080, RTX 4090) are estimates until someone measures
them on that hardware.

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
