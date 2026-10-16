# Troubleshooting

**Getting help:** copy the error message — from the command window or from
`logs\server.log` — into Claude (<https://claude.ai>) and ask what to do. Describe what
you did right before the error; Claude can usually explain the message and the fix.

Most problems show a message in the app. Full details are always in `logs\`:
`server.log` (the app), `llama-server.log` (the language model), `selftest.log` and the
setup logs. Re-running `install.bat` is safe: it repairs missing parts and skips what is
already there.

## Installation

**"Windows protected your PC" (SmartScreen)** when starting a `.bat` file: click
*More info* → *Run anyway*. The scripts are plain text; open them in Notepad to see what
they do.

**Antivirus blocks or quarantines a program** (`uv.exe`, `llama-server.exe`,
`ffmpeg.exe`): these come from their official GitHub releases and are checksum-verified.
Allow them and run `install.bat` again.

**"No NVIDIA GPU / driver found"**: install or update the NVIDIA driver from
<https://www.nvidia.com/en-us/drivers/>. The app needs an NVIDIA RTX GPU.

**"Driver … is older than 570.65"**: PyTorch needs CUDA 12.8, which requires driver
570.65 or newer. Update the driver and run `install.bat` again. With a driver 570–580
the installer uses the CUDA 12 build of llama.cpp; after updating to 581+ run
`install.bat` again to switch to the faster CUDA 13 build.

**"llama.cpp could not use the GPU"**: run
`runtime\llama.cpp\llama-server.exe --list-devices` in a terminal. If no CUDA device is
listed, update the driver. To try the other build: `set LLT_LLAMA_BUILD=cuda-12.4`,
then `install.bat` in the same terminal.

**Download interrupted / slow**: run `install.bat` again; downloads continue where they
stopped.

**FAT32 drive**: files larger than 4 GB (language models) cannot be stored. Use an NTFS
or exFAT drive.

**Odd errors with a path containing umlauts or a very long path**: move the program
folder to a short, plain path like `D:\LocalLanternTalk`.

## Starting

**`start.bat` says "Local Lantern Talk is already running"**: the app is already open;
the browser tab is opened again. Use *Quit app* in the old tab (or close the old
`start.bat` window) to stop it.

**The browser shows "This site can't be reached"**: the app is still starting or was
closed. Look at the `start.bat` window for errors.

**"This conversation was opened in another browser tab"**: only one tab is active at a
time. Use the newest tab, or click *Use this tab instead*.

## Microphone

**"Mic blocked" / "Microphone access was not allowed"**: allow the microphone for
`127.0.0.1` in the browser (the icon left of the address bar), then reload the page.
In Windows, check *Settings → Privacy & security → Microphone* ("Let desktop apps access
your microphone" and the browser must be allowed).

**The wrong microphone is used**: choose it in **Settings → Audio devices**.

**The app does not react when I speak**: watch the level meter next to the status. If
it does not move, the browser gets no sound (wrong device, muted microphone). If it moves
but nothing happens, lower **Speech detection** in Settings.

**My sentence is cut off / the character interrupts my thinking pauses**: switch the talk
mode under the chat to **Smart** — it waits while you sound unfinished and answers quickly
when you are done — or to **Push to talk**, or click **Hold on** while you think and
**Send** when you are done. In *Auto* you can also move **Thinking pauses → End of speech
after** in the right panel to the right (1.2–2 s works well; up to 3 s).

**Smart says the model could not be downloaded**: older installations fetch the Smart
Turn model (9 MB) once when you choose *Smart*; that needs an internet connection. Running
`install.bat` again downloads it as well.

**Push to talk does not react to the Space bar**: click into an empty part of the chat
first; while you type in a text field, Space types a space.

**The app reacts to background noise or its own voice**: raise **Speech detection**, and
use headphones if your speakers are loud. The microphone is muted while the character
speaks and while a recording is replayed.

## Speech output

**No sound**: check the speaker in **Settings → Audio devices** and the system volume.
The browser only plays sound after you clicked *Start*.

**Replies are text only**: *Voice output* is off globally (right panel) or for this
character (character editor).

**A cloned voice sounds wrong**: see [voices.md](voices.md) — a clean 10–13 s clip and an
exact reference text matter most.

## Language model and GPU memory

**"… is too big for the free GPU memory"** or **"The GPU ran out of memory"**: pick a
smaller model or quantization in *Change model…*, switch the **Context length** in
Settings to *Automatic* (or move it into the green zone), close other programs that use
the GPU (games, video editors, other AI tools such as LM Studio), or turn voice output
off. The VRAM bar in the right panel shows what uses the memory.

**Replies are slow and a notice says part of the model runs on the CPU**: the model does
not fit completely into GPU memory. A smaller quantization is faster.

**"The language model had stopped and was started again"**: llama-server crashed (often
because GPU memory ran out). If it happens repeatedly, choose a smaller model.

**A downloaded model does not load**: only GGUF files work. See
[models.md](models.md#your-own-models).

**The model writes its reasoning or speaks stage directions**: reasoning is disabled and
filtered; actions in `*asterisks*` are never spoken. If a model still misbehaves, try
another model from the catalog.

## Updating

**"Local Lantern Talk is running. Quit it first"**: use *Quit app* or close the
`start.bat` window, then run `update.bat` again.

**"github.com/… could not be reached"** (installation from a ZIP): `update.bat` downloads
the newest version from GitHub. While the repository is private, GitHub only shows it to
logged-in users: install the GitHub CLI once (`winget install --id GitHub.cli -e`, then a
new window) and log in with `gh auth login --hostname github.com --git-protocol https --web`.

**"This git checkout has local changes"**: you edited files in the program folder.
Commit or stash them (or undo them), then update again.

## Starting over

- Reset a character's memory: right panel → *Reset this character's memory…*.
- Reset all settings: close the app and delete `user_data\settings.json`.
- Restore the example characters: close the app and delete
  `user_data\characters\.defaults.json` (and `.defaults-copied`, if it is there); missing
  example characters are copied again on the next start (your own characters stay).
- Rebuild the runtime: delete `runtime\` and run `install.bat` (models and your data
  stay).
