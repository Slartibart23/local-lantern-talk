"""Installation self-test (spec §4 step 6): GPU / CUDA, files, speech output, microphone.

Run by install.bat, or any time: ``python -m lantern_talk.selftest``.
``--no-mic`` skips the interactive microphone test, ``--no-sound`` does not play audio.
Exit code 1 if any check FAILED (warnings do not fail).
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

from .hardware import detect_gpu, effective_profile
from .logging_setup import setup_logging
from .paths import Paths, apply_portable_env, get_paths
from .settings import SettingsStore
from .textmatch import word_similarity

log = logging.getLogger(__name__)

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
TTS_SENTENCE = "Hello! This is the self test of Local Lantern Talk."


@dataclass
class Result:
    name: str
    status: str
    detail: str
    hint: str = ""


def _run(cmd: list[str], timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


# ---------------------------------------------------------------------------- checks

def check_gpu() -> Result:
    gpu = detect_gpu()
    if gpu is None:
        return Result("NVIDIA GPU", FAIL, "nvidia-smi found no NVIDIA GPU",
                      "Install the latest NVIDIA driver: https://www.nvidia.com/en-us/drivers/")
    return Result("NVIDIA GPU", PASS, f"{gpu.name}, {gpu.vram_mib / 1024:.1f} GB VRAM, driver {gpu.driver}")


def check_torch_cuda() -> Result:
    try:
        import torch

        if not torch.cuda.is_available():
            return Result("PyTorch CUDA", FAIL, "CUDA not available to PyTorch",
                          "Update the NVIDIA driver (570.65 or newer), then re-run install.bat.")
        return Result("PyTorch CUDA", PASS, f"torch {torch.__version__}, CUDA {torch.version.cuda}")
    except Exception as exc:  # noqa: BLE001
        return Result("PyTorch CUDA", FAIL, f"PyTorch failed to load: {exc}", "Re-run install.bat.")


def check_ctranslate2(profile_device: str) -> Result:
    if profile_device != "cuda":
        return Result("Whisper GPU", WARN, "Whisper runs on the CPU for this hardware profile (slower)")
    try:
        from .gpu import register_cuda_dlls

        register_cuda_dlls()
        import ctranslate2

        n = ctranslate2.get_cuda_device_count()
        if n < 1:
            return Result("Whisper GPU", FAIL, "CTranslate2 sees no CUDA device", "Re-run install.bat.")
        return Result("Whisper GPU", PASS, f"CTranslate2 {ctranslate2.__version__}, {n} CUDA device(s)")
    except Exception as exc:  # noqa: BLE001
        return Result("Whisper GPU", FAIL, f"CTranslate2 failed to load: {exc}", "Re-run install.bat.")


def check_llama(paths: Paths) -> Result:
    if not paths.llama_server.exists():
        return Result("llama.cpp", FAIL, "llama-server.exe missing", "Re-run install.bat.")
    try:
        out = _run([str(paths.llama_server), "--list-devices"])
        text = out.stdout + out.stderr
    except Exception as exc:  # noqa: BLE001
        return Result("llama.cpp", FAIL, f"llama-server did not start: {exc}", "Re-run install.bat.")
    devices = [line.strip() for line in text.splitlines() if line.strip().startswith("CUDA")]
    if not devices:
        return Result("llama.cpp", FAIL, "llama-server found no CUDA device (would run on the CPU)",
                      "Update the NVIDIA driver, then re-run install.bat.")
    return Result("llama.cpp", PASS, devices[0].split("(")[0].strip())


def check_ffmpeg(paths: Paths) -> Result:
    if not paths.ffmpeg.exists():
        return Result("ffmpeg", FAIL, "ffmpeg.exe missing", "Re-run install.bat.")
    try:
        first = _run([str(paths.ffmpeg), "-version"]).stdout.splitlines()[0]
        return Result("ffmpeg", PASS, " ".join(first.split()[:3]))
    except Exception as exc:  # noqa: BLE001
        return Result("ffmpeg", FAIL, f"ffmpeg did not run: {exc}", "Re-run install.bat.")


def check_files(paths: Paths) -> list[Result]:
    results = []
    core = {
        "Whisper model": paths.whisper_models / "large-v3-turbo" / "model.bin",
        "OmniVoice model": paths.omnivoice_model / "model.safetensors",
        "Silero VAD": paths.vad_models / "silero_vad.jit",
    }
    for name, f in core.items():
        results.append(Result(name, PASS, "present") if f.exists()
                       else Result(name, FAIL, f"missing: {f.relative_to(paths.root)}", "Re-run install.bat."))
    voices = list(paths.builtin_voices.glob("*.pt"))
    results.append(Result("Built-in voices", PASS if len(voices) >= 8 else FAIL, f"{len(voices)} of 8",
                          "" if len(voices) >= 8 else "Re-run install.bat to generate them."))
    models = sorted(paths.llm_models.glob("*.gguf"))
    results.append(Result("Language model", PASS, ", ".join(m.name for m in models)) if models else
                   Result("Language model", WARN, "none in models\\llm",
                          "Re-run install.bat and accept the starter model, or copy a GGUF file into models\\llm."))
    return results


def check_tts(paths: Paths, settings, stt, play: bool) -> Result:  # noqa: ANN001
    """Speak a sentence with a built-in voice and verify it with Whisper."""
    from .tts import OmniVoiceTTS, VoiceLibrary

    try:
        tts = OmniVoiceTTS(paths.omnivoice_model, num_step=settings.num_step, precision=settings.tts_precision)
        voices = VoiceLibrary(tts, paths.config / "builtin_voices.yaml", paths.builtin_voices, paths.voice_cache,
                              transcribe=lambda a, sr, lang: stt.transcribe_at(a, sr, lang).text)
        tts.load()
        result = tts.synthesize(TTS_SENTENCE, voices.prompt_for("builtin:en_female"), "en")
        heard = stt.transcribe_at(result.audio, 24_000, "en").text
        score = word_similarity(TTS_SENTENCE, heard)
        tts.unload()
    except Exception as exc:  # noqa: BLE001
        log.exception("TTS self-test failed")
        return Result("Speech output", FAIL, f"OmniVoice failed: {exc}", "See logs\\selftest.log.")
    played = ""
    if play:
        try:
            import sounddevice as sd

            sd.play(result.audio, 24_000)
            sd.wait()
            played = ", played on the default speakers"
        except Exception as exc:  # noqa: BLE001
            return Result("Speech output", WARN, f"speech generated but playback failed: {exc}",
                          "Check that a speaker / headset is connected and set as default output device.")
    detail = f"synthesized in {result.seconds:.1f} s, Whisper understood {score:.0%}{played}"
    if score < 0.75:
        return Result("Speech output", WARN, detail + f" (heard: {heard!r})", "Re-run install.bat to regenerate voices.")
    return Result("Speech output", PASS, detail)


def check_mic(stt, interactive: bool) -> Result:  # noqa: ANN001
    try:
        import sounddevice as sd

        device = sd.query_devices(kind="input")
    except Exception as exc:  # noqa: BLE001
        return Result("Microphone", WARN, f"no input device found ({exc})",
                      "Connect a microphone, or use the app in text-only mode.")
    seconds = 4.0 if interactive else 1.0
    if interactive:
        print("\n    Microphone test: please say a short sentence now (4 seconds) ...", flush=True)
    try:
        audio = sd.rec(int(16_000 * seconds), samplerate=16_000, channels=1, dtype="float32")
        sd.wait()
    except Exception as exc:  # noqa: BLE001
        return Result("Microphone", FAIL, f"could not record from {device['name']}: {exc}",
                      "Check Windows Settings > Privacy > Microphone (allow desktop apps).")
    audio = audio[:, 0]
    level = float(np.sqrt(np.mean(audio**2)))
    name = device["name"]
    if not interactive:
        return Result("Microphone", PASS if level > 0 else WARN,
                      f"{name}: recording works (level {level:.4f}; speech not tested)")
    if level < 0.003:
        return Result("Microphone", WARN, f"{name}: almost silent (level {level:.4f})",
                      "Check that the microphone is not muted and is the Windows default input.")
    heard = stt.transcribe(audio).text.strip()
    if not heard:
        return Result("Microphone", WARN, f"{name}: sound recorded but no speech recognized", "Speak closer to the mic.")
    return Result("Microphone", PASS, f"{name}: heard \"{heard}\"")


# ---------------------------------------------------------------------------- main

def print_summary(results: list[Result]) -> None:
    colors = {PASS: "\x1b[32m", WARN: "\x1b[33m", FAIL: "\x1b[31m"}
    print("\n    Self-test summary")
    print("    " + "-" * 60)
    for r in results:
        print(f"    {colors[r.status]}{r.status}\x1b[0m  {r.name:<16} {r.detail}")
        if r.hint and r.status != PASS:
            print(f"          {'':<16} -> {r.hint}")
    print("    " + "-" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check that Local Lantern Talk is installed correctly.")
    parser.add_argument("--no-mic", action="store_true", help="skip the interactive microphone test")
    parser.add_argument("--no-sound", action="store_true", help="do not play the test sentence")
    args = parser.parse_args(argv)
    paths = get_paths()
    apply_portable_env(paths)
    setup_logging(paths.logs, "selftest")
    os.system("")  # ANSI colors in the classic console
    settings = SettingsStore(paths.settings_file).settings
    profile = effective_profile(asdict(settings))

    results: list[Result] = [check_gpu(), check_torch_cuda(), check_ctranslate2(profile.whisper_device),
                             check_llama(paths), check_ffmpeg(paths), *check_files(paths)]
    if not any(r.status == FAIL for r in results if r.name in ("Whisper model", "OmniVoice model")):
        from .stt import WhisperSTT

        stt = WhisperSTT(paths.whisper_models / "large-v3-turbo",
                         device=profile.whisper_device, compute_type=profile.whisper_compute)
        t0 = time.perf_counter()
        try:
            stt.load()
            results.append(Result("Whisper", PASS, f"loaded in {time.perf_counter() - t0:.1f} s "
                                                   f"({profile.whisper_device}, {profile.whisper_compute})"))
            results.append(check_tts(paths, settings, stt, play=not args.no_sound))
            results.append(check_mic(stt, interactive=not args.no_mic))
        except Exception as exc:  # noqa: BLE001
            log.exception("Whisper failed")
            results.append(Result("Whisper", FAIL, f"could not load: {exc}", "See logs\\selftest.log."))
    print_summary(results)
    failed = [r for r in results if r.status == FAIL]
    log.info("Self-test: %s", [(r.name, r.status, r.detail) for r in results])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
