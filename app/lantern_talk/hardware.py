"""Hardware detection and hardware profiles (spec §5).

The profile is chosen from the GPU's VRAM and decides how Whisper runs and which
LLM quantization is the default. It is stored in ``user_data\\settings.json``
(``hardware_profile``); ``hardware_profile_override`` lets the user pick another.

Run ``python -m lantern_talk.hardware --write`` to detect and store the profile.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass

# Cards report a little less than their marketing size (an RTX 5090 Laptop GPU
# "24 GB" reports 24 463 MiB = 23.9 GiB, the desktop RTX 5090 32 607 MiB), so the
# limits sit just below it.
ULTRA_MIN_MIB = 30_000  # "32 GB+"
HIGH_MIN_MIB = 22_500   # "24 GB+"
MID_MIN_MIB = 15_000    # "16 GB"
LOW_MIN_MIB = 9_500     # "10 GB"


@dataclass(frozen=True)
class Profile:
    name: str                 # high | mid | low | unsupported
    description: str
    whisper_device: str       # cuda | cpu
    whisper_compute: str      # CTranslate2 compute type
    default_quant: str        # default LLM quantization (model catalog, M4)
    max_model_billions: int   # largest model enabled by default
    tts_precision: str
    tts_steps: int
    max_auto_context: int     # largest context length "Automatic" picks (it also has to fit)


PROFILES: dict[str, Profile] = {
    "ultra": Profile("ultra", "32 GB+ VRAM: best quantizations (Q6) and long contexts",
                     "cuda", "float16", "Q6_K", 32, "fp16", 32, 32768),
    "high": Profile("high", "24-31 GB VRAM: all catalog models", "cuda", "float16", "Q5_K_M", 32, "fp16", 32, 32768),
    "mid": Profile("mid", "16-23 GB VRAM: 24-30B models may need reduced context",
                   "cuda", "int8_float16", "Q4_K_M", 30, "fp16", 32, 16384),
    "low": Profile("low", "10-15 GB VRAM: models up to 12B", "cuda", "int8_float16", "Q4_K_M", 12, "fp16", 32, 8192),
    "unsupported": Profile("unsupported", "Less than 10 GB VRAM: best effort, text-only recommended",
                           "cpu", "int8", "Q4_K_M", 8, "fp16", 16, 4096),
}


def select_profile(vram_mib: int | None) -> Profile:
    """Profile for a GPU with ``vram_mib`` MiB of VRAM (``None`` = no NVIDIA GPU)."""
    if vram_mib is None or vram_mib < LOW_MIN_MIB:
        return PROFILES["unsupported"]
    if vram_mib >= ULTRA_MIN_MIB:
        return PROFILES["ultra"]
    if vram_mib >= HIGH_MIN_MIB:
        return PROFILES["high"]
    if vram_mib >= MID_MIN_MIB:
        return PROFILES["mid"]
    return PROFILES["low"]


def effective_profile(settings: dict) -> Profile:
    """The user's override if set and valid, else the detected profile."""
    for key in ("hardware_profile_override", "hardware_profile"):
        name = settings.get(key) or ""
        if name in PROFILES:
            return PROFILES[name]
    return PROFILES["high"]


@dataclass(frozen=True)
class GpuInfo:
    name: str
    vram_mib: int
    driver: str


def detect_gpu() -> GpuInfo | None:
    """First NVIDIA GPU via nvidia-smi, or ``None``."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = [line for line in out.stdout.splitlines() if line.strip()]
    if out.returncode != 0 or not lines:
        return None
    name, vram, driver = (p.strip() for p in lines[0].split(","))
    return GpuInfo(name=name, vram_mib=int(float(vram)), driver=driver)


def write_profile(settings_file, gpu: GpuInfo | None, profile: Profile) -> None:  # noqa: ANN001
    """Store the detected hardware in settings.json, keeping every other setting."""
    from .settings import SettingsStore

    store = SettingsStore(settings_file)
    s = store.settings
    first_run = not settings_file.exists()
    s.hardware_profile = profile.name
    s.gpu_name = gpu.name if gpu else ""
    s.gpu_vram_mib = gpu.vram_mib if gpu else 0
    if first_run:
        # Profile defaults only on a fresh install; never overwrite the user's choices.
        s.tts_precision = profile.tts_precision
        s.num_step = profile.tts_steps
    store.save()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect the GPU and choose a hardware profile.")
    parser.add_argument("--write", action="store_true", help="store the result in user_data\\settings.json")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args(argv)
    gpu = detect_gpu()
    profile = select_profile(gpu.vram_mib if gpu else None)
    if args.write:
        from .paths import get_paths

        write_profile(get_paths().settings_file, gpu, profile)
    if args.json:
        print(json.dumps({"gpu": asdict(gpu) if gpu else None, "profile": asdict(profile)}))
    else:
        gpu_text = f"{gpu.name}, {gpu.vram_mib / 1024:.1f} GB VRAM, driver {gpu.driver}" if gpu else "no NVIDIA GPU"
        print(f"    {gpu_text}")
        print(f"    Hardware profile: {profile.name} ({profile.description})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
