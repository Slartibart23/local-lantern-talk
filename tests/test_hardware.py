"""Hardware profile selection (spec §5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lantern_talk.hardware import PROFILES, GpuInfo, effective_profile, select_profile, write_profile


@pytest.mark.parametrize(
    "vram_mib, expected",
    [
        (32_607, "ultra"),      # RTX 5090
        (30_000, "ultra"),
        (24_564, "high"),       # RTX 4090
        (24_463, "high"),       # RTX 5090 Laptop GPU reports 23.9 GiB
        (16_376, "mid"),        # 16 GB cards (4080, 5080, 4060 Ti 16 GB)
        (20_480, "mid"),        # 20 GB
        (12_282, "low"),        # RTX 4070 / 3060 12 GB
        (10_240, "low"),        # RTX 3080 10 GB (spec minimum)
        (8_188, "unsupported"),
        (None, "unsupported"),  # no NVIDIA GPU
    ],
)
def test_select_profile(vram_mib: int | None, expected: str) -> None:
    assert select_profile(vram_mib).name == expected


def test_profiles_follow_spec_table() -> None:
    assert PROFILES["ultra"].default_quant == "Q6_K"
    assert PROFILES["high"].default_quant == "Q5_K_M"
    assert [p.max_auto_context for p in PROFILES.values()] == [32768, 32768, 16384, 8192, 4096]
    assert PROFILES["mid"].default_quant == PROFILES["low"].default_quant == "Q4_K_M"
    assert PROFILES["mid"].whisper_compute.startswith("int8")
    assert PROFILES["low"].max_model_billions == 12
    assert PROFILES["unsupported"].whisper_device == "cpu"


def test_override_wins() -> None:
    assert effective_profile({"hardware_profile": "high", "hardware_profile_override": "low"}).name == "low"
    assert effective_profile({"hardware_profile": "mid", "hardware_profile_override": ""}).name == "mid"
    assert effective_profile({"hardware_profile": "bogus"}).name == "high"


def test_write_profile_keeps_user_choices(tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    gpu = GpuInfo("NVIDIA GeForce RTX 3080", 10_240, "581.42")
    write_profile(settings_file, gpu, select_profile(gpu.vram_mib))
    data = json.loads(settings_file.read_text())
    assert data["hardware_profile"] == "low" and data["gpu_name"] == "NVIDIA GeForce RTX 3080"
    # A later re-run (e.g. new GPU) updates the hardware but keeps what the user chose.
    data["num_step"] = 64
    settings_file.write_text(json.dumps(data))
    write_profile(settings_file, GpuInfo("RTX 5090", 32_607, "616.92"), select_profile(32_607))
    data = json.loads(settings_file.read_text())
    assert data["hardware_profile"] == "ultra" and data["num_step"] == 64


def test_installer_uses_the_same_high_profile_threshold() -> None:
    from lantern_talk.hardware import HIGH_MIN_MIB

    script = (Path(__file__).parents[1] / "scripts" / "install.ps1").read_text(encoding="utf-8")
    assert f"$HighProfileMinMiB = {HIGH_MIN_MIB}" in script
