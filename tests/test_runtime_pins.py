"""Consistency checks for config/runtime_pins.json and the documented driver requirements."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PINS = json.loads((ROOT / "config" / "runtime_pins.json").read_text(encoding="utf-8"))
BUILDS = PINS["llama_cpp"]["builds"]


def ver(text: str) -> tuple[int, ...]:
    return tuple(int(p) for p in text.split("."))


def select(driver: str, builds: list[dict] = BUILDS) -> str | None:
    """Python mirror of Select-LlamaBuild in scripts/install.ps1 (driver part)."""
    for b in builds:
        if ver(driver) >= ver(b["min_driver"]):
            return b["name"]
    return None


def test_cuda13_build_needs_driver_581() -> None:
    assert BUILDS[0]["name"].startswith("cuda-13")
    assert BUILDS[0]["min_driver"] == "581.0"


@pytest.mark.parametrize(
    "driver, expected",
    [("570.65", "cuda-12.4"), ("576.88", "cuda-12.4"), ("580.97", "cuda-12.4"),
     ("581.08", "cuda-13.4"), ("616.92", "cuda-13.4")],
)
def test_build_selection(driver: str, expected: str) -> None:
    assert select(driver) == expected


def test_every_supported_driver_gets_a_build() -> None:
    assert ver(BUILDS[-1]["min_driver"]) <= ver(PINS["nvidia"]["min_driver"])
    assert select(PINS["nvidia"]["min_driver"]) is not None


def test_builds_ordered_by_preference() -> None:
    drivers = [ver(b["min_driver"]) for b in BUILDS]
    assert drivers == sorted(drivers, reverse=True)


@pytest.mark.parametrize("build", BUILDS, ids=[b["name"] for b in BUILDS])
def test_build_bundles_matching_cuda_runtime(build: dict) -> None:
    major = re.match(r"cuda-(\d+)\.", build["name"]).group(1)
    for lib in ("cudart64", "cublas64", "cublasLt64"):
        assert f"{lib}_{major}.dll" in build["required_dlls"]
    assert build["cudart"].startswith("cudart-") and f"cuda-{build['name'][5:]}" in build["cudart"]
    assert PINS["llama_cpp"]["tag"] in build["zip"]
    assert PINS["llama_cpp"]["base_url"].endswith(f"/{PINS['llama_cpp']['tag']}/")


def test_readme_states_the_driver_requirements() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert PINS["nvidia"]["min_driver"] in readme
    assert "581" in readme
