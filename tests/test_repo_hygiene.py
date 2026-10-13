"""Repository hygiene: personal data and binaries must never be tracked (spec §15.1)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
FORBIDDEN = re.compile(
    r"^(user_data|tmp|logs|models|runtime)/|\.(mp3|wav|flac|ogg|m4a|opus|pt|gguf|safetensors|bin)$",
    re.IGNORECASE,
)


def tracked_files() -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return out.stdout.splitlines()


def test_no_personal_data_or_binaries_tracked() -> None:
    offending = [f for f in tracked_files() if FORBIDDEN.search(f)]
    assert offending == [], f"must never be committed: {offending}"


def test_gitignore_covers_personal_folders() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for folder in ("/user_data/", "/models/", "/runtime/", "/logs/", "/tmp/"):
        assert folder in ignore


def test_pre_commit_hook_blocks_the_same_paths() -> None:
    hook = (ROOT / "scripts" / "git-hooks" / "pre-commit").read_text(encoding="utf-8")
    for token in ("user_data", "tmp", "logs", "models", "runtime", "mp3", "wav", "gguf"):
        assert token in hook


def test_python_sources_compile_without_warnings() -> None:
    """Catches invalid escape sequences, e.g. a Windows path written with single backslashes."""
    import warnings

    sources = [f for f in tracked_files() if f.endswith(".py")]
    for name in sources:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            compile((ROOT / name).read_text(encoding="utf-8"), name, "exec")
