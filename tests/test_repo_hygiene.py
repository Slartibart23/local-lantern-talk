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
# The only audio in the repository: the shipped built-in voice clips (spec §7.2, owner's choice).
ALLOWED = re.compile(r"^config/builtin_voices/[a-z]{2}_(female|male)\.flac$")


def tracked_files() -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return out.stdout.splitlines()


def test_no_personal_data_or_binaries_tracked() -> None:
    offending = [f for f in tracked_files() if FORBIDDEN.search(f) and not ALLOWED.match(f)]
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


@pytest.mark.skipif(not Path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe").exists(),
                    reason="Windows PowerShell not available")
def test_installer_script_parses_in_windows_powershell() -> None:
    """install.bat runs the script with Windows PowerShell 5.1; a syntax error would break every install."""
    script = ROOT / "scripts" / "install.ps1"
    check = (
        "$errors = $null; "
        f"[void][System.Management.Automation.Language.Parser]::ParseFile('{script}', [ref]$null, [ref]$errors); "
        "if ($errors) { $errors | ForEach-Object { $_.ToString() }; exit 1 }"
    )
    out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", check], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
