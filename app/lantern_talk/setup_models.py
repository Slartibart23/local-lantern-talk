"""Downloads the core models into ``models\\`` (called by the installer).

Every download is resumable and skipped when the files already exist.
Run with: ``python -m lantern_talk.setup_models``
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from .paths import Paths, apply_portable_env, get_paths


def _load_pins(paths: Paths) -> dict:
    return json.loads((paths.config / "runtime_pins.json").read_text(encoding="utf-8"))


def _download_url(url: str, target: Path) -> None:
    if target.exists():
        print(f"    Already present: {target.name}")
        return
    partial = target.with_suffix(target.suffix + ".part")
    print(f"    Downloading {target.name} ...")
    urllib.request.urlretrieve(url, partial)
    partial.replace(target)


def download_whisper(paths: Paths, pins: dict, name: str = "large-v3-turbo") -> Path:
    from huggingface_hub import snapshot_download

    spec = pins["models"]["whisper"][name]
    target = paths.whisper_models / name
    print(f"  Whisper {name} ({spec['repo']})")
    snapshot_download(repo_id=spec["repo"], revision=spec.get("revision"), local_dir=target)
    return target


def download_omnivoice(paths: Paths, pins: dict) -> Path:
    from huggingface_hub import snapshot_download

    spec = pins["models"]["omnivoice"]
    print(f"  OmniVoice ({spec['repo']})")
    snapshot_download(repo_id=spec["repo"], revision=spec.get("revision"), local_dir=paths.omnivoice_model)
    return paths.omnivoice_model


def download_vad(paths: Paths, pins: dict) -> Path:
    spec = pins["models"]["vad"]
    target = paths.vad_models / spec["file"]
    print("  Silero VAD")
    _download_url(spec["url"], target)
    return target


def download_m1_llm(paths: Paths, pins: dict) -> Path:
    from huggingface_hub import hf_hub_download

    spec = pins["m1_llm"]
    print(f"  LLM {spec['file']} ({spec['repo']})")
    return Path(hf_hub_download(repo_id=spec["repo"], filename=spec["file"], local_dir=paths.llm_models))


def main() -> int:
    paths = get_paths()
    apply_portable_env(paths)
    pins = _load_pins(paths)
    try:
        download_vad(paths, pins)
        download_whisper(paths, pins)
        download_omnivoice(paths, pins)
        download_m1_llm(paths, pins)
    except Exception as exc:  # noqa: BLE001 - shown to the user, details in the trace below
        print(f"\nDownload failed: {exc}\nRe-run the installer to resume.", file=sys.stderr)
        raise
    print("  All core models are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
