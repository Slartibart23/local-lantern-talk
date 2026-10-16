"""Downloads the core models into ``models\\`` (called by the installer).

Every download is resumable and skipped when the files already exist.
Run with: ``python -m lantern_talk.setup_models``
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from .paths import Paths, apply_portable_env, get_paths


def _load_pins(paths: Paths) -> dict:
    return json.loads((paths.config / "runtime_pins.json").read_text(encoding="utf-8"))


def _download_url(url: str, target: Path) -> None:
    if target.exists():
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


def fetch_smart_turn(paths: Paths, spec: dict) -> Path:
    """Smart Turn v3 (talk mode "Smart"), checked against its pinned SHA-256."""
    from huggingface_hub import hf_hub_download

    target = paths.vad_models / spec["file"]
    if target.exists():
        return target
    path = Path(hf_hub_download(repo_id=spec["repo"], filename=spec["file"], revision=spec["revision"],
                                local_dir=paths.vad_models))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec["sha256"]:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"{spec['file']} has the wrong checksum ({digest}).")
    return path


def download_smart_turn(paths: Paths, pins: dict) -> Path:
    print("  Smart Turn v3 (end-of-turn detection)")
    return fetch_smart_turn(paths, pins["models"]["smart_turn"])


def download_starter_llm(paths: Paths, pins: dict, quant: str) -> Path:
    """Optional starter model, offered by the installer until the model catalog (M4) exists."""
    from huggingface_hub import hf_hub_download

    spec = pins["starter_llm"]
    entry = spec["files"].get(quant) or next(iter(spec["files"].values()))
    print(f"  Starter language model {entry['file']} ({spec['repo']})")
    return Path(hf_hub_download(repo_id=spec["repo"], filename=entry["file"], local_dir=paths.llm_models))


def download_fonts(paths: Paths, pins: dict) -> None:
    """IBM Plex web fonts for the offline UI (SIL Open Font License)."""
    spec = pins["fonts"]
    paths.fonts.mkdir(parents=True, exist_ok=True)
    print(f"  Fonts (IBM Plex {spec['version']})")
    for f in spec["files"]:
        _download_url(f["url"], paths.fonts / f["file"])
    _download_url(spec["license_url"], paths.fonts / "OFL-LICENSE.txt")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Download the core models (resumable).")
    parser.add_argument("--starter-llm", metavar="QUANT",
                        help="also download the starter language model in this quantization (e.g. Q5_K_M)")
    args = parser.parse_args(argv)
    paths = get_paths()
    apply_portable_env(paths)
    pins = _load_pins(paths)
    try:
        download_fonts(paths, pins)
        download_vad(paths, pins)
        download_smart_turn(paths, pins)
        download_whisper(paths, pins)
        download_omnivoice(paths, pins)
        if args.starter_llm:
            download_starter_llm(paths, pins, args.starter_llm)
    except Exception as exc:  # noqa: BLE001 - shown to the user, details in the trace below
        print(f"\nDownload failed: {exc}\nRe-run install.bat to resume.", file=sys.stderr)
        raise
    print("  All core models are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
