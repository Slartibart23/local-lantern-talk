"""Generates the 8 built-in voices with OmniVoice voice design (spec §7.2).

Run by the installer; also usable to regenerate them:
``python -m lantern_talk.setup_voices [--regenerate]``.
Every generated reference clip is checked with Whisper (voice design occasionally
produces noise instead of speech) and regenerated if it does not match its text.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import asdict

from .hardware import effective_profile
from .logging_setup import setup_logging
from .paths import apply_portable_env, get_paths
from .settings import SettingsStore
from .stt import WhisperSTT
from .tts import OmniVoiceTTS, VoiceLibrary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the built-in voices.")
    parser.add_argument("--regenerate", action="store_true", help="replace existing built-in voices")
    args = parser.parse_args(argv)

    paths = get_paths()
    apply_portable_env(paths)
    setup_logging(paths.logs, "setup_voices")
    settings = SettingsStore(paths.settings_file).settings
    profile = effective_profile(asdict(settings))

    stt = WhisperSTT(paths.whisper_models / "large-v3-turbo",
                     device=profile.whisper_device, compute_type=profile.whisper_compute)
    tts = OmniVoiceTTS(paths.omnivoice_model, precision=settings.tts_precision)
    voices = VoiceLibrary(tts, paths.config / "builtin_voices.yaml", paths.builtin_voices, paths.voice_cache,
                          transcribe=lambda audio, sr, lang: stt.transcribe_at(audio, sr, lang).text)

    todo = [v for v in voices.builtins.values()
            if args.regenerate or not (paths.builtin_voices / f"{v.id}.pt").exists()]
    if not todo:
        print("    All 8 built-in voices are present.")
        return 0
    print(f"    Generating {len(todo)} built-in voice(s) on the GPU ...")
    t0 = time.perf_counter()
    try:
        for v in todo:
            print(f"      {v.name} ...", flush=True)
            if args.regenerate:
                voices.regenerate_builtin(v.id)
            else:
                voices.prompt_for(f"builtin:{v.id}")
    except Exception as exc:  # noqa: BLE001 - short message; details in logs\setup_voices.log
        logging.getLogger(__name__).exception("Voice generation failed")
        print(f"    Voice generation failed: {exc}. Details: logs\\setup_voices.log", file=sys.stderr)
        return 1
    print(f"    Done in {time.perf_counter() - t0:.0f} s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
