"""User settings persisted in ``user_data\\settings.json``."""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class Settings:
    voice: str = "builtin:en_female"
    model: str = ""                 # file name in models\llm; "" = first file found
    mic_enabled: bool = True
    voice_output: bool = True
    match_language: bool = True
    show_actions: bool = True
    end_silence_ms: int = 700
    vad_threshold: float = 0.5
    num_step: int = 32              # OmniVoice diffusion steps (16 fast ... 64 best)
    tts_precision: str = "fp16"     # OmniVoice precision: fp16 | bf16 | fp32
    context: int = 8192
    hardware_profile: str = ""          # detected by the installer (high | mid | low | unsupported)
    hardware_profile_override: str = "" # user's choice; "" = use the detected profile
    gpu_name: str = ""
    gpu_vram_mib: int = 0

    def update(self, values: dict[str, Any]) -> dict[str, Any]:
        """Apply known keys with type coercion; returns the keys that changed."""
        changed: dict[str, Any] = {}
        for f in fields(self):
            if f.name not in values:
                continue
            current = getattr(self, f.name)
            value = values[f.name]
            try:
                value = type(current)(value) if not isinstance(current, bool) else bool(value)
            except (TypeError, ValueError):
                log.warning("Ignoring invalid value for setting %s: %r", f.name, value)
                continue
            if value != current:
                setattr(self, f.name, value)
                changed[f.name] = value
        return changed


class SettingsStore:
    """Loads and atomically saves :class:`Settings`; unknown keys in the file are kept."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._extra: dict[str, Any] = {}
        self.settings = self._load()

    def _load(self) -> Settings:
        settings = Settings()
        if not self.path.exists():
            return settings
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not read %s (%s); using defaults.", self.path, exc)
            return settings
        known = {f.name for f in fields(Settings)}
        self._extra = {k: v for k, v in data.items() if k not in known}
        settings.update(data)
        return settings

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({**self._extra, **asdict(self.settings)}, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
