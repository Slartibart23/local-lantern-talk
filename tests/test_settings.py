from __future__ import annotations

import json
from pathlib import Path

from lantern_talk.settings import Settings, SettingsStore


def test_defaults_when_missing(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    assert store.settings == Settings()


def test_roundtrip_and_unknown_keys_kept(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"voice": "builtin:de_male", "hardware_profile": "high"}))
    store = SettingsStore(path)
    assert store.settings.voice == "builtin:de_male"
    store.settings.update({"mic_enabled": False})
    store.save()
    data = json.loads(path.read_text())
    assert data["mic_enabled"] is False
    assert data["hardware_profile"] == "high"  # written by other parts of the app


def test_update_coerces_and_reports_changes() -> None:
    s = Settings()
    changed = s.update({"end_silence_ms": "900", "voice_output": 0, "nonsense": 1, "num_step": "abc"})
    assert changed == {"end_silence_ms": 900, "voice_output": False}
    assert s.num_step == 32


def test_corrupt_file_falls_back_to_defaults(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{not json")
    assert SettingsStore(path).settings == Settings()
