"""Audio file decoding and resampling helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample mono float32 audio (band-limited sinc via torchaudio)."""
    if orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    import torch
    import torchaudio

    tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    return torchaudio.functional.resample(tensor, orig_sr, target_sr).numpy()


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Decode MP3 / WAV / FLAC / OGG to mono float32 at ``sample_rate``."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return resample(data.mean(axis=1), int(sr), sample_rate)
