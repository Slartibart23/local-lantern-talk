"""The streaming resampler must match one-piece resampling (no clicks at chunk borders)."""

from __future__ import annotations

import numpy as np
import pytest

from lantern_talk.audio_io import resample
from lantern_talk.resample import StreamingResampler


def signal(sr: int, seconds: float = 1.5) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    rng = np.random.default_rng(1)
    return (0.4 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 3100 * t)
            + 0.05 * rng.standard_normal(len(t))).astype(np.float32)


@pytest.mark.parametrize("orig_sr", [48_000, 44_100, 32_000, 16_000])
@pytest.mark.parametrize("chunk", [128, 441, 1000, 4800])
def test_streaming_equals_offline(orig_sr: int, chunk: int) -> None:
    x = signal(orig_sr)
    r = StreamingResampler(orig_sr, 16_000)
    parts = [r.feed(x[i: i + chunk]) for i in range(0, len(x), chunk)]
    parts.append(r.flush())
    streamed = np.concatenate(parts)
    offline = resample(x, orig_sr, 16_000)
    assert abs(len(streamed) - len(offline)) <= 1
    n = min(len(streamed), len(offline))
    # The very end differs slightly (offline sees no future input there either way).
    interior = slice(0, n - 64)
    assert np.max(np.abs(streamed[interior] - offline[interior])) < 1e-4


def test_output_is_produced_while_streaming() -> None:
    r = StreamingResampler(48_000, 16_000, block_ms=40)
    out = sum(len(r.feed(np.zeros(4800, dtype=np.float32))) for _ in range(10))  # 1 s
    assert out >= 16_000 * 0.85  # only the block + context latency is held back
