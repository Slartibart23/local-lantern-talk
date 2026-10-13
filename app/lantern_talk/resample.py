"""Seamless streaming resampling for live microphone audio."""

from __future__ import annotations

from math import gcd

import numpy as np

from .audio_io import resample


class StreamingResampler:
    """Resamples a stream chunk by chunk with the same result as resampling it in one piece.

    Input is processed in blocks whose length is a whole number of resampling
    periods, each with extra context on both sides, so block boundaries are
    seamless (no clicks). Output lags the input by about ``block_ms + context``.
    """

    def __init__(self, orig_sr: int, target_sr: int, block_ms: int = 40, context_samples: int = 256) -> None:
        self.orig_sr = orig_sr
        self.target_sr = target_sr
        g = gcd(orig_sr, target_sr)
        self.in_unit = orig_sr // g      # input samples per resampling period
        self.out_unit = target_sr // g   # output samples per resampling period
        units = max(1, round(orig_sr * block_ms / 1000 / self.in_unit))
        self.block_in = units * self.in_unit
        self.block_out = units * self.out_unit
        ctx_units = -(-context_samples // self.in_unit)  # ceil
        self.ctx_in = ctx_units * self.in_unit
        self.ctx_out = ctx_units * self.out_unit
        self._buffer = np.zeros(self.ctx_in, dtype=np.float32)  # silence as left context
        self._pending_in = 0  # real input samples in the buffer not yet emitted

    def feed(self, samples: np.ndarray) -> np.ndarray:
        """Add input samples; returns all output that is final so far."""
        if self.orig_sr == self.target_sr:
            return samples.astype(np.float32, copy=False)
        self._buffer = np.concatenate([self._buffer, samples.astype(np.float32, copy=False)])
        self._pending_in += len(samples)
        out: list[np.ndarray] = []
        while len(self._buffer) >= self.ctx_in + self.block_in + self.ctx_in:
            out.append(self._process_block())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def flush(self) -> np.ndarray:
        """Emit the remaining samples (pads with silence) and reset."""
        if self.orig_sr == self.target_sr:
            return np.zeros(0, dtype=np.float32)
        remaining = self._pending_in
        expected = int(round(remaining * self.target_sr / self.orig_sr))
        pad = self.block_in + 2 * self.ctx_in
        self._buffer = np.concatenate([self._buffer, np.zeros(pad, dtype=np.float32)])
        out: list[np.ndarray] = []
        while len(self._buffer) >= self.ctx_in + self.block_in + self.ctx_in and sum(map(len, out)) < expected:
            out.append(self._process_block())
        result = np.concatenate(out)[:expected] if out else np.zeros(0, dtype=np.float32)
        self.reset()
        return result

    def reset(self) -> None:
        self._buffer = np.zeros(self.ctx_in, dtype=np.float32)
        self._pending_in = 0

    def _process_block(self) -> np.ndarray:
        segment = self._buffer[: self.ctx_in + self.block_in + self.ctx_in]
        converted = resample(segment, self.orig_sr, self.target_sr)
        block = converted[self.ctx_out: self.ctx_out + self.block_out]
        self._buffer = self._buffer[self.block_in:]
        self._pending_in = max(0, self._pending_in - self.block_in)
        return block
