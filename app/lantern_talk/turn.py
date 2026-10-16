"""Smart end-of-turn detection with Smart Turn v3 (pipecat-ai, BSD-2-Clause).

Silence alone cannot tell "I'm done" from "I'm thinking". Smart Turn looks at the
last 8 seconds of speech (intonation, rhythm, filler sounds) and says whether the
speaker sounds finished. The model is small (8 MB, int8 ONNX) and runs on the CPU
in about 15 ms, so it is asked every time the speaker pauses.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MODEL_FILE = "smart-turn-v3.2-cpu.onnx"
WINDOW_SAMPLES = 8 * 16_000   # the model sees the last 8 s at 16 kHz
COMPLETE_THRESHOLD = 0.5


class SmartTurn:
    def __init__(self, model_file: Path) -> None:
        self.model_file = model_file
        self._session = None
        self._features = None

    @property
    def available(self) -> bool:
        return self.model_file.exists()

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def load(self) -> None:
        if self._session is not None:
            return
        if not self.model_file.exists():
            raise FileNotFoundError(f"Smart Turn model not found: {self.model_file}")
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 2
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(self.model_file), sess_options=options,
                                             providers=["CPUExecutionProvider"])
        self._features = WhisperFeatureExtractor(chunk_length=8)
        log.info("Smart Turn loaded")

    def unload(self) -> None:
        self._session = None
        self._features = None

    def probability(self, audio: np.ndarray) -> float:
        """How likely the speaker has finished, for 16 kHz mono audio ending now."""
        self.load()
        assert self._session is not None and self._features is not None
        audio = audio[-WINDOW_SAMPLES:].astype(np.float32)
        audio = np.pad(audio, (WINDOW_SAMPLES - len(audio), 0))  # silence goes in front, as in training
        features = self._features(audio, sampling_rate=16_000, return_tensors="np", padding="max_length",
                                  max_length=WINDOW_SAMPLES, truncation=True, do_normalize=True)
        out = self._session.run(None, {"input_features": features.input_features.astype(np.float32)})
        return float(out[0][0].item())

    def is_complete(self, audio: np.ndarray) -> bool:
        return self.probability(audio) >= COMPLETE_THRESHOLD
