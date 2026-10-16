"""Smart Turn on a real clip: finished at the end, unfinished in the middle (needs the installed model)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lantern_talk.resample import resample
from lantern_talk.turn import MODEL_FILE, SmartTurn

ROOT = Path(__file__).parents[1]
MODEL = ROOT / "models" / "vad" / MODEL_FILE

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="Smart Turn model not installed (install.bat)")


def test_sounds_finished_only_at_the_end() -> None:
    audio, sr = sf.read(str(ROOT / "config" / "builtin_voices" / "de_male.flac"), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    speech = resample(audio, sr, 16_000)
    turn = SmartTurn(MODEL)
    assert turn.probability(speech) > 0.9                          # the whole sentence
    middle = speech[: int(2.8 * 16_000)]                            # cut mid-sentence
    assert turn.probability(middle) < 0.2
    assert not turn.is_complete(np.concatenate([middle, np.zeros(4800, np.float32)]))  # a pause does not finish it
