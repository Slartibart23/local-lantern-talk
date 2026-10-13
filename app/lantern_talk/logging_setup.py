"""Logging: full details go to ``logs\\``; the console only shows short messages."""

from __future__ import annotations

import logging
import logging.handlers
import warnings
from pathlib import Path


def setup_logging(log_dir: Path, name: str, console_level: int = logging.WARNING) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)

    file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(console)

    for noisy in ("httpx", "httpcore", "urllib3", "faster_whisper", "transformers", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.captureWarnings(True)
    warnings.filterwarnings("ignore", category=FutureWarning)
    return log_file
