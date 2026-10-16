"""Path resolution and portable environment setup.

Every location is derived from the program folder, which is found relative to
this file. Nothing depends on the current working directory, the user profile
or the registry, so the whole folder can be moved to another drive.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Cache / temp variables every process we start must have (spec §3).
CACHE_ENV_VARS: tuple[str, ...] = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_XET_CACHE",
    "TORCH_HOME",
    "XDG_CACHE_HOME",
    "UV_CACHE_DIR",
    "UV_PYTHON_INSTALL_DIR",
    "PIP_CACHE_DIR",
    "NUMBA_CACHE_DIR",
    "MPLCONFIGDIR",
    "TRITON_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "GRADIO_TEMP_DIR",
    "TMP",
    "TEMP",
)

MAX_SAFE_PATH_LENGTH = 120


def find_root(start: Path | None = None) -> Path:
    """Return the program folder: the first ancestor containing ``app`` and ``config``."""
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "app").is_dir() and (candidate / "config").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate the program folder above {here}")


@dataclass(frozen=True)
class Paths:
    """All fixed locations inside the program folder (spec §3)."""

    root: Path

    @property
    def app(self) -> Path:
        return self.root / "app"

    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def runtime(self) -> Path:
        return self.root / "runtime"

    @property
    def cache(self) -> Path:
        return self.runtime / "cache"

    @property
    def tmp(self) -> Path:
        return self.root / "tmp"

    @property
    def ffmpeg(self) -> Path:
        return self.runtime / "ffmpeg" / "bin" / "ffmpeg.exe"

    @property
    def fonts(self) -> Path:
        return self.runtime / "fonts"

    @property
    def web(self) -> Path:
        return self.app / "web"

    @property
    def venv_python(self) -> Path:
        return self.runtime / "venv" / "Scripts" / "python.exe"

    @property
    def llama_server(self) -> Path:
        return self.runtime / "llama.cpp" / "llama-server.exe"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def llm_models(self) -> Path:
        return self.models / "llm"

    @property
    def whisper_models(self) -> Path:
        return self.models / "whisper"

    @property
    def omnivoice_model(self) -> Path:
        return self.models / "omnivoice"

    @property
    def vad_models(self) -> Path:
        return self.models / "vad"

    @property
    def builtin_voices(self) -> Path:
        return self.models / "builtin_voices"

    @property
    def user_data(self) -> Path:
        return self.root / "user_data"

    @property
    def voices(self) -> Path:
        return self.user_data / "voices"

    @property
    def voice_cache(self) -> Path:
        return self.user_data / "voices" / ".cache"

    @property
    def characters(self) -> Path:
        return self.user_data / "characters"

    @property
    def memory(self) -> Path:
        return self.user_data / "memory"

    @property
    def sessions(self) -> Path:
        return self.user_data / "sessions"

    @property
    def images(self) -> Path:
        return self.user_data / "images"

    @property
    def settings_file(self) -> Path:
        return self.user_data / "settings.json"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def cache_env(self) -> dict[str, str]:
        """Environment variables that redirect every known cache into the program folder."""
        c = self.cache
        return {
            "HF_HOME": str(c / "huggingface"),
            "HF_HUB_CACHE": str(c / "huggingface" / "hub"),
            "HF_XET_CACHE": str(c / "huggingface" / "xet"),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TORCH_HOME": str(c / "torch"),
            "XDG_CACHE_HOME": str(c),
            "UV_CACHE_DIR": str(c / "uv"),
            "UV_PYTHON_INSTALL_DIR": str(self.runtime / "python"),
            "PIP_CACHE_DIR": str(c / "pip"),
            "NUMBA_CACHE_DIR": str(c / "numba"),
            "MPLCONFIGDIR": str(c / "matplotlib"),
            "TRITON_CACHE_DIR": str(c / "triton"),
            "CUDA_CACHE_PATH": str(c / "nv_compute"),
            "CUDA_CACHE_MAXSIZE": str(4 * 2**30),  # keep all JIT-compiled kernels (default 256 MB)
            "GRADIO_TEMP_DIR": str(self.tmp / "gradio"),
            "GRADIO_ANALYTICS_ENABLED": "False",
            "TMP": str(self.tmp),
            "TEMP": str(self.tmp),
        }

    def ensure_dirs(self) -> None:
        """Create the writable folders the app needs (never touches existing content)."""
        for d in (
            self.cache, self.tmp, self.llm_models, self.whisper_models,
            self.omnivoice_model, self.vad_models, self.builtin_voices,
            self.voices, self.characters, self.memory, self.sessions, self.images, self.logs,
        ):
            d.mkdir(parents=True, exist_ok=True)


def path_warnings(root: Path) -> list[str]:
    """Return non-blocking warnings about the install path (spec §3)."""
    warnings: list[str] = []
    text = str(root)
    if not text.isascii():
        warnings.append(
            "The program folder path contains non-ASCII characters. Some components "
            "may fail; consider moving the folder to a plain ASCII path."
        )
    if len(text) > MAX_SAFE_PATH_LENGTH:
        warnings.append(
            f"The program folder path is very long ({len(text)} characters). Deeply "
            "nested model files may hit the Windows path length limit."
        )
    return warnings


def apply_portable_env(paths: Paths) -> None:
    """Force the cache variables for this process and every child process it starts."""
    for key, value in paths.cache_env().items():
        os.environ[key] = value
    paths.ensure_dirs()


_PATHS: Paths | None = None


def get_paths() -> Paths:
    """Process-wide :class:`Paths` instance."""
    global _PATHS
    if _PATHS is None:
        _PATHS = Paths(find_root())
    return _PATHS
