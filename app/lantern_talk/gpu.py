"""CUDA helpers shared by the GPU components."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DLL_HANDLES: list[object] = []


def register_cuda_dlls() -> None:
    """Make PyTorch's bundled CUDA DLLs (cuBLAS, cuDNN) visible to CTranslate2.

    PyTorch cu128 wheels ship cuBLAS 12 and cuDNN 9 in ``torch\\lib``; CTranslate2
    (faster-whisper) needs the same libraries but does not bundle cuBLAS. Reusing
    them avoids a separate CUDA toolkit install.
    """
    if sys.platform != "win32" or _DLL_HANDLES:
        return
    import torch  # noqa: F401 - importing torch loads its own DLLs first

    torch_lib = Path(torch.__file__).parent / "lib"
    if torch_lib.is_dir():
        _DLL_HANDLES.append(os.add_dll_directory(str(torch_lib)))
        os.environ["PATH"] = f"{torch_lib}{os.pathsep}{os.environ.get('PATH', '')}"


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - any failure means "no CUDA"
        return False


def vram_used_by_process_mb() -> float:
    """VRAM currently allocated by PyTorch in this process, in MiB."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / 2**20
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def free_cuda_cache() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
