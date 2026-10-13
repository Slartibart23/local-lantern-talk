"""User-facing error messages (spec §15.3): clear advice in the UI, details in logs\\."""

from __future__ import annotations

LOG_HINT = "Details are in logs\\server.log."

GPU_MEMORY = ("The GPU ran out of memory. Pick a smaller model or quantization (Change model…), "
              "lower the context length in Settings, or turn voice output off to free memory.")
LLM_CONNECTION = "The connection to the language model was lost. It is restarted with your next message."


def is_out_of_memory(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "out of memory" in text or "outofmemory" in text or "cudaerrormemoryallocation" in text


def is_connection_error(exc: BaseException) -> bool:
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx is a dependency
        return False
    return isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.ReadTimeout))


def describe(exc: BaseException, what: str = "") -> str:
    """A message for the UI. ``what`` names the failing part ("Speech synthesis")."""
    if is_out_of_memory(exc):
        return GPU_MEMORY
    if is_connection_error(exc):
        return LLM_CONNECTION
    prefix = f"{what} failed" if what else "Something went wrong"
    detail = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else type(exc).__name__
    return f"{prefix} ({detail}). {LOG_HINT}"
