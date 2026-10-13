"""LLM access: manages ``llama-server.exe`` and streams chat completions from it."""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .procutil import tie_to_this_process

log = logging.getLogger(__name__)

HOST = "127.0.0.1"

# No "thinking" before answering: it costs seconds per reply, and some models (e.g. Qwen3)
# would otherwise write their reasoning into the reply. enable_thinking is the switch the
# Qwen / Gemma 4 chat templates use; templates without it ignore the variable.
NO_THINKING_ARGS = ["--reasoning-budget", "0", "--chat-template-kwargs", '{"enable_thinking": false}']


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return int(s.getsockname()[1])


@dataclass
class ServerConfig:
    model: Path
    context: int = 8192
    gpu_layers: str = "all"          # "all", "auto" or a number (partial CPU offload)
    flash_attn: str = "auto"
    extra_args: list[str] = field(default_factory=list)


class LlamaServer:
    """Runs ``llama-server.exe`` as a child process bound to 127.0.0.1."""

    def __init__(self, exe: Path, log_dir: Path, env: dict[str, str] | None = None) -> None:
        self.exe = exe
        self.log_dir = log_dir
        self.env = env
        self.port = 0
        self.config: ServerConfig | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._log_file = None

    @property
    def base_url(self) -> str:
        return f"http://{HOST}:{self.port}"

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, config: ServerConfig, timeout_s: float = 300.0) -> None:
        """Start (or restart) the server with ``config`` and wait until it is ready."""
        if not self.exe.exists():
            raise FileNotFoundError(f"llama-server.exe not found at {self.exe}. Run install.bat.")
        if not config.model.exists():
            raise FileNotFoundError(f"Model file not found: {config.model}")
        self.stop()
        self.port = _free_port()
        self.config = config
        args = [
            str(self.exe),
            "--model", str(config.model),
            "--host", HOST,
            "--port", str(self.port),
            "--ctx-size", str(config.context),
            "--n-gpu-layers", str(config.gpu_layers),
            "--flash-attn", config.flash_attn,
            "--parallel", "1",
            "--no-webui",
            "--jinja",
            *config.extra_args,
        ]
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_dir / "llama-server.log", "ab")  # noqa: SIM115 - kept open while running
        self._log_file.write(f"\n===== {time.ctime()} {' '.join(args)}\n".encode())
        self._log_file.flush()
        log.info("Starting llama-server on port %d with %s", self.port, config.model.name)
        self._proc = subprocess.Popen(
            args,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            env=self.env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if not tie_to_this_process(self._proc):
            log.warning("llama-server is not tied to this process; it may outlive a crash.")
        self._wait_ready(timeout_s)

    def _wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._proc is None or self._proc.poll() is not None:
                raise RuntimeError(
                    "llama-server exited while loading the model "
                    f"(see {self.log_dir / 'llama-server.log'}). The model may not fit in VRAM."
                )
            try:
                r = httpx.get(f"{self.base_url}/health", timeout=2.0)
                if r.status_code == 200:
                    log.info("llama-server ready")
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        self.stop()
        raise TimeoutError("llama-server did not become ready in time.")

    def stop(self) -> None:
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            self._proc = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class SamplingConfig:
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 600


class LLMClient:
    """Streams chat completions from llama-server's OpenAI-compatible API."""

    def __init__(self, server: LlamaServer, sampling: SamplingConfig | None = None) -> None:
        self.server = server
        self.sampling = sampling or SamplingConfig()

    def stream_chat(self, messages: list[ChatMessage]) -> Iterator[str]:
        """Yield content deltas as they arrive."""
        payload = {
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            "temperature": self.sampling.temperature,
            "top_p": self.sampling.top_p,
            "max_tokens": self.sampling.max_tokens,
        }
        url = f"{self.server.base_url}/v1/chat/completions"
        with httpx.stream("POST", url, json=payload, timeout=httpx.Timeout(300.0, connect=5.0)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    yield delta

    def complete(self, messages: list[ChatMessage]) -> str:
        return "".join(self.stream_chat(messages))
