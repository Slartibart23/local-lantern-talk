"""GGUF header reader, tested on a synthetic file."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from lantern_talk.gguf_meta import GGUFError, model_info, read_metadata


def _string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv_string(key: str, value: str) -> bytes:
    return _string(key) + struct.pack("<I", 8) + _string(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _string(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_long_array(key: str, n: int) -> bytes:
    # array of n strings (like a tokenizer vocabulary) - must be skipped, not stored
    body = b"".join(_string(f"tok{i}") for i in range(n))
    return _string(key) + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", n) + body


def write_gguf(path: Path, kvs: list[bytes]) -> None:
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs))
    path.write_bytes(header + b"".join(kvs) + b"\x00" * 64)


def test_reads_model_info(tmp_path: Path) -> None:
    f = tmp_path / "test.gguf"
    write_gguf(f, [
        _kv_string("general.architecture", "gemma4"),
        _kv_string("general.name", "Test Model"),
        _kv_u32("gemma4.block_count", 48),
        _kv_u32("gemma4.context_length", 262144),
        _kv_long_array("tokenizer.ggml.tokens", 5000),
        _kv_string("tokenizer.chat_template", "{% if enable_thinking %}...{% endif %}"),
    ])
    info = model_info(f)
    assert (info.architecture, info.name, info.layers, info.context_length) == ("gemma4", "Test Model", 48, 262144)
    assert info.has_thinking
    assert read_metadata(f)["tokenizer.ggml.tokens"] is None  # long arrays are skipped


def test_rejects_non_gguf(tmp_path: Path) -> None:
    f = tmp_path / "fake.gguf"
    f.write_bytes(b"not a model")
    with pytest.raises(GGUFError):
        model_info(f)


def test_truncated_file(tmp_path: Path) -> None:
    f = tmp_path / "cut.gguf"
    f.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 5) + b"\x01")
    with pytest.raises(GGUFError):
        read_metadata(f)
