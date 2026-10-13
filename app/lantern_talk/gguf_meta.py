"""Minimal GGUF header reader (stdlib only).

Reads the key/value metadata at the start of a GGUF file — architecture, layer
count, context length, model name — without loading any tensors. Large arrays
(e.g. the tokenizer vocabulary) are skipped, so this is fast even for 20 GB files.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

GGUF_MAGIC = b"GGUF"

# GGUF value types
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _FLOAT32, _BOOL, _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = range(13)
_SCALARS = {
    _UINT8: "<B", _INT8: "<b", _UINT16: "<H", _INT16: "<h", _UINT32: "<I", _INT32: "<i",
    _FLOAT32: "<f", _BOOL: "<?", _UINT64: "<Q", _INT64: "<q", _FLOAT64: "<d",
}
MAX_ARRAY_ITEMS_KEPT = 64  # longer arrays (vocabularies) are skipped, not stored


class GGUFError(ValueError):
    pass


def _read(f: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise GGUFError("unexpected end of file")
    return struct.unpack(fmt, data)[0]


def _read_string(f: BinaryIO) -> str:
    length = _read(f, "<Q")
    if length > 50_000_000:
        raise GGUFError("implausible string length")
    return f.read(length).decode("utf-8", errors="replace")


def _skip_value(f: BinaryIO, vtype: int) -> None:
    if vtype in _SCALARS:
        f.seek(struct.calcsize(_SCALARS[vtype]), 1)
    elif vtype == _STRING:
        f.seek(_read(f, "<Q"), 1)
    else:
        raise GGUFError(f"cannot skip value type {vtype}")


def _read_value(f: BinaryIO, vtype: int) -> Any:
    if vtype in _SCALARS:
        return _read(f, _SCALARS[vtype])
    if vtype == _STRING:
        return _read_string(f)
    if vtype == _ARRAY:
        item_type = _read(f, "<I")
        count = _read(f, "<Q")
        if count > MAX_ARRAY_ITEMS_KEPT:
            if item_type in _SCALARS:
                f.seek(count * struct.calcsize(_SCALARS[item_type]), 1)
            else:
                for _ in range(count):
                    _skip_value(f, item_type)
            return None
        return [_read_value(f, item_type) for _ in range(count)]
    raise GGUFError(f"unknown value type {vtype}")


def read_metadata(path: Path) -> dict[str, Any]:
    """All header key/value pairs (long arrays are returned as ``None``)."""
    with open(path, "rb") as f:
        if f.read(4) != GGUF_MAGIC:
            raise GGUFError(f"{path.name} is not a GGUF file")
        version = _read(f, "<I")
        if version < 2:
            raise GGUFError(f"GGUF version {version} is not supported")
        _read(f, "<Q")  # tensor count
        kv_count = _read(f, "<Q")
        meta: dict[str, Any] = {"GGUF.version": version}
        for _ in range(kv_count):
            key = _read_string(f)
            vtype = _read(f, "<I")
            meta[key] = _read_value(f, vtype)
        return meta


@dataclass(frozen=True)
class ModelInfo:
    architecture: str
    name: str
    layers: int
    context_length: int
    file_size: int
    has_thinking: bool   # chat template contains a reasoning / thinking mode


def model_info(path: Path) -> ModelInfo:
    meta = read_metadata(path)
    arch = str(meta.get("general.architecture", "unknown"))
    template = str(meta.get("tokenizer.chat_template") or "")
    return ModelInfo(
        architecture=arch,
        name=str(meta.get("general.name") or path.stem),
        layers=int(meta.get(f"{arch}.block_count") or 0),
        context_length=int(meta.get(f"{arch}.context_length") or 0),
        file_size=path.stat().st_size,
        has_thinking=any(t in template for t in ("enable_thinking", "<think>", "thought", "[THINK]")),
    )
