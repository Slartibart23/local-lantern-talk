"""LLM model catalog (spec §13): shipped catalog, custom models and local GGUF files.

* ``config\\model_catalog.yaml`` — the shipped catalog (editable without code changes).
* ``user_data\\custom_models.json`` — models the user added (Hugging Face repo or URL)
  and variant labels for GGUF files copied into ``models\\llm`` by hand.

Everything here is pure logic (no GPU, no network) so it can be unit-tested; the
network part of adding a custom model lives in :mod:`lantern_talk.downloads`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import yaml

VARIANTS = ("censored", "uncensored", "unknown")
# Best quality first. Used to pick the largest quantization that fits.
QUANT_ORDER = ("Q8_0", "Q6_K_L", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "IQ4_XS", "Q3_K_M", "IQ3_M", "Q3_K_S")
QUANT_PATTERN = re.compile(r"(?<![A-Z0-9])(I?Q\d(?:_[A-Z0-9]+)+|BF16|F16|F32)(?=[.\-]|$)", re.IGNORECASE)
SPLIT_PATTERN = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)
HF_REPO_PATTERN = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")
# Partial CPU offload is offered when at least ~70 % of the model fits on the GPU.
PARTIAL_FACTOR = 1.4
OVERHEAD_GB = 0.5
DEFAULT_KV_GB_8K = 1.2


@dataclass(frozen=True)
class QuantFile:
    quant: str
    file: str               # path inside the repository (may contain folders)
    size_bytes: int
    min_vram_gb: float

    @property
    def local_name(self) -> str:
        return Path(self.file).name

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1e9


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    display_name: str
    family: str
    variant: str
    purpose: str
    hf_repo: str
    context_default: int
    license_note: str
    languages_note: str
    files: dict[str, QuantFile]
    note: str = ""
    source: str = "catalog"   # catalog | custom | local
    url: str = ""             # direct download URL (custom models from a URL)

    def quants_by_quality(self) -> list[QuantFile]:
        def rank(q: str) -> int:
            return QUANT_ORDER.index(q) if q in QUANT_ORDER else len(QUANT_ORDER)

        return sorted(self.files.values(), key=lambda f: (rank(f.quant), -f.size_bytes))


# ---------------------------------------------------------------------------- loading

def estimate_min_vram_gb(size_bytes: int, kv_gb: float = DEFAULT_KV_GB_8K) -> float:
    """File size + KV cache (8k context) + compute buffers, rounded up to 0.5 GB."""
    import math

    return math.ceil((size_bytes / 1e9 + kv_gb + OVERHEAD_GB) * 2) / 2


def _entry_from_dict(d: dict[str, Any], source: str) -> CatalogEntry:
    try:
        files = {
            quant: QuantFile(quant=quant, file=str(f["file"]), size_bytes=int(f["size_bytes"]),
                             min_vram_gb=float(f.get("min_vram_gb") or estimate_min_vram_gb(int(f["size_bytes"]))))
            for quant, f in (d.get("files") or {}).items()
        }
        variant = str(d.get("variant", "unknown"))
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if not files:
            raise ValueError("no files")
        return CatalogEntry(
            id=str(d["id"]), display_name=str(d.get("display_name") or d["id"]), family=str(d.get("family", "")),
            variant=variant, purpose=str(d.get("purpose", "")), hf_repo=str(d.get("hf_repo", "")),
            context_default=int(d.get("context_default", 8192)), license_note=str(d.get("license_note", "")),
            languages_note=str(d.get("languages_note", "")), note=str(d.get("note", "")),
            files=files, source=source, url=str(d.get("url", "")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid model entry {d.get('id', '?')!r}: {exc}") from exc


def load_catalog(path: Path) -> list[CatalogEntry]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = [_entry_from_dict(d, "catalog") for d in data.get("models") or []]
    ids = [e.id for e in entries]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate model ids in {path.name}: {sorted(duplicates)}")
    return entries


class CustomModels:
    """User-added models and labels, stored in ``user_data\\custom_models.json``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.models: list[dict[str, Any]] = []
        self.labels: dict[str, str] = {}   # GGUF file name -> variant
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.models = list(data.get("models") or [])
                self.labels = {k: v for k, v in (data.get("labels") or {}).items() if v in VARIANTS}
            except (OSError, ValueError):
                pass  # a broken file must not stop the app; it is rewritten on the next change

    def entries(self) -> list[CatalogEntry]:
        out = []
        for d in self.models:
            try:
                out.append(_entry_from_dict(d, "custom"))
            except ValueError:
                continue
        return out

    def add(self, entry: dict[str, Any]) -> None:
        self.models = [m for m in self.models if m.get("id") != entry["id"]] + [entry]
        self.save()

    def remove(self, model_id: str) -> None:
        self.models = [m for m in self.models if m.get("id") != model_id]
        self.save()

    def set_label(self, file_name: str, variant: str) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.labels[file_name] = variant
        for m in self.models:  # keep custom entries consistent with their label
            if any(Path(f["file"]).name == file_name for f in (m.get("files") or {}).values()):
                m["variant"] = variant
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"models": self.models, "labels": self.labels}, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)


# ---------------------------------------------------------------------------- custom input

def quant_of(file_name: str) -> str:
    m = QUANT_PATTERN.search(Path(file_name).stem.upper())
    return m.group(1).upper() if m else "?"


@dataclass(frozen=True)
class CustomSource:
    kind: str            # "repo" | "url"
    repo: str = ""
    file: str = ""       # file inside the repo (if the user named one)
    url: str = ""


def parse_custom_source(text: str) -> CustomSource:
    """Understands ``owner/name``, Hugging Face page/file URLs and direct GGUF URLs."""
    text = text.strip()
    if not text:
        raise ValueError("Enter a Hugging Face repo id (owner/name) or a link to a .gguf file.")
    if HF_REPO_PATTERN.match(text):
        return CustomSource("repo", repo=text)
    parsed = urlparse(text)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Not a Hugging Face repo id (owner/name) or an http(s) link.")
    parts = [unquote(p) for p in parsed.path.strip("/").split("/") if p]
    if parsed.netloc.endswith("huggingface.co") and len(parts) >= 2:
        repo = f"{parts[0]}/{parts[1]}"
        # https://huggingface.co/<repo>/(resolve|blob)/<revision>/<path to file>
        if len(parts) >= 5 and parts[2] in ("resolve", "blob"):
            file = "/".join(parts[4:])
            if not file.lower().endswith(".gguf"):
                raise ValueError("The link does not point to a .gguf file.")
            return CustomSource("repo", repo=repo, file=file)
        return CustomSource("repo", repo=repo)
    if not parsed.path.lower().endswith(".gguf"):
        raise ValueError("Direct links must point to a .gguf file.")
    return CustomSource("url", url=text, file=parts[-1])


def pick_repo_file(files: dict[str, int], quant: str | None, preferred: str) -> tuple[str, int]:
    """Choose one GGUF from a repository listing ``{path: size}``.

    ``quant`` is the user's explicit choice; otherwise the ``preferred`` quantization
    (from the hardware profile) or the closest smaller one is used. Split GGUFs and
    helper files (mmproj, MTP drafts) are ignored.
    """
    candidates = {
        p: s for p, s in files.items()
        if p.lower().endswith(".gguf") and not SPLIT_PATTERN.search(p)
        and not Path(p).name.lower().startswith(("mmproj", "mtp-"))
    }
    if not candidates:
        raise ValueError("No single-file GGUF found in this repository (split GGUFs are not supported).")
    by_quant: dict[str, tuple[str, int]] = {}
    for p, s in candidates.items():
        q = quant_of(p)
        if q not in by_quant or s > by_quant[q][1]:
            by_quant[q] = (p, s)
    if quant:
        if quant.upper() not in by_quant:
            raise ValueError(f"No {quant.upper()} file in this repository. Available: {', '.join(sorted(by_quant))}.")
        return by_quant[quant.upper()]
    order = list(QUANT_ORDER)
    start = order.index(preferred) if preferred in order else order.index("Q4_K_M")
    for q in order[start:] + order[:start][::-1]:
        if q in by_quant:
            return by_quant[q]
    return max(by_quant.values(), key=lambda ps: ps[1])  # unknown naming: take the largest


def custom_entry(source: CustomSource, file: str, size_bytes: int, variant: str) -> dict[str, Any]:
    """Catalog-style dict for a user-added model (stored in custom_models.json)."""
    name = Path(file).name
    q = quant_of(name)
    stem = re.sub(r"[-._]" + re.escape(q) + r"$", "", Path(name).stem, flags=re.IGNORECASE) if q != "?" else Path(name).stem
    return {
        "id": "custom-" + re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower()).strip("-"),
        "display_name": stem.replace("_", " "),
        "family": "",
        "variant": variant if variant in VARIANTS else "unknown",
        "purpose": "Your model",
        "hf_repo": source.repo,
        "url": source.url,
        "context_default": 8192,
        "license_note": "Check the license on the model page.",
        "languages_note": "",
        "files": {q: {"file": file, "size_bytes": size_bytes}},
    }


# ---------------------------------------------------------------------------- local files

def local_gguf_files(llm_dir: Path) -> dict[str, int]:
    """``{file name: size}`` of complete GGUF files in models\\llm (downloads in progress excluded)."""
    if not llm_dir.is_dir():
        return {}
    return {f.name: f.stat().st_size for f in llm_dir.glob("*.gguf") if f.is_file()}


def local_only_entries(llm_dir: Path, known: list[CatalogEntry], labels: dict[str, str]) -> list[CatalogEntry]:
    """GGUF files copied in by hand that belong to no catalog / custom entry (spec §13.4)."""
    known_files = {qf.local_name for e in known for qf in e.files.values()}
    out = []
    for name, size in sorted(local_gguf_files(llm_dir).items()):
        if name in known_files:
            continue
        q = quant_of(name)
        out.append(CatalogEntry(
            id="local-" + re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower()).strip("-"),
            display_name=Path(name).stem, family="", variant=labels.get(name, "unknown"),
            purpose="Copied into models\\llm", hf_repo="", context_default=8192,
            license_note="", languages_note="",
            files={q: QuantFile(q, name, size, estimate_min_vram_gb(size))}, source="local",
        ))
    return out


# ---------------------------------------------------------------------------- VRAM fit

@dataclass(frozen=True)
class Fit:
    status: str          # fits | partial | too_big
    quant: str
    min_vram_gb: float
    available_gb: float


def classify(min_vram_gb: float, available_gb: float) -> str:
    if min_vram_gb <= available_gb:
        return "fits"
    if min_vram_gb <= available_gb * PARTIAL_FACTOR:
        return "partial"
    return "too_big"


def choose_quant(entry: CatalogEntry, preferred: str, available_gb: float,
                 downloaded: set[str] | None = None) -> Fit:
    """Pick the quantization to use (spec §13.3).

    Start at the hardware profile's preferred quantization and go down until one
    fits completely on the GPU. A quantization that is already downloaded wins
    over downloading another one, as long as it fits at least partly.
    """
    quants = entry.quants_by_quality()
    downloaded = downloaded or set()
    for qf in quants:
        if qf.local_name in downloaded and classify(qf.min_vram_gb, available_gb) != "too_big":
            return Fit(classify(qf.min_vram_gb, available_gb), qf.quant, qf.min_vram_gb, available_gb)
    order = list(QUANT_ORDER)
    pref_rank = order.index(preferred) if preferred in order else order.index("Q4_K_M")
    eligible = [qf for qf in quants if (order.index(qf.quant) if qf.quant in order else 99) >= pref_rank] or quants
    for qf in eligible:
        if classify(qf.min_vram_gb, available_gb) == "fits":
            return Fit("fits", qf.quant, qf.min_vram_gb, available_gb)
    smallest = min(quants, key=lambda f: f.min_vram_gb)
    return Fit(classify(smallest.min_vram_gb, available_gb), smallest.quant, smallest.min_vram_gb, available_gb)
