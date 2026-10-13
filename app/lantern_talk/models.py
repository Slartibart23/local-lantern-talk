"""Model management for the web app: catalog view, downloads, custom models (spec §13)."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .catalog import (
    VARIANTS, CatalogEntry, CustomModels, choose_quant, classify, custom_entry, load_catalog,
    local_gguf_files, local_only_entries, parse_custom_source, pick_repo_file,
)
from .downloads import DownloadError, Downloader, DownloadJob, hf_file_url, list_repo_files, remote_size
from .gguf_meta import GGUFError, model_info
from .paths import Paths

log = logging.getLogger(__name__)


class ModelError(Exception):
    """A model problem with a message meant for the user."""


class ModelManager:
    def __init__(self, paths: Paths, on_download: Callable[[DownloadJob], None]) -> None:
        self.paths = paths
        self.catalog = load_catalog(paths.config / "model_catalog.yaml")
        self.custom = CustomModels(paths.user_data / "custom_models.json")
        self.downloader = Downloader(on_download)

    # -------------------------------------------------------------- listing
    def entries(self) -> list[CatalogEntry]:
        known = self.catalog + self.custom.entries()
        return known + local_only_entries(self.paths.llm_models, known, self.custom.labels)

    def entry(self, entry_id: str) -> CatalogEntry:
        for e in self.entries():
            if e.id == entry_id:
                return e
        raise ModelError(f"Unknown model {entry_id}.")

    def entry_for_file(self, file_name: str) -> CatalogEntry | None:
        for e in self.entries():
            if any(qf.local_name == file_name for qf in e.files.values()):
                return e
        return None

    def downloaded_files(self) -> dict[str, int]:
        return local_gguf_files(self.paths.llm_models)

    def view(self, available_gb: float, preferred_quant: str, loaded_file: str | None) -> list[dict[str, Any]]:
        """Everything the catalog screen shows, one dict per model."""
        local = self.downloaded_files()
        job = self.downloader.job
        out = []
        for e in self.entries():
            downloaded = {qf.local_name for qf in e.files.values() if qf.local_name in local}
            fit = choose_quant(e, preferred_quant, available_gb, downloaded)
            quants = [{
                "quant": qf.quant, "file": qf.local_name, "size_gb": round(qf.size_gb, 1),
                "min_vram_gb": qf.min_vram_gb, "downloaded": qf.local_name in local,
                "fit": classify(qf.min_vram_gb, available_gb),
            } for qf in e.quants_by_quality()]
            downloading = bool(job and job.key == e.id and job.state in ("queued", "running"))
            loaded = bool(loaded_file and loaded_file in {qf.local_name for qf in e.files.values()})
            out.append({
                "id": e.id, "name": e.display_name, "family": e.family, "variant": e.variant,
                "purpose": e.purpose, "source": e.source, "note": e.note,
                "license_note": e.license_note, "languages_note": e.languages_note,
                "hf_repo": e.hf_repo, "quants": quants,
                "recommended": {"quant": fit.quant, "status": fit.status, "min_vram_gb": fit.min_vram_gb},
                "state": "loaded" if loaded else "downloading" if downloading else
                         "downloaded" if downloaded else "available",
                "downloaded_files": sorted(downloaded),
            })
        return out

    # -------------------------------------------------------------- downloads
    def start_download(self, entry_id: str, quant: str | None, available_gb: float, preferred_quant: str,
                       on_done: Callable[[DownloadJob], None]) -> DownloadJob:
        e = self.entry(entry_id)
        if e.source == "local":
            raise ModelError("This model was copied in by hand; there is nothing to download.")
        if quant is None:
            quant = choose_quant(e, preferred_quant, available_gb).quant
        if quant not in e.files:
            raise ModelError(f"{e.display_name} has no {quant} file.")
        qf = e.files[quant]
        url = e.url or hf_file_url(e.hf_repo, qf.file)
        try:
            return self.downloader.start(e.id, url, self.paths.llm_models / qf.local_name, qf.size_bytes, on_done)
        except DownloadError as exc:
            raise ModelError(str(exc)) from exc

    def cancel_download(self) -> None:
        self.downloader.cancel()

    # -------------------------------------------------------------- custom models
    def add_custom(self, text: str, quant: str | None, variant: str, preferred_quant: str) -> CatalogEntry:
        """Resolve a repo id / link to one GGUF file and store it as a custom model (network)."""
        if variant not in VARIANTS:
            variant = "unknown"
        try:
            source = parse_custom_source(text)
            if source.kind == "repo":
                files = list_repo_files(source.repo)
                if source.file:
                    if source.file not in files:
                        raise ModelError(f"{source.file} is not in {source.repo}.")
                    file, size = source.file, files[source.file]
                else:
                    file, size = pick_repo_file(files, quant, preferred_quant)
            else:
                file, size = source.file, remote_size(source.url)
                if not size:
                    raise ModelError("The server does not report the file size; this link cannot be used.")
        except (ValueError, DownloadError) as exc:
            raise ModelError(str(exc)) from exc
        if size < 100_000_000:
            raise ModelError(f"{Path(file).name} is only {size / 1e6:.0f} MB - that is not a language model.")
        entry = custom_entry(source, file, size, variant)
        self.custom.add(entry)
        return self.entry(entry["id"])

    def remove_custom(self, entry_id: str) -> None:
        self.custom.remove(entry_id)

    def set_label(self, file_name: str, variant: str) -> None:
        try:
            self.custom.set_label(file_name, variant)
        except ValueError as exc:
            raise ModelError(str(exc)) from exc

    # -------------------------------------------------------------- files
    def delete_file(self, file_name: str) -> None:
        path = self.paths.llm_models / Path(file_name).name
        if path.suffix.lower() != ".gguf" or not path.exists():
            raise ModelError(f"{file_name} was not found in models\\llm.")
        path.unlink()
        part = path.with_name(path.name + ".part")
        part.unlink(missing_ok=True)

    def gpu_layers(self, file_name: str, context: int, margin_mib: int) -> tuple[int, int]:
        """``(layers on GPU, total layers)`` llama.cpp would use with the given free-memory margin.

        Uses llama-fit-params, which only reads the model header (about half a second).
        """
        path = self.paths.llm_models / file_name
        try:
            total = model_info(path).layers
        except (OSError, GGUFError):
            total = 0
        exe = self.paths.runtime / "llama.cpp" / "llama-fit-params.exe"
        try:
            out = subprocess.run(
                [str(exe), "-m", str(path), "-c", str(context), "--fit-target", str(margin_mib)],
                capture_output=True, text=True, timeout=60,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout
            tokens = out.split()
            ngl = int(tokens[tokens.index("-ngl") + 1])
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            log.warning("llama-fit-params failed for %s", file_name)
            return (total, total)  # unknown: assume it fits, llama.cpp adjusts anyway
        if ngl < 0 or (total and ngl >= total):
            return (total, total)
        return (ngl, total)
