"""Model management for the web app: catalog view, downloads, custom models (spec §13)."""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import (
    VARIANTS, CatalogEntry, CustomModels, choose_quant, classify, custom_entry, load_catalog,
    local_gguf_files, local_only_entries, parse_custom_source, pick_repo_file,
)
from .downloads import (
    DownloadError, Downloader, DownloadJob, hf_file_url, list_repo_files, remote_size, search_gguf_repos,
)
from .gguf_meta import GGUFError, model_info
from .paths import Paths

log = logging.getLogger(__name__)


class ModelError(Exception):
    """A model problem with a message meant for the user."""


class NoGGUFError(ModelError):
    """The repository has no usable GGUF file; ``suggestions`` are GGUF repos of the same model."""

    def __init__(self, message: str, suggestions: list[str]) -> None:
        super().__init__(message)
        self.suggestions = suggestions


@dataclass(frozen=True)
class FitResult:
    args: list[str]        # llama-server arguments from llama-fit-params (without -c)
    gpu_layers: int        # layers placed on the GPU
    total_layers: int
    cpu_blocks: int        # blocks whose expert weights (MoE) stay on the CPU
    known: bool = True     # False if llama-fit-params could not run

    @property
    def partial(self) -> bool:
        return self.known and (self.cpu_blocks > 0 or (self.total_layers and self.gpu_layers < self.total_layers))

    @property
    def nothing_fits(self) -> bool:
        return self.known and self.total_layers > 0 and self.gpu_layers == 0


_LLAMA_FREE = re.compile(r"projected to use \d+ MiB of device memory vs\. (\d+) MiB of free device memory")


def parse_llama_free_mib(stderr: str) -> int | None:
    """Free device memory as llama.cpp sees it (from llama-fit-params' log)."""
    m = _LLAMA_FREE.search(stderr)
    return int(m.group(1)) if m else None


def parse_fit_output(stdout: str, total_layers: int) -> FitResult:
    """Parse the CLI arguments printed by llama-fit-params.

    Examples: ``-c 8192 -ngl -1`` (everything on the GPU) or, for mixture-of-experts
    models, ``-c 8192 -ngl 31 -ot "blk\\.24\\.ffn_...=CPU,blk\\.25\\..."`` (all layers on
    the GPU, but the expert weights of some blocks stay on the CPU).
    """
    line = next((ln for ln in reversed(stdout.splitlines()) if "-ngl" in ln), "")
    if not line:
        raise ValueError("no fitted arguments in llama-fit-params output")
    tokens = shlex.split(line, posix=True)
    args: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] in ("-c", "--ctx-size"):
            i += 2  # the app passes its own context size
            continue
        args.append(tokens[i])
        i += 1
    ngl = int(args[args.index("-ngl") + 1])
    gpu_layers = total_layers if ngl < 0 or (total_layers and ngl >= total_layers) else ngl
    cpu_blocks = 0
    if "-ot" in args:
        cpu_blocks = len(set(re.findall(r"blk\\?\.(\d+)\\?\.", args[args.index("-ot") + 1])))
    return FitResult(args=args, gpu_layers=gpu_layers, total_layers=total_layers, cpu_blocks=cpu_blocks)


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
                elif not any(p.lower().endswith(".gguf") for p in files):
                    raise self._no_gguf(source.repo, files)
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

    @staticmethod
    def _no_gguf(repo: str, files: dict[str, int]) -> NoGGUFError:
        """Explain why a repository cannot be used and suggest GGUF versions of the same model."""
        name = repo.split("/", 1)[1]
        suggestions = [s["id"] for s in search_gguf_repos(name) if s["id"] != repo]
        original = any(p.endswith((".safetensors", ".bin", ".pt")) for p in files)
        why = ("contains the original model weights (safetensors), not GGUF" if original
               else "contains no GGUF file")
        hint = ("GGUF versions of this model: " + ", ".join(suggestions) + "." if suggestions
                else "Search Hugging Face for a GGUF version of this model (often named …-GGUF).")
        return NoGGUFError(f"{repo} {why}. The app needs a GGUF file. {hint}", suggestions)

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

    def fit(self, file_name: str, context: int, margin_mib: int, true_free_mib: int | None = None) -> FitResult:
        """How llama.cpp should split the model between GPU and CPU, keeping ``margin_mib`` free.

        Runs llama-fit-params (reads only the model header, a few seconds). Its
        result is passed to llama-server unchanged, so what the UI reports is what
        happens.

        On Windows, CUDA's "free memory" ignores much of what other processes (this
        app's Whisper / OmniVoice) hold, so llama.cpp over-estimates free memory.
        ``true_free_mib`` (from nvidia-smi) corrects that: the margin is raised by
        the difference between llama.cpp's view and the real free memory.
        """
        path = self.paths.llm_models / file_name
        try:
            total = model_info(path).layers
        except (OSError, GGUFError):
            total = 0
        try:
            out, err = self._run_fit(path, context, margin_mib)
            llama_free = parse_llama_free_mib(err)
            if true_free_mib is not None and llama_free is not None and llama_free > true_free_mib + 256:
                corrected = margin_mib + (llama_free - true_free_mib)
                log.info("llama.cpp sees %d MiB free, nvidia-smi %d MiB: fitting %s with a %d MiB margin",
                         llama_free, true_free_mib, file_name, corrected)
                out, err = self._run_fit(path, context, corrected)
            result = parse_fit_output(out, total)
            log.info("Fit for %s: %s", file_name, result)
            return result
        except (OSError, ValueError, subprocess.TimeoutExpired):
            log.warning("llama-fit-params failed for %s; letting llama.cpp decide", file_name)
            return FitResult(args=[], gpu_layers=total, total_layers=total, cpu_blocks=0, known=False)

    def _run_fit(self, path: Path, context: int, margin_mib: int) -> tuple[str, str]:
        exe = self.paths.runtime / "llama.cpp" / "llama-fit-params.exe"
        proc = subprocess.run(
            # -lv 4: the "free device memory" line used for the Windows correction is only logged verbosely
            [str(exe), "-m", str(path), "-c", str(context), "--fit-target", str(margin_mib), "-lv", "4",
             "--log-colors", "off"],
            capture_output=True, text=True, timeout=120,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return proc.stdout, proc.stderr
