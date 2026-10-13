"""Model downloads: one GGUF file at a time, resumable, with progress (spec §13.3/§13.4).

Only the single requested file is downloaded — repositories are never cloned.
Partial data is kept as ``<file>.part`` so a download continues after a restart.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import quote

import httpx

log = logging.getLogger(__name__)

CHUNK = 1 << 20
PROGRESS_INTERVAL_S = 0.5
DISK_RESERVE_BYTES = 1 << 30   # keep 1 GB free on the drive
HF = "https://huggingface.co"


class DownloadError(Exception):
    """A download problem with a message meant for the user."""


def hf_file_url(repo: str, file: str) -> str:
    return f"{HF}/{repo}/resolve/main/{quote(file)}"


def _auth_headers(url: str) -> dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    if token and url.startswith(HF):
        return {"Authorization": f"Bearer {token}"}  # httpx drops it on redirects to other hosts
    return {}


def _http_error(status: int, what: str) -> DownloadError:
    if status in (401, 403):
        return DownloadError(f"{what} requires accepting its license on Hugging Face (gated model). "
                             "Accept it on the model page and set an HF_TOKEN, or pick another model.")
    if status == 404:
        return DownloadError(f"{what} was not found. Check the repository name or link.")
    return DownloadError(f"{what}: the server answered with HTTP {status}.")


def list_repo_files(repo: str, timeout: float = 20.0) -> dict[str, int]:
    """``{path: size}`` of all files in a Hugging Face model repository."""
    url = f"{HF}/api/models/{repo}/tree/main?recursive=true"
    try:
        r = httpx.get(url, headers=_auth_headers(url), timeout=timeout, follow_redirects=True)
    except httpx.HTTPError as exc:
        raise DownloadError(f"Could not reach Hugging Face ({exc.__class__.__name__}). Check the internet connection.") from exc
    if r.status_code != 200:
        raise _http_error(r.status_code, f"Repository {repo}")
    return {f["path"]: int((f.get("lfs") or {}).get("size", f.get("size", 0)))
            for f in r.json() if f.get("type") == "file"}


def search_gguf_repos(query: str, limit: int = 3, timeout: float = 20.0) -> list[dict[str, object]]:
    """Most downloaded Hugging Face repositories with GGUF files matching ``query``."""
    try:
        r = httpx.get(f"{HF}/api/models", params={"search": query, "filter": "gguf", "sort": "downloads",
                                                   "limit": limit}, timeout=timeout)
        r.raise_for_status()
        return [{"id": m["id"], "downloads": int(m.get("downloads") or 0)} for m in r.json()]
    except (httpx.HTTPError, ValueError, KeyError):
        return []  # suggestions are a convenience; never fail because of them


def remote_size(url: str, timeout: float = 20.0) -> int:
    """Size of a remote file in bytes (0 if the server does not say)."""
    try:
        r = httpx.head(url, headers=_auth_headers(url), timeout=timeout, follow_redirects=True)
    except httpx.HTTPError as exc:
        raise DownloadError(f"Could not reach {httpx.URL(url).host} ({exc.__class__.__name__}).") from exc
    if r.status_code >= 400:
        raise _http_error(r.status_code, "The file")
    return int(r.headers.get("content-length") or 0)


@dataclass
class DownloadJob:
    key: str                      # model id
    file_name: str
    url: str
    dest: str
    expected_size: int = 0
    done: int = 0
    total: int = 0
    speed: float = 0.0            # bytes per second
    state: str = "queued"         # queued | running | done | error | cancelled
    error: str = ""
    started: float = field(default_factory=time.time)

    def public(self) -> dict[str, object]:
        d = asdict(self)
        d.pop("url")
        d.pop("dest")
        return d


class Downloader:
    """Runs one download at a time in a background thread."""

    def __init__(self, on_update: Callable[[DownloadJob], None]) -> None:
        self.on_update = on_update
        self.job: DownloadJob | None = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self.job is not None and self.job.state in ("queued", "running")

    def start(self, key: str, url: str, dest: Path, expected_size: int = 0,
              on_done: Callable[[DownloadJob], None] | None = None) -> DownloadJob:
        with self._lock:
            if self.busy:
                raise DownloadError("Another download is running. Wait for it or cancel it first.")
            free = shutil.disk_usage(dest.parent).free
            part = dest.with_name(dest.name + ".part")
            already = part.stat().st_size if part.exists() else 0
            if expected_size and free < expected_size - already + DISK_RESERVE_BYTES:
                raise DownloadError(f"Not enough free disk space: {(expected_size - already) / 1e9:.1f} GB needed, "
                                    f"{free / 1e9:.1f} GB free.")
            self._cancel.clear()
            self.job = DownloadJob(key=key, file_name=dest.name, url=url, dest=str(dest), expected_size=expected_size)
            threading.Thread(target=self._run, args=(self.job, on_done), name="download", daemon=True).start()
            return self.job

    def cancel(self) -> None:
        self._cancel.set()

    def _emit(self, job: DownloadJob) -> None:
        try:
            self.on_update(job)
        except Exception:  # noqa: BLE001
            log.exception("download progress callback failed")

    def _run(self, job: DownloadJob, on_done: Callable[[DownloadJob], None] | None) -> None:
        dest = Path(job.dest)
        part = dest.with_name(dest.name + ".part")
        try:
            job.state = "running"
            self._download(job, part)
            if self._cancel.is_set():
                job.state = "cancelled"  # the .part file is kept so the download can resume later
            else:
                if job.expected_size and part.stat().st_size != job.expected_size:
                    part.unlink(missing_ok=True)
                    raise DownloadError("The downloaded file has the wrong size and was discarded. Please try again.")
                os.replace(part, dest)
                job.state = "done"
        except DownloadError as exc:
            job.state, job.error = "error", str(exc)
        except Exception as exc:  # noqa: BLE001 - network hiccups, disk full ...
            log.exception("Download failed")
            job.state, job.error = "error", f"Download failed ({exc.__class__.__name__}). It can be resumed."
        self._emit(job)
        if on_done is not None:
            on_done(job)

    def _download(self, job: DownloadJob, part: Path) -> None:
        part.parent.mkdir(parents=True, exist_ok=True)
        offset = part.stat().st_size if part.exists() else 0
        if job.expected_size and offset > job.expected_size:
            part.unlink()
            offset = 0
        if job.expected_size and offset == job.expected_size:
            job.done = job.total = offset
            return
        headers = {**_auth_headers(job.url), **({"Range": f"bytes={offset}-"} if offset else {})}
        timeout = httpx.Timeout(30.0, read=60.0)
        with httpx.stream("GET", job.url, headers=headers, timeout=timeout, follow_redirects=True) as r:
            if r.status_code == 416:  # range not satisfiable: we already have everything
                job.done = job.total = offset
                return
            if r.status_code >= 400:
                raise _http_error(r.status_code, job.file_name)
            if offset and r.status_code != 206:
                offset = 0  # server ignored the range: start over
            length = int(r.headers.get("content-length") or 0)
            job.total = job.expected_size or (offset + length if length else 0)
            job.done = offset
            mode = "ab" if offset else "wb"
            last_emit, last_bytes, last_time = 0.0, offset, time.monotonic()
            with open(part, mode) as out:
                for chunk in r.iter_bytes(CHUNK):
                    if self._cancel.is_set():
                        return
                    out.write(chunk)
                    job.done += len(chunk)
                    now = time.monotonic()
                    if now - last_emit >= PROGRESS_INTERVAL_S:
                        job.speed = (job.done - last_bytes) / max(now - last_time, 1e-6)
                        last_emit, last_bytes, last_time = now, job.done, now
                        self._emit(job)
