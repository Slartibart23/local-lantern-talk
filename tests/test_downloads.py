"""Resumable downloads against a local HTTP server with Range support."""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from lantern_talk.downloads import DownloadError, Downloader, DownloadJob

DATA = bytes(range(256)) * 40_000  # ~10 MB
SLOW = threading.Event()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:  # keep test output quiet
        pass

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/gated"):
            self.send_response(403)
            self.end_headers()
            return
        start = 0
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(DATA) - 1}/{len(DATA)}")
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(DATA) - start))
        self.end_headers()
        body = DATA[start:]
        for i in range(0, len(body), 1 << 16):
            try:
                self.wfile.write(body[i: i + (1 << 16)])
            except OSError:
                return
            if SLOW.is_set():
                time.sleep(0.01)


@pytest.fixture(scope="module")
def server():  # noqa: ANN201
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def run(downloader: Downloader, *args: object, **kwargs: object) -> DownloadJob:
    done = threading.Event()
    result: list[DownloadJob] = []

    def on_done(job: DownloadJob) -> None:
        result.append(job)
        done.set()

    downloader.start(*args, on_done=on_done, **kwargs)  # type: ignore[arg-type]
    assert done.wait(30)
    return result[0]


def test_full_download(server: str, tmp_path: Path) -> None:
    updates: list[int] = []
    d = Downloader(lambda job: updates.append(job.done))
    job = run(d, "m", f"{server}/model.gguf", tmp_path / "model.gguf", len(DATA))
    assert job.state == "done" and (tmp_path / "model.gguf").read_bytes() == DATA
    assert not (tmp_path / "model.gguf.part").exists()
    assert updates[-1] == len(DATA)


def test_resume_from_partial_file(server: str, tmp_path: Path) -> None:
    (tmp_path / "model.gguf.part").write_bytes(DATA[:3_000_000])
    job = run(Downloader(lambda j: None), "m", f"{server}/model.gguf", tmp_path / "model.gguf", len(DATA))
    assert job.state == "done" and (tmp_path / "model.gguf").read_bytes() == DATA


def test_cancel_keeps_partial_file(server: str, tmp_path: Path) -> None:
    SLOW.set()
    try:
        d = Downloader(lambda j: None)
        done = threading.Event()
        d.start("m", f"{server}/model.gguf", tmp_path / "model.gguf", len(DATA), on_done=lambda j: done.set())
        time.sleep(0.3)
        assert d.busy
        with pytest.raises(DownloadError, match="Another download"):
            d.start("x", f"{server}/x.gguf", tmp_path / "x.gguf", 1)
        d.cancel()
        assert done.wait(10)
        assert d.job is not None and d.job.state == "cancelled"
        assert not (tmp_path / "model.gguf").exists()
        assert 0 < (tmp_path / "model.gguf.part").stat().st_size < len(DATA)
    finally:
        SLOW.clear()


def test_wrong_size_is_rejected(server: str, tmp_path: Path) -> None:
    job = run(Downloader(lambda j: None), "m", f"{server}/model.gguf", tmp_path / "model.gguf", len(DATA) + 5)
    assert job.state == "error" and "wrong size" in job.error
    assert not (tmp_path / "model.gguf").exists() and not (tmp_path / "model.gguf.part").exists()


def test_gated_model_message(server: str, tmp_path: Path) -> None:
    job = run(Downloader(lambda j: None), "m", f"{server}/gated/model.gguf", tmp_path / "model.gguf", 0)
    assert job.state == "error" and "license" in job.error


def test_not_enough_disk_space(server: str, tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="disk space"):
        Downloader(lambda j: None).start("m", f"{server}/m.gguf", tmp_path / "m.gguf", 10**15)
