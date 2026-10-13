"""Web backend: serves the UI and bridges the browser and the :class:`Engine`.

Run via ``start.bat``. Binds to 127.0.0.1 only. One browser tab is active at a
time; opening another tab takes over the session.

WebSocket protocol (``/ws``)
----------------------------
Browser → server, JSON: ``text``, ``mic_format`` (sample_rate), ``playback_ended``,
``pause``, ``settings`` (values), ``new_conversation``, ``reference_text`` (voice, text),
``download_model`` (id, quant), ``cancel_download``, ``delete_model`` (file),
``add_custom_model`` (source, quant, variant), ``remove_custom_model`` (id),
``set_model_label`` (file, variant). Binary: microphone PCM
(int16 mono at the announced sample rate).

Server → browser, JSON: ``hello`` (full snapshot), ``state``, ``user_message``,
``assistant_start``, ``segment``, ``audio`` (immediately followed by one binary
message with int16 PCM), ``turn_done``, ``settings``, ``components``, ``voices``, ``voice_info``,
``models``, ``model_status``, ``download`` (progress),
``cleared``, ``error``, ``replaced``.

HTTP: ``POST /api/voices?filename=…`` with the raw audio file as body adds a
custom voice to ``user_data/voices``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import socket
import sys
import threading
import webbrowser
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .engine import MAX_VOICE_UPLOAD_BYTES, Engine, UserError, build_components
from .gpu import query_gpu
from .logging_setup import setup_logging
from .paths import Paths, apply_portable_env, get_paths, path_warnings
from .settings import SettingsStore

log = logging.getLogger(__name__)

HOST = "127.0.0.1"
DEFAULT_PORT = 7865
APP_ID = "local-lantern-talk"


class Hub:
    """Delivers engine events (from worker threads) to the active WebSocket."""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.queue: asyncio.Queue[tuple[dict[str, Any], bytes | None]] | None = None
        self.client: WebSocket | None = None
        self.engine: Engine | None = None

    def emit(self, msg: dict[str, Any], binary: bytes | None = None) -> None:
        if self.loop is None or self.queue is None:
            return
        if self.client is None and msg.get("type") == "audio":
            # Nobody can play it: don't let the engine wait for a playback that never happens.
            if self.engine is not None:
                self.engine.playback_ended()
            return
        self.loop.call_soon_threadsafe(self.queue.put_nowait, (msg, binary))


def create_app(paths: Paths, port: int, engine_factory=None) -> FastAPI:  # noqa: ANN001
    hub = Hub()
    allowed_hosts = {f"{HOST}:{port}", f"localhost:{port}"}
    allowed_origins = {f"http://{h}" for h in allowed_hosts}

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ANN202
        hub.loop = asyncio.get_running_loop()
        hub.queue = asyncio.Queue()
        if engine_factory is not None:
            hub.engine = engine_factory(hub.emit)
        else:
            store = SettingsStore(paths.settings_file)
            components = build_components(paths, store.settings)
            hub.engine = Engine(paths, store, components, hub.emit)
        hub.engine.start()
        sender = asyncio.create_task(_sender(hub))
        try:
            yield
        finally:
            sender.cancel()
            await asyncio.to_thread(hub.engine.shutdown)

    app = FastAPI(title="Local Lantern Talk", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.hub = hub

    @app.middleware("http")
    async def check_host(request: Request, call_next):  # noqa: ANN001, ANN202
        # Blocks DNS-rebinding: only requests addressed to this local server are served.
        if request.headers.get("host") not in allowed_hosts:
            return PlainTextResponse("Forbidden", status_code=403)
        return await call_next(request)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(paths.web / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/ping")
    async def ping() -> dict[str, str]:
        return {"app": APP_ID, "version": __version__}

    @app.get("/api/gpu")
    async def gpu() -> JSONResponse:
        info = await asyncio.to_thread(query_gpu)
        return JSONResponse(info or {})

    @app.post("/api/voices")
    async def upload_voice(request: Request, filename: str) -> JSONResponse:
        # A foreign page can POST to localhost too (CSRF); only our own page may upload.
        if request.headers.get("origin") not in allowed_origins:
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_VOICE_UPLOAD_BYTES:
                return JSONResponse({"error": "The file is too large."}, status_code=413)
        assert hub.engine is not None
        try:
            result = await asyncio.to_thread(hub.engine.add_voice, filename, bytes(body))
        except UserError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(result)

    app.mount("/static", StaticFiles(directory=paths.web), name="static")
    if paths.fonts.is_dir():
        app.mount("/fonts", StaticFiles(directory=paths.fonts), name="fonts")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        # Any web page could try to connect to localhost; only our own page may.
        if ws.headers.get("origin") not in allowed_origins or ws.headers.get("host") not in allowed_hosts:
            await ws.close(code=1008)
            return
        await ws.accept()
        previous, hub.client = hub.client, ws
        if previous is not None:
            with contextlib.suppress(Exception):
                await previous.send_json({"type": "replaced"})
                await previous.close()
        engine = hub.engine
        assert engine is not None
        await ws.send_json(engine.snapshot())
        sample_rate = 48_000
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    engine.on_audio(message["bytes"], sample_rate)
                    continue
                data = json.loads(message.get("text") or "{}")
                kind = data.get("type")
                if kind != "text":  # message text is personal; never log it
                    log.debug("ws <- %s", data)
                if kind == "mic_format":
                    sample_rate = int(data["sample_rate"])
                elif kind == "text":
                    engine.submit_text(str(data.get("text", ""))[:4000])
                elif kind == "playback_ended":
                    engine.playback_ended()
                elif kind == "pause":
                    engine.set_paused(bool(data.get("paused")))
                elif kind == "settings":
                    engine.update_settings(dict(data.get("values") or {}))
                elif kind == "new_conversation":
                    engine.new_conversation()
                elif kind == "download_model":
                    engine.download_model(str(data.get("id", "")), data.get("quant") or None)
                elif kind == "cancel_download":
                    engine.cancel_download()
                elif kind == "delete_model":
                    engine.delete_model(str(data.get("file", "")))
                elif kind == "add_custom_model":
                    engine.add_custom_model(str(data.get("source", "")), data.get("quant") or None,
                                            str(data.get("variant", "unknown")))
                elif kind == "remove_custom_model":
                    engine.remove_custom_model(str(data.get("id", "")))
                elif kind == "set_model_label":
                    engine.set_model_label(str(data.get("file", "")), str(data.get("variant", "")))
                elif kind == "reference_text":
                    engine.set_reference_text(str(data.get("voice", "")), str(data.get("text", "")))
                elif kind == "refresh":
                    await ws.send_json(engine.snapshot())
        except WebSocketDisconnect:
            pass
        finally:
            if hub.client is ws:
                hub.client = None
                engine.playback_ended()  # nobody is listening any more

    return app


async def _sender(hub: Hub) -> None:
    assert hub.queue is not None
    while True:
        msg, binary = await hub.queue.get()
        ws = hub.client
        if ws is None:
            continue
        try:
            await ws.send_json(msg)
            if binary is not None:
                await ws.send_bytes(binary)
        except Exception:  # noqa: BLE001 - client went away; the receive loop cleans up
            log.debug("send failed", exc_info=True)


# ---------------------------------------------------------------------------- startup

def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((HOST, port))
            return True
        except OSError:
            return False


def _running_instance(port: int) -> bool:
    try:
        return httpx.get(f"http://{HOST}:{port}/api/ping", timeout=1.0).json().get("app") == APP_ID
    except Exception:  # noqa: BLE001
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="start.bat", description="Local Lantern Talk web app")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = parser.parse_args(argv)

    paths = get_paths()
    apply_portable_env(paths)
    log_file = setup_logging(paths.logs, "server")
    for w in path_warnings(paths.root):
        print(f"Warning: {w}")

    port = args.port
    if not _port_free(port):
        if _running_instance(port):
            print(f"Local Lantern Talk is already running: http://{HOST}:{port}")
            if not args.no_browser:
                webbrowser.open(f"http://{HOST}:{port}")
            return 0
        port = next((p for p in range(port + 1, port + 50) if _port_free(p)), 0)
        if not port:
            print("No free port found.")
            return 1

    url = f"http://{HOST}:{port}"
    print(f"Local Lantern Talk is starting at {url}")
    print("Keep this window open while you use the app; close it to quit.")
    print(f"Log file: {log_file}")
    config = uvicorn.Config(create_app(paths, port), host=HOST, port=port, log_level="warning",
                            access_log=False, ws_max_size=8 * 2**20)
    server = uvicorn.Server(config)

    if not args.no_browser:
        def open_when_ready() -> None:
            while not server.started and not server.should_exit:
                threading.Event().wait(0.1)
            if server.started:
                webbrowser.open(url)

        threading.Thread(target=open_when_ready, daemon=True).start()
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
