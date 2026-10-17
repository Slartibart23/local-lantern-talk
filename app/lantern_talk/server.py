"""Web backend: serves the UI and bridges the browser and the :class:`Engine`.

Run via ``start.bat``. Binds to 127.0.0.1 only. One browser tab is active at a
time; opening another tab takes over the session.

WebSocket protocol (``/ws``)
----------------------------
Browser → server, JSON: ``text``, ``mic_format`` (sample_rate), ``playback_ended``,
``pause``, ``settings`` (values), ``new_conversation``, ``reference_text`` (voice, text),
``download_model`` (id, quant), ``cancel_download``, ``delete_model`` (file),
``add_custom_model`` (source, quant, variant), ``remove_custom_model`` (id),
``set_model_label`` (file, variant), ``preview_voice`` (voice), ``rename_voice`` (voice, name),
``delete_voice`` (voice), ``regenerate_builtin_voices``, ``delete_from`` (id), ``regenerate``,
``edit_last`` (text), ``stop_reply``, ``talk_start``, ``talk_end`` (send), ``director_note`` (text),
``pin_note`` (text), ``unpin_note`` (index), ``continue``. Binary: microphone PCM
(int16 mono at the announced sample rate).

Server → browser, JSON: ``hello`` (full snapshot), ``state``, ``user_message``,
``assistant_start``, ``segment``, ``audio`` (immediately followed by one binary
message with int16 PCM), ``turn_done``, ``settings``, ``components``, ``voices``, ``voice_info``,
``models``, ``model_status``, ``download`` (progress), ``custom_suggestions``,
``preview_start`` / ``preview_done`` (voice preview audio in between), ``notice``,
``cleared``, ``error``, ``replaced``.

HTTP: ``POST /api/voices?filename=…`` with the raw audio file as body adds a
custom voice to ``user_data/voices``. ``POST /api/images?kind=avatar|background|user-avatar``
stores a picture the browser has cropped and shrunk (``user_data/images``);
``GET /api/images/<name>`` serves it to the app's own page.
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
from urllib.parse import quote

import httpx
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .engine import MAX_VOICE_UPLOAD_BYTES, Engine, UserError, build_components
from .gpu import query_gpu
from .images import MAX_IMAGE_BYTES, ImageStore
from .logging_setup import setup_logging
from .paths import Paths, apply_portable_env, get_paths, path_warnings
from .recorder import SessionRecorder, list_sessions, load_session, valid_part
from .settings import SettingsStore

log = logging.getLogger(__name__)

# WebSocket messages with personal content (chat text, characters, facts) never go to the log.
PRIVATE_MESSAGES = {"text", "save_character", "add_fact", "update_fact", "edit_last", "director_note", "pin_note",
                    "premise", "new_story", "director_draft", "draft_character", "edit_message", "settings"}

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
    app.state.request_quit = lambda: None  # set by main(): stops the server (and with it the engine)

    @app.middleware("http")
    async def check_host(request: Request, call_next):  # noqa: ANN001, ANN202
        # Blocks DNS-rebinding: only requests addressed to this local server are served.
        if request.headers.get("host") not in allowed_hosts:
            return PlainTextResponse("Forbidden", status_code=403)
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            # Always revalidate the page's scripts and styles, so a browser never mixes a new
            # page with an old cached app.js after an update (cheap on localhost: 304s).
            response.headers["Cache-Control"] = "no-cache"
        return response

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

    def same_site(request: Request) -> bool:
        # Recordings are personal: another web page must not even embed them (spec §12).
        return request.headers.get("sec-fetch-site", "same-origin") in ("same-origin", "none")

    @app.post("/api/images")
    async def upload_image(request: Request, kind: str) -> JSONResponse:
        if request.headers.get("origin") not in allowed_origins:
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_IMAGE_BYTES:
                return JSONResponse({"error": "The picture is too large."}, status_code=413)
        assert hub.engine is not None
        try:
            result = await asyncio.to_thread(hub.engine.add_image, kind, bytes(body))
        except UserError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(result)

    images = ImageStore(paths.images)

    @app.get("/api/characters/{char_id}/export")
    async def export_character(char_id: str, request: Request) -> Any:
        if not same_site(request):  # only our own page may read characters
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        assert hub.engine is not None
        try:
            name, data = await asyncio.to_thread(hub.engine.export_character, char_id)
        except UserError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        quoted = quote(name)
        return Response(data, media_type="application/json",
                        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quoted}",
                                 "Cache-Control": "no-store"})

    @app.post("/api/characters/import")
    async def import_character(request: Request) -> JSONResponse:
        if request.headers.get("origin") not in allowed_origins:
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > 4 * MAX_IMAGE_BYTES:
                return JSONResponse({"error": "The file is too large."}, status_code=413)
        assert hub.engine is not None
        try:
            result = await asyncio.to_thread(hub.engine.import_character, bytes(body))
        except UserError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(result)

    @app.get("/api/images/{name}")
    async def image(name: str, request: Request) -> Any:
        if not same_site(request):  # personal pictures: no other web page may embed them
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        path = images.path(name)
        if path is None:
            return JSONResponse({"error": "Not found"}, status_code=404)
        media = {".webp": "image/webp", ".png": "image/png", ".jpg": "image/jpeg"}[path.suffix]
        # Every new picture gets a new name, so a cached copy is never outdated.
        return FileResponse(path, media_type=media, headers={"Cache-Control": "private, max-age=31536000, immutable"})

    def session_file(name: str, suffix: str) -> Any:
        if not SessionRecorder.valid_name(name):
            return None
        path = paths.sessions / f"{name}{suffix}"
        return path if path.is_file() else None

    @app.get("/api/sessions")
    async def sessions(request: Request) -> JSONResponse:
        if not same_site(request):
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        return JSONResponse(await asyncio.to_thread(list_sessions, paths.sessions))

    @app.get("/api/sessions/{name}.mp3")
    async def session_audio(name: str, request: Request) -> Any:
        path = session_file(name, ".mp3")
        if not same_site(request):
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        if path is None:
            return JSONResponse({"error": "Not found"}, status_code=404)
        return FileResponse(path, media_type="audio/mpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/sessions/{name}/parts/{part}")
    async def session_part(name: str, part: str, request: Request) -> Any:
        """One utterance of a chat that is still open (its own MP3 in the session's folder)."""
        if not same_site(request):
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        if not SessionRecorder.valid_name(name) or not valid_part(part):
            return JSONResponse({"error": "Not found"}, status_code=404)
        path = paths.sessions / name / part
        if not path.is_file():
            return JSONResponse({"error": "Not found"}, status_code=404)
        return FileResponse(path, media_type="audio/mpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/sessions/{name}")
    async def session_transcript(name: str, request: Request) -> JSONResponse:
        path = session_file(name, ".md")
        if not same_site(request):
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        if path is None:
            return JSONResponse({"error": "Not found"}, status_code=404)
        data = await asyncio.to_thread(load_session, paths.sessions, name)
        has_audio = (paths.sessions / f"{name}.mp3").is_file()
        return JSONResponse({"name": name, "audio": has_audio, **data})

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
                if kind not in PRIVATE_MESSAGES:  # personal content is never logged
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
                elif kind == "delete_from":
                    engine.delete_from(int(data.get("id", 0)))
                elif kind == "regenerate":
                    engine.regenerate()
                elif kind == "edit_last":
                    engine.edit_last(str(data.get("text", ""))[:4000])
                elif kind == "edit_message":
                    engine.edit_message(int(data.get("id", 0)), str(data.get("text", ""))[:4000])
                elif kind == "stop_reply":
                    engine.stop_reply()
                elif kind == "director_note":
                    engine.set_pending_note(str(data.get("text", "")), str(data.get("target", "")))
                elif kind == "director_draft":
                    engine.set_director_draft(str(data.get("text", "")), str(data.get("target", "")))
                elif kind == "pin_note":
                    engine.pin_note(str(data.get("text", ""))[:300], str(data.get("target", "")))
                elif kind == "unpin_note":
                    engine.unpin_note(int(data.get("index", -1)))
                elif kind == "continue":
                    engine.continue_story(str(data.get("speaker", "")))
                elif kind == "set_group":
                    engine.set_group([str(i) for i in data.get("ids") or []])
                elif kind == "open_group":
                    engine.open_group([str(i) for i in data.get("ids") or []])
                elif kind == "group_mode":
                    engine.set_group_mode(str(data.get("mode", "")))
                elif kind == "story_play":
                    engine.story_play()
                elif kind == "story_pause":
                    engine.story_pause()
                elif kind == "next_speaker":
                    engine.set_next_speaker(str(data.get("id", "")))
                elif kind == "premise":
                    engine.set_premise(str(data.get("text", ""))[:3000])
                elif kind == "new_story":
                    engine.new_story(str(data.get("text", ""))[:3000])
                elif kind == "draft_character":
                    values = data.get("character")
                    if isinstance(values, dict):
                        engine.draft_character(values)
                elif kind == "talk_start":
                    engine.talk_start()
                elif kind == "talk_end":
                    engine.talk_end(bool(data.get("send", True)))
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
                elif kind == "preview_voice":
                    engine.preview_voice(str(data.get("voice", "")))
                elif kind == "rename_voice":
                    engine.rename_voice(str(data.get("voice", "")), str(data.get("name", ""))[:120])
                elif kind == "delete_voice":
                    engine.delete_voice(str(data.get("voice", "")))
                elif kind == "regenerate_builtin_voices":
                    engine.regenerate_builtin_voices()
                elif kind == "voice_language":
                    engine.set_voice_language(str(data.get("voice", "")), str(data.get("language", ""))[:8])
                elif kind == "reference_text":
                    engine.set_reference_text(str(data.get("voice", "")), str(data.get("text", "")))
                elif kind == "select_character":
                    engine.select_character(str(data.get("id", "")))
                elif kind == "save_character":
                    values = data.get("character")
                    if isinstance(values, dict):
                        engine.save_character(values, str(data["id"]) if data.get("id") else None)
                elif kind == "duplicate_character":
                    engine.duplicate_character(str(data.get("id", "")))
                elif kind == "delete_character":
                    engine.delete_character(str(data.get("id", "")))
                elif kind == "add_fact":
                    engine.add_fact(str(data.get("text", ""))[:500])
                elif kind == "update_fact":
                    engine.update_fact(int(data.get("id", 0)), str(data.get("text", ""))[:500])
                elif kind == "delete_fact":
                    engine.delete_fact(int(data.get("id", 0)))
                elif kind == "reset_memory":
                    engine.reset_memory()
                elif kind == "replay":
                    engine.set_replaying(bool(data.get("playing")))
                elif kind == "open_sessions_folder":
                    await asyncio.to_thread(engine.open_sessions_folder)
                elif kind == "refresh":
                    await ws.send_json(engine.snapshot())
                elif kind == "quit":
                    # "Quit app" in the UI: stop everything and free the GPU memory.
                    log.info("Quit requested from the browser")
                    await ws.send_json({"type": "bye"})
                    app.state.request_quit()
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
    print("Loading the models can take up to 2 minutes; the browser page tells you when it is ready.")
    print("Keep this window open while you use the app; close it (or click Quit app) to quit.")
    print(f"Log file: {log_file}")
    config = uvicorn.Config(create_app(paths, port), host=HOST, port=port, log_level="warning",
                            access_log=False, ws_max_size=8 * 2**20, timeout_graceful_shutdown=5)
    server = uvicorn.Server(config)
    config.app.state.request_quit = lambda: setattr(server, "should_exit", True)

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
    print("Local Lantern Talk has stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
