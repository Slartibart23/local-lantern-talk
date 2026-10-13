"""Web server tests: local-only access checks and the WebSocket protocol (fake engine)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from lantern_talk.paths import Paths
from lantern_talk.server import create_app

ROOT = Path(__file__).parents[1]
PORT = 7865
BASE = f"http://127.0.0.1:{PORT}"
GOOD_ORIGIN = {"origin": BASE}
WS_URL = f"ws://127.0.0.1:{PORT}/ws"  # TestClient would otherwise use host "testserver"


class FakeEngine:
    def __init__(self, emit) -> None:  # noqa: ANN001
        self.emit = emit
        self.calls: list[tuple[str, Any]] = []

    def start(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def snapshot(self) -> dict[str, Any]:
        return {"type": "hello", "state": "listening"}

    def submit_text(self, text: str) -> None:
        self.calls.append(("text", text))
        self.emit({"type": "user_message", "text": text}, None)

    def on_audio(self, pcm: bytes, sample_rate: int) -> None:
        self.calls.append(("audio", (len(pcm), sample_rate)))

    def playback_ended(self) -> None:
        self.calls.append(("playback_ended", None))

    def set_paused(self, paused: bool) -> None:
        self.calls.append(("pause", paused))

    def update_settings(self, values: dict[str, Any]) -> None:
        self.calls.append(("settings", values))

    def new_conversation(self) -> None:
        self.calls.append(("new", None))

    def __getattr__(self, name: str):  # noqa: ANN204 - character / memory commands are only recorded
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *args: self.calls.append((name, args))

    def add_voice(self, filename: str, data: bytes) -> dict[str, Any]:
        from lantern_talk.engine import UserError

        if not filename.endswith(".wav"):
            raise UserError("Please upload an MP3, WAV, FLAC or OGG file.")
        self.calls.append(("add_voice", (filename, len(data))))
        return {"id": f"custom:{filename}", "warnings": []}


@pytest.fixture()
def client():  # noqa: ANN201
    engines: list[FakeEngine] = []

    def factory(emit):  # noqa: ANN001, ANN202
        engines.append(FakeEngine(emit))
        return engines[-1]

    app = create_app(Paths(ROOT), PORT, engine_factory=factory)
    with TestClient(app, base_url=BASE) as c:
        c.engine = engines[0]  # type: ignore[attr-defined]
        yield c


def test_index_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200 and "Local Lantern Talk" in r.text
    js = client.get("/static/app.js")
    assert js.status_code == 200 and js.headers["cache-control"] == "no-cache"  # never a stale app.js
    assert client.get("/api/ping").json()["app"] == "local-lantern-talk"


def test_foreign_host_rejected(client: TestClient) -> None:
    # DNS rebinding: a foreign hostname pointing at 127.0.0.1 must not reach the app.
    assert client.get("/", headers={"host": "evil.example:7865"}).status_code == 403


@pytest.mark.parametrize("origin", [None, "http://evil.example", "http://127.0.0.1:9999"])
def test_websocket_rejects_foreign_origin(client: TestClient, origin: str | None) -> None:
    headers = {"origin": origin} if origin else {}
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(WS_URL, headers=headers) as ws:
            ws.receive_json()


def test_websocket_protocol(client: TestClient) -> None:
    engine: FakeEngine = client.engine  # type: ignore[attr-defined]
    with client.websocket_connect(WS_URL, headers=GOOD_ORIGIN) as ws:
        assert ws.receive_json()["type"] == "hello"
        ws.send_json({"type": "mic_format", "sample_rate": 44100})
        ws.send_bytes(b"\x00\x00" * 1764)
        ws.send_json({"type": "text", "text": "Hallo"})
        assert ws.receive_json() == {"type": "user_message", "text": "Hallo"}
        ws.send_json({"type": "settings", "values": {"mic_enabled": False}})
        ws.send_json({"type": "pause", "paused": True})
        ws.send_json({"type": "playback_ended"})
        ws.send_json({"type": "new_conversation"})
        ws.send_json({"type": "refresh"})
        assert ws.receive_json()["type"] == "hello"
    names = [c[0] for c in engine.calls]
    assert ("audio", (3528, 44100)) in engine.calls
    assert names[:6] == ["audio", "text", "settings", "pause", "playback_ended", "new"]
    assert names[-1] == "playback_ended"  # disconnect releases a waiting turn


def test_character_and_memory_commands(client: TestClient) -> None:
    engine: FakeEngine = client.engine  # type: ignore[attr-defined]
    with client.websocket_connect(WS_URL, headers=GOOD_ORIGIN) as ws:
        ws.receive_json()
        ws.send_json({"type": "select_character", "id": "max"})
        ws.send_json({"type": "save_character", "character": {"name": "Hilde"}})
        ws.send_json({"type": "save_character", "id": "max", "character": {"name": "Max"}})
        ws.send_json({"type": "save_character", "character": "not a dict"})
        ws.send_json({"type": "duplicate_character", "id": "max"})
        ws.send_json({"type": "delete_character", "id": "max"})
        ws.send_json({"type": "add_fact", "text": "x" * 900})
        ws.send_json({"type": "update_fact", "id": "3", "text": "y"})
        ws.send_json({"type": "delete_fact", "id": 3})
        ws.send_json({"type": "reset_memory"})
        ws.send_json({"type": "refresh"})
        ws.receive_json()
    assert engine.calls[:9] == [
        ("select_character", ("max",)),
        ("save_character", ({"name": "Hilde"}, None)),
        ("save_character", ({"name": "Max"}, "max")),
        ("duplicate_character", ("max",)),
        ("delete_character", ("max",)),
        ("add_fact", ("x" * 500,)),
        ("update_fact", (3, "y")),
        ("delete_fact", (3,)),
        ("reset_memory", ()),
    ]


def test_quit_says_bye_and_stops_the_server(client: TestClient) -> None:
    quits: list[bool] = []
    client.app.state.request_quit = lambda: quits.append(True)
    with client.websocket_connect(WS_URL, headers=GOOD_ORIGIN) as ws:
        ws.receive_json()
        ws.send_json({"type": "quit"})
        assert ws.receive_json() == {"type": "bye"}
    assert quits == [True]


def test_second_tab_takes_over(client: TestClient) -> None:
    with client.websocket_connect(WS_URL, headers=GOOD_ORIGIN) as first:
        first.receive_json()
        with client.websocket_connect(WS_URL, headers=GOOD_ORIGIN) as second:
            assert second.receive_json()["type"] == "hello"
            assert first.receive_json()["type"] == "replaced"


def test_voice_upload(client: TestClient) -> None:
    r = client.post("/api/voices", params={"filename": "me.wav"}, content=b"x" * 100, headers=GOOD_ORIGIN)
    assert r.status_code == 200 and r.json()["id"] == "custom:me.wav"
    assert ("add_voice", ("me.wav", 100)) in client.engine.calls  # type: ignore[attr-defined]
    r = client.post("/api/voices", params={"filename": "me.exe"}, content=b"x", headers=GOOD_ORIGIN)
    assert r.status_code == 400 and "MP3" in r.json()["error"]


@pytest.mark.parametrize("origin", [None, "http://evil.example"])
def test_voice_upload_rejects_foreign_origin(client: TestClient, origin: str | None) -> None:
    # CSRF: a foreign page can POST to localhost, but must not be able to add files.
    headers = {"origin": origin} if origin else {}
    r = client.post("/api/voices", params={"filename": "me.wav"}, content=b"x", headers=headers)
    assert r.status_code == 403
    assert not any(c[0] == "add_voice" for c in client.engine.calls)  # type: ignore[attr-defined]


def test_session_files_are_served_only_to_the_app(tmp_path: Path) -> None:
    import shutil

    root = tmp_path / "root"
    shutil.copytree(ROOT / "app" / "web", root / "app" / "web")
    paths = Paths(root)
    paths.sessions.mkdir(parents=True)
    (paths.sessions / "2026-10-13_1538_lena.md").write_text(
        "# Lena — 13 Oct 2026, 15:38\n\n### [00:00.0] Lena · greeting · no audio\n\nHallo!\n\n"
        "### [00:00.5] You · spoken · de · 1.2 s\n\nHi.\n", encoding="utf-8")
    (paths.sessions / "2026-10-13_1538_lena.mp3").write_bytes(b"\xff\xf3" + b"\x00" * 200)
    app = create_app(paths, PORT, engine_factory=lambda emit: FakeEngine(emit))
    with TestClient(app, base_url=BASE) as c:
        listed = c.get("/api/sessions").json()
        assert listed[0]["name"] == "2026-10-13_1538_lena" and listed[0]["character"] == "lena"
        data = c.get("/api/sessions/2026-10-13_1538_lena").json()
        assert data["audio"] and [e["role"] for e in data["entries"]] == ["assistant", "user"]
        assert data["entries"][1]["duration"] == 1.2
        r = c.get("/api/sessions/2026-10-13_1538_lena.mp3", headers={"range": "bytes=0-9"})
        assert r.status_code == 206 and len(r.content) == 10
        assert c.get("/api/sessions/..%2Fsettings").status_code == 404
        assert c.get("/api/sessions/missing.mp3").status_code == 404
        foreign = {"sec-fetch-site": "cross-site"}
        assert c.get("/api/sessions/2026-10-13_1538_lena.mp3", headers=foreign).status_code == 403
        assert c.get("/api/sessions", headers=foreign).status_code == 403
