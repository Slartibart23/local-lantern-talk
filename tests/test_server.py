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
    assert client.get("/static/app.js").status_code == 200
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


def test_second_tab_takes_over(client: TestClient) -> None:
    with client.websocket_connect(WS_URL, headers=GOOD_ORIGIN) as first:
        first.receive_json()
        with client.websocket_connect(WS_URL, headers=GOOD_ORIGIN) as second:
            assert second.receive_json()["type"] == "hello"
            assert first.receive_json()["type"] == "replaced"
