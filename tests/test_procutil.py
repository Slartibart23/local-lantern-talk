"""Child processes must die with the app, even when it is killed hard (closed console window)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows job objects")

APP = str(Path(__file__).parents[1] / "app")

PARENT = """
import subprocess, sys, time
from lantern_talk.procutil import tie_to_this_process
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
print(child.pid, tie_to_this_process(child), flush=True)
time.sleep(120)
"""


def _alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


def test_child_dies_when_parent_is_killed() -> None:
    env = {**os.environ, "PYTHONPATH": APP}
    parent = subprocess.Popen([sys.executable, "-c", PARENT], stdout=subprocess.PIPE, text=True, env=env)
    try:
        line = parent.stdout.readline().split()  # type: ignore[union-attr]
        child_pid, tied = int(line[0]), line[1] == "True"
        assert tied
        assert _alive(child_pid)
        parent.kill()  # TerminateProcess: no cleanup code runs, like closing the console window
        parent.wait()
        deadline = time.monotonic() + 5
        while _alive(child_pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not _alive(child_pid)
    finally:
        if parent.poll() is None:
            parent.kill()
