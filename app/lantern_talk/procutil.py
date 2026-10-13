"""Child process lifetime: child processes never outlive the app.

On Windows every child is assigned to a Job Object with
``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``. The job handle is only held by this
process, so when it exits for any reason — normal exit, crash, closed console
window, Task Manager — Windows kills the children too. Without this, a closed
console window leaves ``llama-server.exe`` running with several GB of VRAM.
"""

from __future__ import annotations

import ctypes
import logging
import subprocess
import sys
from ctypes import wintypes

log = logging.getLogger(__name__)

_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
        ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


_job_handle: int | None = None


def _kill_on_close_job() -> int | None:
    """Create (once) the job object; the handle stays open until this process exits."""
    global _job_handle
    if _job_handle is not None:
        return _job_handle
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        log.warning("CreateJobObject failed (error %d)", ctypes.get_last_error())
        return None
    info = _ExtendedLimitInformation()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS, ctypes.byref(info), ctypes.sizeof(info)
    ):
        log.warning("SetInformationJobObject failed (error %d)", ctypes.get_last_error())
        kernel32.CloseHandle(job)
        return None
    _job_handle = job
    return job


def tie_to_this_process(proc: subprocess.Popen) -> bool:  # type: ignore[type-arg]
    """Make ``proc`` die together with the current process. Returns False if not possible."""
    if sys.platform != "win32":
        return False
    job = _kill_on_close_job()
    if job is None:
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    handle = int(proc._handle)  # type: ignore[attr-defined]  # Popen's process handle on Windows
    if not kernel32.AssignProcessToJobObject(job, handle):
        log.warning("AssignProcessToJobObject failed (error %d)", ctypes.get_last_error())
        return False
    return True
