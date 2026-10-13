@echo off
rem ---------------------------------------------------------------------------
rem Sets up the portable environment for Local Lantern Talk.
rem Call it from the other .bat files:  call "%~dp0scripts\env.bat"
rem Every cache and temp folder is redirected into the program folder so that
rem nothing is written to %USERPROFILE% or %APPDATA%.
rem ---------------------------------------------------------------------------

for %%I in ("%~dp0..") do set "LLT_ROOT=%%~fI"

set "LLT_RUNTIME=%LLT_ROOT%\runtime"
set "LLT_CACHE=%LLT_RUNTIME%\cache"

rem uv and the portable Python
set "UV_INSTALL_DIR=%LLT_RUNTIME%\uv"
set "UV_CACHE_DIR=%LLT_CACHE%\uv"
set "UV_PYTHON_INSTALL_DIR=%LLT_RUNTIME%\python"
set "UV_PYTHON_PREFERENCE=only-managed"
set "UV_PROJECT_ENVIRONMENT=%LLT_RUNTIME%\venv"
set "UV_LINK_MODE=copy"
set "UV_NO_CONFIG=1"
set "UV_TOOL_DIR=%LLT_RUNTIME%\uv-tools"
set "UV_PYTHON_BIN_DIR=%LLT_RUNTIME%\uv-bin"

rem Library caches
set "HF_HOME=%LLT_CACHE%\huggingface"
set "HF_HUB_CACHE=%LLT_CACHE%\huggingface\hub"
set "HF_XET_CACHE=%LLT_CACHE%\huggingface\xet"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "TORCH_HOME=%LLT_CACHE%\torch"
set "XDG_CACHE_HOME=%LLT_CACHE%"
set "PIP_CACHE_DIR=%LLT_CACHE%\pip"
set "NUMBA_CACHE_DIR=%LLT_CACHE%\numba"
set "MPLCONFIGDIR=%LLT_CACHE%\matplotlib"
set "TRITON_CACHE_DIR=%LLT_CACHE%\triton"
set "CUDA_CACHE_PATH=%LLT_CACHE%\nv_compute"
rem Kernels JIT-compiled for newer GPUs (e.g. the CUDA 12 llama.cpp build on RTX 50)
rem are cached here; the default 256 MB limit is too small to keep them all.
set "CUDA_CACHE_MAXSIZE=4294967296"
set "GRADIO_TEMP_DIR=%LLT_ROOT%\tmp\gradio"
set "GRADIO_ANALYTICS_ENABLED=False"
set "TMP=%LLT_ROOT%\tmp"
set "TEMP=%LLT_ROOT%\tmp"

rem Python
set "PYTHONPATH=%LLT_ROOT%\app"
set "PYTHONUTF8=1"

if not exist "%LLT_ROOT%\tmp" mkdir "%LLT_ROOT%\tmp"
if not exist "%LLT_CACHE%" mkdir "%LLT_CACHE%"

set "PATH=%UV_INSTALL_DIR%;%LLT_RUNTIME%\venv\Scripts;%PATH%"
exit /b 0
