@echo off
setlocal
title Local Lantern Talk - Console (M1)
call "%~dp0scripts\env.bat"
if not exist "%LLT_RUNTIME%\venv\Scripts\python.exe" (
    echo The runtime is not installed yet. Run install.bat first.
    pause
    exit /b 1
)
"%LLT_RUNTIME%\venv\Scripts\python.exe" -m lantern_talk.console %*
pause
