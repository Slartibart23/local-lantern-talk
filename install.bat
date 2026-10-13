@echo off
setlocal
title Local Lantern Talk - Install
call "%~dp0scripts\env.bat"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1"
if errorlevel 1 (
    echo.
    echo Installation failed. See the messages above. Re-running install.bat resumes where it stopped.
    pause
    exit /b 1
)
pause
