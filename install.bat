@echo off
setlocal
title Local Lantern Talk - Install
call "%~dp0scripts\env.bat"
rem Options: -Yes (no questions), -NoStarterModel, -NoMicTest
rem Started from a PowerShell 7 terminal, Windows PowerShell would inherit PowerShell 7's
rem module path and miss its own built-in commands; let it use its defaults.
set "PSModulePath="
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1" %*
set "RESULT=%ERRORLEVEL%"
if "%RESULT%"=="1" (
    echo.
    echo Installation failed. See the messages above.
    echo Re-running install.bat resumes where it stopped.
    echo Need help? Copy the messages above into Claude at https://claude.ai and ask what to do.
)
pause
exit /b %RESULT%
