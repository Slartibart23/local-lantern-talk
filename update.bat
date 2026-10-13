@echo off
setlocal
title Local Lantern Talk - Update
call "%~dp0scripts\env.bat"
rem Updates the app code and its dependencies. Never touches models\ or user_data\.
rem Options: -Yes (no questions), -NoSelfTest
set "PSModulePath="
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\update.ps1" %*
set "RESULT=%ERRORLEVEL%"
if "%RESULT%"=="1" (
    echo.
    echo The update did not finish. See the messages above.
)
pause
exit /b %RESULT%
