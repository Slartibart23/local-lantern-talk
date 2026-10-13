@echo off
rem Installs the repository's Git hooks (blocks commits of personal data / binaries).
git -C "%~dp0.." config core.hooksPath scripts/git-hooks
if errorlevel 1 (echo Could not install the Git hooks. & exit /b 1)
echo Git hooks installed.
