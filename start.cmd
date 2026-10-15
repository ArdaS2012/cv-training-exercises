@echo off
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo Install uv ^(see README.md^), reopen the terminal, and try again.
  pause
  exit /b 1
)
uv run --locked python start.py %*
if errorlevel 1 pause
