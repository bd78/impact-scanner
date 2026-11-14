@echo off
rem Author: bd78
rem Starts the target analyser. Runs setup.bat first if the venv does not exist yet.
rem Usage:  run.bat                       (database: targets.db next to this script)
rem         run.bat C:\path\to\other.db   (use another database file)
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    call "%~dp0setup.bat"
    if errorlevel 1 exit /b %errorlevel%
)

"%~dp0.venv\Scripts\python.exe" -m target_analyzer %*
