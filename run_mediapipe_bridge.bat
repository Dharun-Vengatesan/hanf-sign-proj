@echo off
echo =======================================================
echo   Starting MediaPipe Hands Real-Time UDP Bridge
echo =======================================================

set PYTHON_EXE=

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
) else if exist "C:\Python311\python.exe" (
    set "PYTHON_EXE=C:\Python311\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo Using Python: %PYTHON_EXE%
"%PYTHON_EXE%" "%~dp0python\mediapipe_bridge.py" --preview
pause
