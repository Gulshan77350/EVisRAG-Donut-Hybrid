@echo off
REM Double-click to start the Donut -> EVisRAG demo, then open http://127.0.0.1:7860
cd /d "%~dp0"
start "" /b cmd /c "timeout /t 60 >nul && start http://127.0.0.1:7860"
env\python.exe app.py %*
pause
