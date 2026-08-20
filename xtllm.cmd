@echo off
setlocal
set "XTLLM_ROOT=%~dp0"
if exist "%XTLLM_ROOT%.venv\Scripts\python.exe" (
  "%XTLLM_ROOT%.venv\Scripts\python.exe" "%XTLLM_ROOT%xtllm.py" %*
) else (
  py -3 "%XTLLM_ROOT%xtllm.py" %*
)
exit /b %ERRORLEVEL%
