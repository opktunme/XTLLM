@echo off
rem Backward-compatible alias. New commands and documentation use xtllm.cmd.
call "%~dp0xtllm.cmd" %*
exit /b %ERRORLEVEL%
