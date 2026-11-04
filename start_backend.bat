@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
uvicorn api:app --host 127.0.0.1 --port 8000
