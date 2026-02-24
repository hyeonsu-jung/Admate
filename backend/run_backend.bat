@echo off
echo Starting Admate RAG Chatbot Backend...
cd /d %~dp0
call .venv\Scripts\activate
python -m app.main
pause
