@echo off
setlocal
cd /d %~dp0
conda run -n langgraph-react-agent python -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
