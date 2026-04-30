@echo off
setlocal
cd /d %~dp0
conda run -n langgraph-react-agent python -m unittest discover -s tests
