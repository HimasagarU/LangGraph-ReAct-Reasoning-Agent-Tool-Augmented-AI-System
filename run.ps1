Set-Location $PSScriptRoot
conda run -n langgraph-react-agent python -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
