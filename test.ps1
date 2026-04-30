Set-Location $PSScriptRoot
conda run -n langgraph-react-agent python -m unittest discover -s tests
