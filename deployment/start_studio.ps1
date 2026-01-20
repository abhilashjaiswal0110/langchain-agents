# Start LangGraph Studio UI
$ErrorActionPreference = "Stop"

Write-Host "Starting LangGraph Studio UI..." -ForegroundColor Green
Write-Host ""
Write-Host "Endpoints:" -ForegroundColor Cyan
Write-Host "  API:       http://127.0.0.1:2024" -ForegroundColor White
Write-Host "  Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024" -ForegroundColor White
Write-Host "  API Docs:  http://127.0.0.1:2024/docs" -ForegroundColor White
Write-Host ""

# Run langgraph dev with blocking allowed (for development)
& "$PSScriptRoot\.venv\Scripts\python.exe" -m langgraph_cli dev --port 2024 --allow-blocking
