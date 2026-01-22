# LangGraph CLI Setup Guide

## Current Status ✅

Your LangGraph environment is configured and ready to use with all your agents:

### Configured Agents in `langgraph.json`:
1. ✅ `servicenow_agent` - ServiceNow integration
2. ✅ `document_agent` - Document processing
3. ✅ `it_helpdesk` - IT helpdesk operations
4. ✅ **`it_operations_agent`** - IT Operations Deep Agent (NEW)
5. ✅ **`sales_intelligence_agent`** - Sales & Pre-Sales Intelligence Deep Agent (NEW)

## Environment Setup

- **Python Environment**: `.venv` in deployment folder (Python 3.11)
- **Package Manager**: `uv` (fast Python package manager)
- **LangGraph CLI**: v0.1.54 installed in `.venv/Scripts/langgraph.exe`
- **Configuration**: `deployment/langgraph.json`

## Running LangGraph Locally

### Option 1: Using Existing FastAPI Server (RECOMMENDED)

Your current FastAPI server (`deployment/app/server.py`) already works and includes all agents:

```powershell
cd deployment
uvicorn app.server:app --reload --port 8000
```

**Access at**: http://localhost:8000
- UI: http://localhost:8000/static/chat.html
- API Docs: http://localhost:8000/docs

### Option 2: LangGraph Studio (Visual Development)

**Note**: Requires newer `langgraph-cli` with `dev` command. Currently installed version (0.1.54) only has `up` command.

#### Start Development Server (No Docker Required):
```powershell
cd deployment
.\.venv\Scripts\langgraph.exe dev --port 2024
```

This will:
- Start a lightweight dev server on port 2024
- Enable hot reloading
- Provide Studio UI for visual debugging
- Use in-memory state (no PostgreSQL needed)

#### Start Production-like Server (Docker Required):
```powershell
# Make sure Docker Desktop is running first!
cd deployment
.\.venv\Scripts\langgraph.exe up --port 8123
```

This will:
- Start production-like environment in Docker
- Use PostgreSQL for state persistence
- Use Redis for caching
- Run on port 8123

### Option 3: Using `uv run` Command

```powershell
cd deployment
uv run uvicorn app.server:app --reload --port 8000
```

## Testing Agents

### Test IT Operations Agent:
```python
from app.deepagents.it_operations_agent import create_it_operations_agent

agent = create_it_operations_agent()
response = agent.chat(
    message="Show me recent P1 incidents",
    session_id="test-session-001"
)
print(response)
```

### Test Sales Intelligence Agent:
```python
from app.deepagents.sales_intelligence_agent import create_sales_intelligence_agent

agent = create_sales_intelligence_agent()
response = agent.chat(
    message="Analyze deal pipeline for Q1 2026",
    session_id="test-session-002"
)
print(response)
```

## Current Limitations & Solutions

### Issue 1: Docker Not Running
**Error**: `Error: Docker not installed or not running`

**Solution**:
- Start Docker Desktop, OR
- Use `langgraph dev` (requires upgrade), OR
- Use your existing FastAPI server (recommended)

### Issue 2: `langgraph dev` Not Available
**Current Version**: 0.1.54 (only has `up` command)
**Needed Version**: >= 0.4.x (has `dev` command)

**Solution**: Upgrade when ready:
```powershell
uv add "langgraph-cli[inmem]>=0.4.0"
```

### Issue 3: Cache Issues on Windows
If you see hardlink errors:

```powershell
# Clear uv cache
uv cache clean
# Then reinstall
uv sync
```

## Next Steps

1. **For Development**: Use your existing FastAPI server (already working)
2. **For Visual Debugging**:
   - Upgrade langgraph-cli to get `dev` command
   - Or start Docker and use `langgraph up`
3. **For Production**: Use Docker deployment with `langgraph up`

## Files Modified

- ✅ `deployment/langgraph.json` - Added Deep Agents
- ✅ `deployment/pyproject.toml` - Added langgraph-cli dependency
- ✅ `deployment/uv.lock` - Locked dependencies

## API Endpoints (FastAPI Server)

Your existing server exposes these endpoints:

```
POST /deep-agent/it-operations/chat        - IT Operations chat
POST /deep-agent/it-operations/stream      - IT Operations streaming
POST /deep-agent/sales-intelligence/chat   - Sales Intelligence chat
POST /deep-agent/sales-intelligence/stream - Sales Intelligence streaming
```

## Resources

- [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/)
- [LangGraph CLI Reference](https://docs.langchain.com/oss/python/langgraph/cli)
- [Studio Setup](https://docs.langchain.com/oss/python/langgraph/studio)
