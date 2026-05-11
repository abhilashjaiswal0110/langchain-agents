# Development Setup Guide

> **Last Updated**: 2026-05-11
> **Version**: 2.2.0 — Azure OpenAI primary, Python 3.12 recommended, LangSmith key verification
> **Audience**: Developers, Contributors

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Development Environment](#development-environment)
4. [Project Structure](#project-structure)
5. [Running the Server](#running-the-server)
6. [LangSmith Tracing Setup](#langsmith-tracing-setup)
7. [Testing](#testing)
8. [Code Style](#code-style)
9. [Adding New Agents](#adding-new-agents)
10. [Troubleshooting](#troubleshooting)
11. [Contributing](#contributing)

---

## Prerequisites

### Required Software

| Software | Version | Notes |
|----------|---------|-------|
| Python | **3.12** (3.10–3.12 supported) | 3.13 has a Windows Store restriction with `uv` — use 3.12 |
| Git | Latest | Version control |
| Docker | 24.0+ | Containerization (optional) |
| `uv` | Latest | `pip install uv` — fast package manager |

### Required API Keys

| Service | Required | Description | Get Key |
|---------|----------|-------------|---------|
| Azure OpenAI | **Yes*** | Primary LLM backend (`o4-mini`) | Azure Portal |
| OpenAI | No | Fallback if no Azure | https://platform.openai.com |
| Anthropic | No | Fallback/alternative | https://console.anthropic.com |
| LangSmith | No | Observability tracing (verified on startup) | https://smith.langchain.com |
| Tavily | No | Web search tool | https://tavily.com |

*At least one LLM provider (Azure OpenAI **or** OpenAI) is required.

---

## Quick Start

```bash
# 1. Navigate to the deployment folder
cd deployment

# 2. Copy environment template
cp .env.example .env
# Edit .env — minimum required:
#   AZURE_OPENAI_API_KEY=...
#   AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/

# 3. Start the server (using existing .venv)
.venv/Scripts/python -m uvicorn app.server:app --host 0.0.0.0 --port 8000

# Or with uv pointing to Python 3.12 explicitly (Windows)
uv run --python "C:/Python312/python.exe" uvicorn app.server:app --host 0.0.0.0 --port 8000

# Or with make
make run-reload

# 4. Open browser
# http://localhost:8000/chat   - Web UI (all 16 agents)
# http://localhost:8000/docs   - Swagger API reference
# http://localhost:8000/health - Health check
```

---

## LangSmith Tracing Setup

The server **verifies your LangSmith API key at startup** before enabling tracing. This prevents the endless 403 error flood that occurs with expired keys.

### If you see the verification failure message

```
[LangSmith] ⚠  API key verification FAILED (403 Forbidden).
  → Visit https://smith.langchain.com to generate a new key.
  → Set LANGCHAIN_API_KEY in deployment/.env and restart.
  → Tracing is now DISABLED to prevent log flooding.
```

**Fix**: Generate a fresh key at https://smith.langchain.com → Settings → API Keys → Create API Key.

### Enabling tracing with a valid key

```env
# deployment/.env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_sk_<your_new_key>
LANGSMITH_API_KEY=lsv2_sk_<your_new_key>
LANGCHAIN_PROJECT=langchain-platform
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

Restart the server. On success you'll see:

```
[LangSmith] ✓  Tracing enabled → project: 'langchain-platform' @ https://api.smith.langchain.com
```

All agent invocations will then appear in your LangSmith project dashboard.

---

## Development Environment

### Option A: Virtual Environment (pip)

```bash
cd deployment

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

### Option B: uv (Recommended)

```bash
cd deployment

# Create environment and install
uv venv
uv pip install -e ".[dev]"

# Activate
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### Option C: Docker Development

```bash
cd deployment

# Build development image
docker compose -f docker-compose.dev.yml build

# Run with live reload
docker compose -f docker-compose.dev.yml up
```

### IDE Setup (VSCode)

Recommended extensions:
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- REST Client (humao.rest-client)
- Docker (ms-azuretools.vscode-docker)

Settings (`.vscode/settings.json`):
```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.analysis.typeCheckingMode": "basic",
  "editor.formatOnSave": true,
  "python.formatting.provider": "black"
}
```

---

## Project Structure

```
deployment/
├── app/                          # Application source
│   ├── __init__.py
│   ├── server.py                 # FastAPI application
│   ├── agents/                   # Enterprise agents
│   │   ├── __init__.py
│   │   ├── base/                 # Base agent classes
│   │   │   ├── agent_base.py     # Abstract base agent
│   │   │   └── tools.py          # Shared tool utilities
│   │   ├── research/             # Research agent
│   │   ├── content/              # Content agent (HITL)
│   │   ├── data_analyst/         # Data analyst agent
│   │   ├── documents/            # Document agent
│   │   ├── rag/                  # RAG agent
│   │   ├── it_support/           # HITL IT support
│   │   ├── code_assistant/       # Code assistant
│   │   └── evals/                # Evaluation framework
│   ├── chains/                   # LangChain chains
│   └── static/                   # Web UI assets
├── tests/                        # Test suite
│   ├── conftest.py               # Pytest fixtures
│   ├── test_server.py            # API tests
│   └── test_evaluators.py        # Evaluator tests
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md
│   ├── DEPLOYMENT.md
│   ├── OPERATIONS.md
│   ├── SECURITY.md
│   └── SETUP.md
├── .env.example                  # Environment template
├── Dockerfile                    # Production image
├── docker-compose.yml            # Container config
├── pyproject.toml                # Dependencies
├── Makefile                      # Dev commands
└── README.md                     # Project overview
```

---

## Running the Server

### Development Mode (with reload)

```bash
# Linux / macOS
python -m uvicorn app.server:app --reload --host 0.0.0.0 --port 8000

# Windows (use deployment .venv directly — avoids uv VIRTUAL_ENV conflict)
.venv\Scripts\uvicorn.exe app.server:app --host 0.0.0.0 --port 8000 --reload
```

### Production Mode

```bash
# Linux / macOS
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000

# Windows (recommended — bypass uv sync errors)
.venv\Scripts\uvicorn.exe app.server:app --host 0.0.0.0 --port 8000
```

> **Windows Note**: `uv run uvicorn ...` may fail if the root-level monorepo `.venv`
> is activated, causing a `VIRTUAL_ENV` conflict and `Access denied` errors on
> `langgraph_cli` dist-info. Use `.venv\Scripts\uvicorn.exe` directly from the
> `deployment/` directory instead.

### Docker Mode

```bash
docker compose up -d
docker compose logs -f
```

### LangGraph Studio (Visual Development)

```bash
# Windows (from deployment/ directory)
.venv\Scripts\python.exe -m langgraph_cli dev --port 2024 --allow-blocking

# Access: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
# Or use: .\start_studio.ps1
```

### Verify Server

```bash
# Health check — all flags should be true when API keys are configured
curl http://localhost:8000/health

# List all enterprise agents
curl http://localhost:8000/api/enterprise/agents

# List IT Support agents (it_helpdesk, servicenow, document_intelligence, employee_experience)
curl http://localhost:8000/api/agents

# List Deep Agent subagents
curl http://localhost:8000/api/deepagent/subagents

# Test Enterprise Research Agent (live query)
curl -X POST http://localhost:8000/api/enterprise/research/invoke \
  -H "Content-Type: application/json" \
  -d '{"query": "What is LangChain?"}'

# Test IT Operations Deep Agent
curl -X POST http://localhost:8000/api/deepagent/start \
  -H "Content-Type: application/json" \
  -d '{"user_id": "ops_user"}'
# → returns session_id, then use it:
curl -X POST http://localhost:8000/api/deepagent/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<session_id>", "message": "Show all open P1 incidents"}'

# Test IT Helpdesk Agent
curl -X POST http://localhost:8000/api/conversation/start \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "it_helpdesk"}'
```

---

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html

# Specific test file
pytest tests/test_server.py -v

# Specific test
pytest tests/test_server.py::test_health_endpoint -v
```

### Test Structure

```python
# tests/test_server.py
import pytest
from fastapi.testclient import TestClient
from app.server import app

client = TestClient(app)

def test_health_endpoint():
    """Test health check returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_agents_list():
    """Test agent listing endpoint."""
    response = client.get("/api/enterprise/agents")
    assert response.status_code == 200
    assert "agents" in response.json()
```

### Writing Tests

Follow these patterns:
1. Use `TestClient` for API tests
2. Mock external services (LLM, Tavily)
3. Use fixtures for common setup
4. Test happy path and error cases

```python
# Example with mock
from unittest.mock import patch

def test_research_agent_with_mock():
    """Test research agent with mocked LLM."""
    with patch("app.agents.research.research_agent.ChatOpenAI") as mock_llm:
        mock_llm.return_value.invoke.return_value = "Mocked response"
        # Test logic here
```

---

## Code Style

### Formatting

```bash
# Format with black
black app/ tests/

# Check formatting
black --check app/ tests/
```

### Linting

```bash
# Lint with ruff
ruff check app/ tests/

# Fix auto-fixable issues
ruff check --fix app/ tests/
```

### Type Checking

```bash
# Type check with mypy
mypy app/
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Style Guidelines

1. **Type hints required** for all functions
   ```python
   def process_message(message: str, session_id: str | None = None) -> dict[str, Any]:
   ```

2. **Google-style docstrings**
   ```python
   def create_ticket(title: str, description: str) -> str:
       """Create a support ticket.

       Args:
           title: Ticket title.
           description: Detailed description.

       Returns:
           Ticket ID string.

       Raises:
           ValueError: If title is empty.
       """
   ```

3. **Pydantic models for data**
   ```python
   class AgentRequest(BaseModel):
       message: str
       session_id: str | None = None
   ```

---

## Adding New Agents

### Step 1: Create Agent Module

```bash
mkdir app/agents/my_agent
touch app/agents/my_agent/__init__.py
touch app/agents/my_agent/my_agent.py
```

### Step 2: Implement Agent

```python
# app/agents/my_agent/my_agent.py
from typing import Any
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from app.agents.base.agent_base import BaseAgent, AgentConfig


class MyAgentState(BaseModel):
    """State for MyAgent."""
    messages: list = Field(default_factory=list)
    session_id: str | None = None


class MyAgent(BaseAgent):
    """Custom agent implementation."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        super().__init__(config)
        self.register_tools([
            # Add your tools here
        ])

    def _get_system_prompt(self) -> str:
        return """You are a helpful assistant."""

    def _build_graph(self) -> StateGraph:
        def call_model(state: MyAgentState) -> dict:
            system = SystemMessage(content=self._get_system_prompt())
            messages = [system] + list(state.messages)
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def should_continue(state: MyAgentState) -> str:
            messages = list(state.messages)
            if not messages:
                return "end"
            last = messages[-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return "end"

        graph = StateGraph(MyAgentState)
        graph.add_node("agent", call_model)
        if self._tools:
            graph.add_node("tools", ToolNode(self._tools))
            graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
            graph.add_edge("tools", "agent")
        else:
            graph.add_edge("agent", END)
        graph.add_edge(START, "agent")

        return graph
```

### Step 3: Export Agent

```python
# app/agents/my_agent/__init__.py
from app.agents.my_agent.my_agent import MyAgent, MyAgentState

__all__ = ["MyAgent", "MyAgentState"]
```

### Step 4: Add to Server

```python
# app/server.py
from app.agents.my_agent import MyAgent

# In load_enterprise_agents():
my_agent = MyAgent()
status["my_agent"] = True
```

### Step 5: Add Endpoint

```python
# app/server.py
@app.post("/api/enterprise/my-agent/invoke")
async def my_agent_invoke(request: EnterpriseAgentRequest):
    if my_agent is None:
        raise HTTPException(503, "Agent not available")
    result = my_agent.invoke(message=request.message)
    return EnterpriseAgentResponse(success=True, response=result.get("output"))
```

### Step 6: Add Tests

```python
# tests/test_my_agent.py
def test_my_agent_invoke():
    response = client.post("/api/enterprise/my-agent/invoke",
        json={"message": "Test query"})
    assert response.status_code == 200
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Import errors | Missing dependencies | `pip install -e ".[dev]"` |
| Server won't start | Port in use | Kill process on 8000 |
| Agents not loading | No API keys | Check `.env` file |
| Tests failing | Outdated mocks | Update test fixtures |
| `uv run uvicorn` fails (Windows) | `VIRTUAL_ENV` conflict with root `.venv` | Use `.venv\Scripts\uvicorn.exe` directly |
| `Access denied` on `langgraph_cli` | Windows file lock on dist-info | Run as Administrator or restart terminal |
| `No module named uvicorn` | Wrong Python / `.venv` | Ensure you are in `deployment/` directory |

### Windows Startup Issues

**Symptom**: `uv run uvicorn app.server:app` fails with one of:
- `VIRTUAL_ENV=...\.venv does not match the project environment path`
- `error: failed to remove directory ... Access is denied. (os error 5)`
- `No module named uvicorn`

**Root cause**: The monorepo root `.venv` is activated in the shell but `uv` detects
the deployment-specific `.venv` and tries to re-sync packages, hitting permission errors.

**Solution**: Use the deployment `.venv` executable directly:
```bash
# From deployment/ directory:
.venv\Scripts\uvicorn.exe app.server:app --host 0.0.0.0 --port 8000

# Or if using bash/git-bash on Windows:
.venv/Scripts/uvicorn.exe app.server:app --host 0.0.0.0 --port 8000
```

**Fix incomplete installs** (run from `deployment/` in an Administrator terminal):
```bash
.venv\Scripts\pip install --force-reinstall langchain-core langgraph
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Dependencies

```bash
# Check deployment .venv packages (Windows)
.venv\Scripts\pip list | grep langchain
.venv\Scripts\pip list | grep langgraph

# Linux/macOS
.venv/bin/pip list | grep langchain
```

---

## Contributing

### Workflow

1. **Fork** the repository
2. **Create branch**: `git checkout -b feature/my-feature`
3. **Make changes** following code style
4. **Write tests** for new functionality
5. **Run checks**: `make lint && make test`
6. **Commit**: `git commit -m "feat: add my feature"`
7. **Push**: `git push origin feature/my-feature`
8. **Create PR** with description

### Commit Messages

Follow Conventional Commits:

```
feat: add new agent for X
fix: resolve memory leak in Y
docs: update setup guide
test: add tests for Z
refactor: simplify agent base class
chore: update dependencies
```

### PR Guidelines

- Clear description of changes
- Link to related issues
- Tests passing
- Documentation updated
- Screenshots for UI changes

---

## Related Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md) - System design
- [DEPLOYMENT.md](./DEPLOYMENT.md) - Deployment guide
- [SECURITY.md](./SECURITY.md) - Security guidelines
