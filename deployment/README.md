# Enterprise AI Agents Platform

> **Version**: 2.0 — Production-certified May 2026
> **LLM Backend**: Azure OpenAI (`o4-mini` reasoning model + `text-embedding-3-small`)
> **Status**: All 16 agents live ✅

A production-ready platform serving 16 LangChain/LangGraph agents as REST APIs, with a full Web UI, streaming support, LangSmith tracing, and Azure OpenAI as the primary backend.

---

## What's New (May 2026)

- **Deep Agents Framework** — IT Operations, Sales Intelligence, and Recruitment agents with planning, subagent delegation, and session-scoped file system
- **Software Development Deep Agent** — 54+ SDLC tools including secure bash execution, code generation, and Azure cloud integration
- **Domain Agents** — 8 business-line agents (MarCom, HR, L&D, PreSales, Datacenter, Cloud, Cybersecurity, Data & AI)
- **Azure OpenAI primary** — all agents default to `o4-mini` reasoning model; OpenAI/Anthropic remain as fallbacks
- **LangSmith key verification on startup** — invalid/expired keys are detected immediately with a clear fix message; no more log flooding
- **Response cache** — opt-in via `CACHE_ENABLED=true`
- **Prometheus metrics** at `/metrics`

---

## Agent Inventory

### Deep Agents (planning + subagents + session files)

| Agent | Start Endpoint | Chat Endpoint | Subagents |
|-------|---------------|---------------|-----------|
| IT Operations | `POST /api/deepagent/start` | `POST /api/deepagent/chat/stream` | Incident, Change, Problem, Asset, SLA, Knowledge |
| Sales Intelligence | `POST /api/sales-agent/start` | `POST /api/sales-agent/chat/stream` | Deal, RFP, Pricing, Competitive |
| Recruitment | `POST /api/recruitment-agent/start` | `POST /api/recruitment-agent/chat/stream` | Resume (L1/L2/L3), Interview, Scoring |
| Software Dev | `POST /api/software-dev-agent/start` | `POST /api/software-dev-agent/chat/stream` | CodeGen, Reviewer, Tester, Architect |

### IT Support Agents (conversation memory, session-based)

| Agent | Type | Start via |
|-------|------|-----------|
| IT Helpdesk | LangGraph + MemorySaver | `POST /api/conversation/start` (`agent_type: it_helpdesk`) |
| ServiceNow ITSM | LangGraph + 10 ITSM tools | `POST /api/conversation/start` (`agent_type: servicenow`) |
| Document Intelligence | LangGraph + OCR/PDF | `POST /api/conversation/start` (`agent_type: document_intelligence`) |
| Employee Experience | LangGraph | `POST /api/conversation/start` (`agent_type: employee_experience`) |

### Enterprise Agents (stateless invoke/stream)

| Agent | Endpoint |
|-------|----------|
| AI Research | `POST /api/enterprise/research/invoke` |
| Content Generation | `POST /api/enterprise/content/invoke` |
| Data Analyst | `POST /api/enterprise/data-analyst/invoke` |
| Document Generator | `POST /api/enterprise/documents/invoke` |
| Multilingual RAG | `POST /api/enterprise/rag/invoke` |
| HITL IT Support | `POST /api/enterprise/support/invoke` |
| Code Assistant | `POST /api/enterprise/code/invoke` |
| Document Intelligence | `POST /api/enterprise/document-intelligence/invoke` |

### Domain Agents (business line)

All exposed under `/api/domain/<domain>/invoke` — marcom, hr, lnd, presales, datacenter, cloud, cybersecurity, data_ai.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    FastAPI Application (server.py)                    │
│                                                                        │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────────────────┐  │
│  │ LangServe    │  │ Deep Agents    │  │ IT Support / Enterprise  │  │
│  │ /chat /rag   │  │ IT Ops · Sales │  │ /api/conversation/       │  │
│  │ /agent       │  │ Recruitment    │  │ /api/enterprise/         │  │
│  │ /langgraph   │  │ Software Dev   │  │ /api/domain/             │  │
│  └──────────────┘  └────────────────┘  └──────────────────────────┘  │
│                              │                                         │
│  ┌───────────────────────────┴───────────────────────────────────┐    │
│  │                     LLM Factory                               │    │
│  │  Azure OpenAI o4-mini (primary) · OpenAI · Anthropic         │    │
│  └───────────────────────────────────────────────────────────────┘    │
│                              │                                         │
│  ┌───────────────────────────┴───────────────────────────────────┐    │
│  │   Auth · Cache · Governance · Prometheus · LangSmith Tracing  │    │
│  └───────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.10–3.12 (3.12 recommended; 3.13 has a known Windows Store restriction with uv)
- `uv` package manager (`pip install uv`)
- Azure OpenAI resource **or** OpenAI/Anthropic API key

### 1. Environment setup

```bash
cd deployment
cp .env.example .env
# Edit .env — minimum required:
#   AZURE_OPENAI_API_KEY=...
#   AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
#   AZURE_OPENAI_DEPLOYMENT_NAME=o4-mini   (or gpt-4o)
```

### 2. Start the server

```bash
# Recommended — uses the existing .venv if present
.venv/Scripts/python -m uvicorn app.server:app --host 0.0.0.0 --port 8000

# Or with uv (points explicitly to Python 3.12 to avoid Windows Store Python)
uv run --python "C:/Python312/python.exe" uvicorn app.server:app --host 0.0.0.0 --port 8000

# Or with make
make run-reload
```

### 3. Access the platform

| URL | Description |
|-----|-------------|
| http://localhost:8000/chat | Web UI — all agents |
| http://localhost:8000/docs | Swagger API reference |
| http://localhost:8000/health | Health check (JSON) |
| http://localhost:8000/metrics | Prometheus metrics |

---

## LangSmith Tracing

On startup the server **verifies your API key** before enabling tracing. If the key is expired or invalid you'll see:

```
[LangSmith] ⚠  API key verification FAILED (403 Forbidden).
  → Visit https://smith.langchain.com to generate a new key.
  → Set LANGCHAIN_API_KEY in deployment/.env and restart.
  → Tracing is now DISABLED to prevent log flooding.
```

To enable tracing with a valid key:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_sk_<your_valid_key>
LANGSMITH_API_KEY=lsv2_sk_<your_valid_key>
LANGCHAIN_PROJECT=langchain-platform
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

Restart the server — a successful verification prints:

```
[LangSmith] ✓  Tracing enabled → project: 'langchain-platform' @ https://api.smith.langchain.com
```

---

## API Reference (key endpoints)

### Conversation (IT Support agents)

```bash
# Start a session
curl -X POST http://localhost:8000/api/conversation/start \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "it_helpdesk"}'
# → {"session_id": "abc123", ...}

# Chat in session
curl -X POST http://localhost:8000/api/conversation/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "message": "I cannot connect to VPN"}'
```

### Deep Agent (IT Operations)

```bash
# Start
curl -X POST http://localhost:8000/api/deepagent/start \
  -d '{}'
# → {"session_id": "..."}

# Stream chat
curl -X POST http://localhost:8000/api/deepagent/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id": "...", "message": "Check SLA breach risk for open P1 incidents"}'
```

### Enterprise agent (invoke)

```bash
curl -X POST http://localhost:8000/api/enterprise/research/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"input": "Latest trends in agentic AI 2026"}}'
```

### Webhook (external integrations — Copilot Studio, Azure AI)

```bash
curl -X POST http://localhost:8000/api/webhook/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Reset my password", "session_id": "optional-session"}'
```

---

## Configuration Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_OPENAI_API_KEY` | Yes* | — | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Yes* | — | `https://<resource>.openai.azure.com/` |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | No | `o4-mini` | Chat model deployment name |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | No | `text-embedding-3-small` | Embedding deployment |
| `OPENAI_API_KEY` | No | — | OpenAI fallback |
| `ANTHROPIC_API_KEY` | No | — | Anthropic fallback |
| `LANGCHAIN_TRACING_V2` | No | `true` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | — | LangSmith API key (verified on startup) |
| `LANGCHAIN_PROJECT` | No | `langchain-platform` | LangSmith project |
| `API_KEY_ENABLED` | No | `false` | Enable API key auth (`X-API-Key` header) |
| `API_KEY` | No | — | API key value when auth is enabled |
| `CACHE_ENABLED` | No | `false` | Enable in-memory response cache |
| `SERVICENOW_MODE` | No | `simulation` | `simulation` or `live` |
| `SERVICENOW_INSTANCE` | No | — | ServiceNow instance (for live mode) |
| `SERVICENOW_USERNAME` | No | — | ServiceNow username |
| `SERVICENOW_PASSWORD` | No | — | ServiceNow password |

*At least one LLM provider (Azure OpenAI **or** OpenAI) is required.

---

## Project Structure

```
deployment/
├── app/
│   ├── server.py               # FastAPI app, all routes, startup lifecycle
│   ├── agents/                 # IT Support & Enterprise agents
│   │   ├── it_helpdesk.py
│   │   ├── servicenow_agent.py
│   │   ├── enterprise_agents.py
│   │   └── conversation_manager.py
│   ├── deepagents/             # Deep Agents framework
│   │   ├── core/               # Middleware, types, base classes
│   │   ├── config/             # Agent configurations
│   │   ├── it_operations_agent.py
│   │   ├── sales_intelligence_agent.py
│   │   ├── recruitment_agent.py
│   │   └── software_dev/       # Software Dev Deep Agent
│   │       └── tools/          # 54+ SDLC tools (bash, Azure, git...)
│   ├── chains/                 # LangChain chains (chat, rag, agent)
│   ├── auth/                   # API key middleware
│   ├── cache/                  # Response cache
│   ├── governance/             # Cost estimator, policy
│   ├── integrations/           # Teams, Slack, Copilot Studio webhooks
│   ├── memory/                 # MemorySaver session management
│   ├── monitoring/             # Prometheus instrumentation
│   └── static/                 # Web UI (HTML/JS/CSS)
├── tests/                      # Full test suite (mirrors app/ structure)
├── docs/                       # Architecture, deployment, API docs
├── infrastructure/             # Azure Bicep IaC + Docker Compose prod
├── data/                       # Deep agent context files
├── .env.example                # Environment template
├── pyproject.toml              # Dependencies (uv)
├── Dockerfile                  # Multi-stage production build
├── docker-compose.yml          # Local dev compose
├── langgraph.json              # LangGraph Studio config
├── KNOWLEDGE.md                # AI agent knowledge base (authoritative)
└── README.md                   # This file
```

---

## Docker Deployment

```bash
cp .env.example .env  # add API keys

# Local dev
docker-compose up -d

# Production (with Prometheus + Grafana)
docker-compose -f infrastructure/docker-compose.prod.yml up -d
```

## Kubernetes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 10
```

---

## Development

```bash
make help          # All available commands
make run-reload    # Dev server with auto-reload
make test          # Run test suite
make lint          # ruff check
make format        # ruff format
```

### Adding a new agent

1. Create agent file in `app/agents/` or `app/deepagents/`
2. Register in `conversation_manager.py` (for session-based) or add routes in `server.py`
3. Export in `app/agents/__init__.py`
4. Add tests in `tests/`
5. Update `KNOWLEDGE.md`

---

## Documentation

| File | Purpose |
|------|---------|
| [KNOWLEDGE.md](KNOWLEDGE.md) | Authoritative AI-agent knowledge base — read before making changes |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and patterns |
| [docs/SETUP.md](docs/SETUP.md) | Detailed setup guide |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Production deployment guide |
| [docs/SECURITY.md](docs/SECURITY.md) | Security model and hardening |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | Operations runbook |
| [LANGGRAPH_SETUP.md](LANGGRAPH_SETUP.md) | LangGraph Studio visual development |

---

## Technology Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| Web Framework | FastAPI ≥ 0.115 | |
| LLM Framework | LangChain ≥ 0.3 | |
| Agent Framework | LangGraph ≥ 0.2 | `create_react_agent` pattern |
| API Serving | LangServe ≥ 0.3 | |
| Tracing | LangSmith | Verified on startup |
| Primary LLM | Azure OpenAI `o4-mini` | Reasoning model |
| Embedding | Azure OpenAI `text-embedding-3-small` | |
| Fallback LLM | OpenAI GPT-4o-mini / Anthropic Claude | |
| Python | 3.10–3.12 | 3.12 recommended |

## License

MIT


## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Application                     │
│  ┌─────────────────────────────────────────────────────┐│
│  │              LangServe Routes                        ││
│  │  /chat  │  /rag  │  /agent  │  /langgraph           ││
│  └─────────────────────────────────────────────────────┘│
│                          │                               │
│  ┌───────────────────────┴───────────────────────────┐  │
│  │                Chain Layer                         │  │
│  │  chat_chain │ rag_chain │ agent │ langgraph_agent │  │
│  └───────────────────────────────────────────────────┘  │
│                          │                               │
│  ┌───────────────────────┴───────────────────────────┐  │
│  │              LangSmith Tracing                     │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (required) or Anthropic API key
- Docker (optional, for containerized deployment)

### Local Development

1. **Navigate to deployment directory:**
   ```bash
   cd deployment
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .
   ```

4. **Run the server:**
   ```bash
   python -m uvicorn app.server:app --reload
   ```

5. **Access the API:**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Docker Deployment

```bash
cp .env.example .env
# Edit .env with your API keys

docker-compose up -d
```

## API Endpoints

### Status Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Redirect to API docs |
| `/docs` | GET | Interactive API documentation |
| `/health` | GET | Health check with component status |
| `/ready` | GET | Kubernetes readiness probe |

### LangChain Endpoints (via LangServe)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat/invoke` | POST | Simple chat completion |
| `/chat/stream` | POST | Streaming chat |
| `/rag/invoke` | POST | RAG query |
| `/rag/stream` | POST | Streaming RAG |
| `/agent/invoke` | POST | LangGraph React agent |
| `/agent/stream` | POST | Streaming agent |

### LangGraph Endpoint

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/langgraph/invoke` | POST | LangGraph agent with tools |

## Example Requests

### Chat
```bash
curl -X POST "http://localhost:8000/chat/invoke" \
  -H "Content-Type: application/json" \
  -d '{"input": {"input": "What is LangChain?"}}'
```

### RAG
```bash
curl -X POST "http://localhost:8000/rag/invoke" \
  -H "Content-Type: application/json" \
  -d '{"input": "What is RAG?"}'
```

### LangGraph Agent
```bash
curl -X POST "http://localhost:8000/langgraph/invoke" \
  -H "Content-Type: application/json" \
  -d '{"input": "What is 25 times 4?"}'
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes* | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | No | - | Anthropic API key |
| `LANGCHAIN_TRACING_V2` | No | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | - | LangSmith API key |
| `LANGCHAIN_PROJECT` | No | `langchain-platform` | LangSmith project |
| `TAVILY_API_KEY` | No | - | Tavily search API key |
| `SERVICENOW_MODE` | No | `simulation` | ServiceNow mode: `simulation` or `live` |
| `SERVICENOW_INSTANCE` | No | - | ServiceNow instance name |
| `SERVICENOW_USERNAME` | No | - | ServiceNow API username |
| `SERVICENOW_PASSWORD` | No | - | ServiceNow API password |
| `PORT` | No | `8000` | Server port |

*At least one LLM provider API key is required

### LangSmith Tracing

To enable tracing:

1. Get an API key from https://smith.langchain.com
2. Set in `.env`:
   ```
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_key_here
   LANGCHAIN_PROJECT=your-project-name
   ```

## Project Structure

```
deployment/
├── app/
│   ├── __init__.py
│   ├── server.py              # FastAPI application
│   ├── agents/                # IT Support agents
│   │   ├── it_helpdesk.py     # IT Helpdesk Agent
│   │   ├── servicenow_agent.py # ServiceNow ITSM Agent (10 tools)
│   │   └── conversation_manager.py
│   └── chains/
│       ├── __init__.py
│       ├── chat.py            # Simple chat chain
│       ├── rag.py             # RAG chain with vector store
│       ├── agent.py           # LangGraph React agent
│       └── langgraph_agent.py # LangGraph agent with tools
├── tests/
│   ├── __init__.py
│   ├── test_server.py         # Server endpoint tests
│   └── test_servicenow_agent_tools.py # ServiceNow tools tests
├── .env.example               # Environment template
├── .gitignore
├── Dockerfile                 # Production Docker image
├── docker-compose.yml         # Docker Compose config
├── Makefile                   # Development commands
├── pyproject.toml             # Python dependencies
├── KNOWLEDGE.md               # Knowledge base for AI agents
└── README.md                  # This file
```

## ServiceNow Integration

The ServiceNow Agent provides full ITSM operations with 10 tools:

| Tool | Description |
|------|-------------|
| `search_incidents` | Search incidents by query, state, priority |
| `get_incident_details` | Get detailed incident info |
| `create_incident` | Create new incident |
| `update_incident` | Update incidents with work notes |
| `get_change_requests` | List upcoming changes |
| `get_change_request_details` | Get detailed CHG ticket info |
| `search_cmdb` | Query CMDB configuration items |
| `get_my_tickets` | Get user's tickets |
| `get_service_request_details` | Get REQ/RITM details |
| `search_service_requests` | Search service requests |

**Modes**: `simulation` (default, uses mock data) or `live` (connects to real ServiceNow)

## Development

### Make Commands

```bash
make help          # Show all commands
make install       # Install dependencies
make dev           # Install dev dependencies
make run           # Run server
make run-reload    # Run with auto-reload
make docker-build  # Build Docker image
make docker-run    # Run with Docker Compose
make test          # Run tests
make lint          # Run linter
make format        # Format code
```

### Adding New Chains

1. Create a new file in `app/chains/`
2. Define your chain using LangChain/LangGraph
3. Export in `app/chains/__init__.py`
4. Add routes in `app/server.py`:
   ```python
   from app.chains.your_chain import your_chain
   add_routes(app, your_chain, path="/your-chain")
   ```
5. Update `KNOWLEDGE.md` with the new component

### Adding New Tools

Add to `app/chains/agent.py` or `app/chains/langgraph_agent.py`:

```python
@tool
def your_tool(param: str) -> str:
    """Tool description for LLM.

    Args:
        param: Parameter description.

    Returns:
        Result description.
    """
    return result

tools = [..., your_tool]
```

## Production Deployment

### Kubernetes

Use health endpoints for probes:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
```

### Health Check Response

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "chains_loaded": true,
  "langgraph_loaded": true,
  "tracing_enabled": true,
  "langsmith_project": "langchain-platform"
}
```

### Scaling

- Application is stateless - horizontally scalable
- Consider Redis for caching (uncomment in docker-compose.yml)
- Use managed vector database for production RAG

## Technology Stack

| Component | Technology |
|-----------|------------|
| Web Framework | FastAPI |
| LLM Framework | LangChain |
| Agent Framework | LangGraph |
| API Serving | LangServe |
| Tracing | LangSmith |
| Primary LLM | OpenAI GPT-4o-mini |
| Alternative LLM | Anthropic Claude |

## Documentation

- **KNOWLEDGE.md** - Detailed knowledge base for AI agents and contributors
- **API Docs** - Interactive docs at `/docs` endpoint

## License

MIT
