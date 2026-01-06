# LangChain Platform

A production-ready deployment platform serving LangChain chains and LangGraph agents as REST APIs with full LangSmith tracing support.

## Features

- **LangChain Integration** - Chat, RAG, and Agent chains via LangServe
- **LangGraph Agents** - Stateful agents with tool calling using LangGraph
- **Multi-Provider Support** - Works with OpenAI and Anthropic models
- **IT Support Agents** - IT Helpdesk and ServiceNow ITSM agents with conversation memory
- **ServiceNow Integration** - Full ITSM operations: incidents, changes, service requests, CMDB
- **LangSmith Tracing** - Full observability and debugging
- **Health Checks** - Kubernetes-ready health and readiness endpoints
- **Docker Support** - Multi-stage build for production deployment

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
