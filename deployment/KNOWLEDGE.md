# LangChain Platform - Knowledge Base

> **Purpose**: This document serves as the authoritative knowledge source for AI agents working on this repository. It contains architectural decisions, implementation patterns, and guidelines that must be followed when making changes or enhancements.

**Last Updated**: 2025-12-15

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Directory Structure](#directory-structure)
4. [Key Components](#key-components)
5. [Configuration](#configuration)
6. [API Endpoints](#api-endpoints)
7. [Dependencies](#dependencies)
8. [Development Patterns](#development-patterns)
9. [Testing Strategy](#testing-strategy)
10. [Deployment](#deployment)
11. [Common Tasks](#common-tasks)
12. [Troubleshooting](#troubleshooting)
13. [Change Log](#change-log)

---

## Project Overview

### What is this project?

A **production-ready deployment platform** that serves LangChain chains and LangGraph agents as REST APIs. It provides:

- FastAPI server with LangServe integration
- Multiple AI endpoints (chat, RAG, agents)
- LangSmith tracing for observability
- Docker containerization for deployment
- Kubernetes-ready health checks

### Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Web Framework | FastAPI | >=0.115.0 |
| LLM Framework | LangChain | >=0.3.0 |
| Agent Framework | LangGraph | >=0.2.0 |
| API Serving | LangServe | >=0.3.0 |
| Tracing | LangSmith | >=0.1.0 |
| Primary LLM | OpenAI GPT-4o-mini | - |
| Alternative LLM | Anthropic Claude | - |
| Python | Python | >=3.10 |

### Key Design Decisions

1. **LangGraph over legacy agents**: Uses `langgraph.prebuilt.create_react_agent` instead of deprecated `langchain.agents.create_tool_calling_agent`
2. **Lazy loading**: Chains load only when API keys are available
3. **Provider agnostic**: Supports both OpenAI and Anthropic
4. **Tracing first**: LangSmith tracing enabled by default when configured

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    LangServe Routes                      ││
│  │  /chat  │  /rag  │  /agent  │  /langgraph               ││
│  └─────────────────────────────────────────────────────────┘│
│                            │                                 │
│  ┌─────────────────────────┴─────────────────────────────┐  │
│  │                    Chain Layer                         │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐ │  │
│  │  │chat_chain│  │rag_chain │  │   agent_executor     │ │  │
│  │  │(OpenAI)  │  │(OpenAI)  │  │   (LangGraph)        │ │  │
│  │  └──────────┘  └──────────┘  └──────────────────────┘ │  │
│  │                              ┌──────────────────────┐ │  │
│  │                              │  langgraph_agent     │ │  │
│  │                              │  (OpenAI/Anthropic)  │ │  │
│  │                              └──────────────────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
│                            │                                 │
│  ┌─────────────────────────┴─────────────────────────────┐  │
│  │                  LangSmith Tracing                     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Request Flow

1. Request arrives at FastAPI endpoint
2. LangServe deserializes input
3. Chain/Agent processes request
4. LangSmith captures trace (if enabled)
5. Response returned to client

---

## Directory Structure

```
deployment/
├── app/                          # Application source code
│   ├── __init__.py              # Package marker
│   ├── server.py                # FastAPI application entry point
│   └── chains/                  # Chain and agent implementations
│       ├── __init__.py          # Exports all chains
│       ├── chat.py              # Simple chat chain (OpenAI)
│       ├── rag.py               # RAG chain with vector store
│       ├── agent.py             # LangGraph React agent
│       └── langgraph_agent.py   # LangGraph agent with custom tools
├── tests/                       # Test suite
│   ├── __init__.py
│   └── test_server.py           # Server endpoint tests
├── .env                         # Environment variables (NOT committed)
├── .env.example                 # Environment template (committed)
├── .gitignore                   # Git exclusions
├── Dockerfile                   # Production Docker image
├── docker-compose.yml           # Docker Compose configuration
├── Makefile                     # Development commands
├── pyproject.toml               # Python dependencies
├── README.md                    # User documentation
└── KNOWLEDGE.md                 # This file - AI agent knowledge base
```

---

## Key Components

### 1. Server (`app/server.py`)

**Purpose**: Main FastAPI application with LangServe routes

**Key Functions**:
- `setup_langsmith_tracing()`: Configures LangSmith if enabled
- `load_chains()`: Loads LangChain chains (requires OPENAI_API_KEY)
- `load_langgraph_agent()`: Loads LangGraph agent (OpenAI or Anthropic)
- `setup_langchain_routes()`: Registers LangServe endpoints

**Global State**:
```python
chains_loaded: bool        # True if LangChain chains loaded
langgraph_loaded: bool     # True if LangGraph agent loaded
tracing_enabled: bool      # True if LangSmith tracing active
```

### 2. Chat Chain (`app/chains/chat.py`)

**Purpose**: Simple conversational AI

**Implementation**:
```python
prompt | llm | StrOutputParser()
```

**Input Schema**: `{"input": "user message"}`

### 3. RAG Chain (`app/chains/rag.py`)

**Purpose**: Retrieval-Augmented Generation

**Components**:
- `InMemoryVectorStore` with OpenAI embeddings
- Pre-loaded sample documents about LangChain
- Retriever with k=3

**Input Schema**: `{"input": "question"}`

### 4. Agent (`app/chains/agent.py`)

**Purpose**: LangGraph React agent with tools

**Tools Available**:
- `get_current_time()`: Returns current datetime
- `calculate(expression)`: Evaluates math expressions
- `search_knowledge_base(query)`: Searches simulated knowledge base

**Implementation**: Uses `langgraph.prebuilt.create_react_agent`

### 5. LangGraph Agent (`app/chains/langgraph_agent.py`)

**Purpose**: Advanced LangGraph agent with StateGraph

**Features**:
- Supports OpenAI and Anthropic models
- Auto-selects available provider
- Custom tools: web_search, calculator, get_system_info
- Async support via `ainvoke`

**Model Selection Priority**:
1. Anthropic (if ANTHROPIC_API_KEY set)
2. OpenAI (if OPENAI_API_KEY set)

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes* | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | No | - | Anthropic API key |
| `LANGCHAIN_TRACING_V2` | No | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | - | LangSmith API key |
| `LANGCHAIN_PROJECT` | No | `langchain-platform` | LangSmith project name |
| `LANGCHAIN_ENDPOINT` | No | `https://api.smith.langchain.com` | LangSmith endpoint |
| `TAVILY_API_KEY` | No | - | Tavily search API key |
| `PORT` | No | `8000` | Server port |

*At least one LLM provider key required

### Loading Order

1. `.env` file loaded via `python-dotenv`
2. Environment variables override `.env`
3. Tracing configured before chains load
4. Chains load based on available API keys

---

## API Endpoints

### Health & Status

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Redirects to `/docs` |
| `/docs` | GET | Swagger UI documentation |
| `/health` | GET | Health check with component status |
| `/ready` | GET | Kubernetes readiness probe |

### LangChain Endpoints (via LangServe)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat/invoke` | POST | Chat completion |
| `/chat/stream` | POST | Streaming chat |
| `/chat/batch` | POST | Batch chat requests |
| `/rag/invoke` | POST | RAG query |
| `/rag/stream` | POST | Streaming RAG |
| `/agent/invoke` | POST | Agent execution |
| `/agent/stream` | POST | Streaming agent |

### LangGraph Endpoint (Custom)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/langgraph/invoke` | POST | LangGraph agent |

### Request/Response Formats

**LangServe Format**:
```json
// Request
{"input": {"input": "your message"}}

// Response
{"output": "response", "metadata": {"run_id": "..."}}
```

**LangGraph Format**:
```json
// Request
{"input": "your message"}

// Response
{"output": "response"}
```

---

## Dependencies

### Core Dependencies

```toml
langchain>=0.3.0
langchain-core>=0.3.0
langchain-openai>=0.2.0
langchain-anthropic>=0.3.0
langgraph>=0.2.0
langserve[all]>=0.3.0
langsmith>=0.1.0
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
python-dotenv>=1.0.0
pydantic>=2.0.0
langchain-community>=0.3.0
```

### Development Dependencies

```toml
pytest>=8.0.0
pytest-asyncio>=0.24.0
httpx>=0.27.0
ruff>=0.8.0
mypy>=1.13.0
```

---

## Development Patterns

### Adding a New Chain

1. Create file in `app/chains/`
2. Implement chain using LangChain patterns
3. Export in `app/chains/__init__.py`
4. Add route in `app/server.py`:

```python
from app.chains.your_chain import your_chain

add_routes(
    app,
    your_chain,
    path="/your-chain",
    enabled_endpoints=["invoke", "stream"],
)
```

### Adding a New Tool

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

# Add to tools list
tools = [..., your_tool]
```

### Error Handling Pattern

```python
try:
    result = risky_operation()
except Exception as e:
    logger.error("Operation failed", exc_info=True)
    return {"error": "Operation failed. Please try again."}
```

### Type Hints

All functions must include type hints:

```python
def process_input(data: str, count: int = 10) -> dict[str, Any]:
    """Process input data.

    Args:
        data: Input string to process.
        count: Number of items to return.

    Returns:
        Processed result dictionary.
    """
```

---

## Testing Strategy

### Test Categories

1. **Unit Tests** (`tests/unit/`): No network calls
2. **Integration Tests** (`tests/integration/`): With live APIs

### Running Tests

```bash
# All tests
make test

# Specific file
pytest tests/test_server.py -v

# With coverage
pytest --cov=app --cov-report=html
```

### Test Patterns

```python
def test_endpoint_name():
    """Test description."""
    # Arrange
    client = TestClient(app)

    # Act
    response = client.get("/endpoint")

    # Assert
    assert response.status_code == 200
```

---

## Deployment

### Local Development

```bash
cd deployment
cp .env.example .env
# Edit .env with your API keys
make run-reload
```

### Docker Deployment

```bash
cp .env.example .env
# Edit .env with your API keys
docker-compose up -d
```

### Kubernetes

Use health endpoints for probes:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 5
```

---

## Common Tasks

### Task: Update LLM Model

1. Edit the relevant chain file
2. Change the `model` parameter:
   ```python
   llm = ChatOpenAI(model="gpt-4o", temperature=0)
   ```
3. Restart server

### Task: Add New Endpoint

1. Create chain in `app/chains/`
2. Add to `app/chains/__init__.py`
3. Add route in `app/server.py`
4. Add tests in `tests/`
5. Update this KNOWLEDGE.md

### Task: Enable New LLM Provider

1. Add dependency to `pyproject.toml`
2. Add environment variable to `.env.example`
3. Update `load_chains()` or `load_langgraph_agent()`
4. Update documentation

### Task: Add Tracing Tags

```python
from langsmith import traceable

@traceable(name="custom_operation", tags=["production"])
def my_function():
    pass
```

---

## Troubleshooting

### Issue: Chains not loading

**Symptom**: `chains_loaded: false` in health check

**Solutions**:
1. Verify `OPENAI_API_KEY` is set in `.env`
2. Check `.env` file is in deployment directory
3. Restart server after changing `.env`

### Issue: Import errors with agents

**Symptom**: `cannot import name 'create_tool_calling_agent'`

**Solution**: Use LangGraph's `create_react_agent` instead:
```python
from langgraph.prebuilt import create_react_agent
```

### Issue: Tracing not working

**Symptom**: No traces in LangSmith

**Solutions**:
1. Set `LANGCHAIN_TRACING_V2=true`
2. Verify `LANGCHAIN_API_KEY` is valid
3. Check `LANGCHAIN_PROJECT` name

### Issue: Docker container fails

**Symptom**: Container exits immediately

**Solutions**:
1. Check logs: `docker-compose logs -f`
2. Verify `.env` file exists
3. Ensure port 8000 is not in use

---

## Change Log

### 2025-12-15 - Initial Release

**Added**:
- FastAPI server with LangServe integration
- Chat chain with OpenAI GPT-4o-mini
- RAG chain with in-memory vector store
- LangGraph React agent with tools
- LangGraph agent with Anthropic/OpenAI support
- LangSmith tracing configuration
- Docker and docker-compose setup
- Health and readiness endpoints
- Unit tests for server
- Comprehensive documentation

**Technical Decisions**:
- Chose `langgraph.prebuilt.create_react_agent` over deprecated `create_tool_calling_agent`
- Implemented lazy loading for chains based on API key availability
- Added provider auto-selection (Anthropic > OpenAI) for LangGraph agent

---

## Guidelines for AI Agents

### When Making Changes

1. **Read this file first** before making any changes
2. **Follow existing patterns** in the codebase
3. **Update KNOWLEDGE.md** when adding new features
4. **Run tests** before committing
5. **Use conventional commits** format

### Code Style

- Python 3.10+ features allowed
- Type hints required on all functions
- Google-style docstrings
- Ruff for linting and formatting
- No hardcoded secrets

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Security

- Never commit `.env` files
- Use environment variables for secrets
- Validate all user inputs
- Don't expose internal errors to users
