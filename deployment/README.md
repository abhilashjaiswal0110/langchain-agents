# LangChain Platform

A production-ready LangChain deployment platform with FastAPI and LangServe.

## Features

- **Chat Chain** - Simple conversational AI endpoint
- **RAG Chain** - Retrieval-Augmented Generation with vector search
- **Agent** - AI agent with tools (calculator, time, knowledge base)
- **Health Checks** - Kubernetes-ready health and readiness endpoints
- **Docker Support** - Multi-stage build for production deployment

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- OpenAI API key
- Docker (optional, for containerized deployment)

### Local Development

1. **Clone and navigate to deployment directory:**
   ```bash
   cd deployment
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

3. **Install dependencies:**
   ```bash
   uv pip install -e .
   # Or with pip:
   pip install -e .
   ```

4. **Run the server:**
   ```bash
   # With uv
   uv run uvicorn app.server:app --reload

   # Or directly
   python -m app.server
   ```

5. **Access the API:**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Docker Deployment

1. **Build and run with Docker Compose:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys

   docker-compose up -d
   ```

2. **View logs:**
   ```bash
   docker-compose logs -f
   ```

3. **Stop:**
   ```bash
   docker-compose down
   ```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Redirect to API docs |
| `/docs` | GET | Interactive API documentation |
| `/health` | GET | Health check |
| `/ready` | GET | Kubernetes readiness probe |
| `/chat/invoke` | POST | Chat completion |
| `/chat/stream` | POST | Streaming chat |
| `/rag/invoke` | POST | RAG query |
| `/agent/invoke` | POST | Agent execution |

### Example Requests

**Chat:**
```bash
curl -X POST "http://localhost:8000/chat/invoke" \
  -H "Content-Type: application/json" \
  -d '{"input": {"input": "What is LangChain?"}}'
```

**RAG:**
```bash
curl -X POST "http://localhost:8000/rag/invoke" \
  -H "Content-Type: application/json" \
  -d '{"input": "What is RAG?"}'
```

**Agent:**
```bash
curl -X POST "http://localhost:8000/agent/invoke" \
  -H "Content-Type: application/json" \
  -d '{"input": {"input": "What is 25 * 4?"}}'
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `PORT` | No | Server port (default: 8000) |
| `LANGCHAIN_TRACING_V2` | No | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | LangSmith API key |
| `LANGCHAIN_PROJECT` | No | LangSmith project name |

## Project Structure

```
deployment/
├── app/
│   ├── __init__.py
│   ├── server.py          # FastAPI application
│   └── chains/
│       ├── __init__.py
│       ├── chat.py        # Chat chain
│       ├── rag.py         # RAG chain
│       └── agent.py       # Agent with tools
├── Dockerfile             # Production Docker image
├── docker-compose.yml     # Docker Compose configuration
├── pyproject.toml         # Python dependencies
├── Makefile              # Development commands
├── .env.example          # Environment template
└── README.md             # This file
```

## Development

### Available Make Commands

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
2. Define your chain using LangChain
3. Export it in `app/chains/__init__.py`
4. Add routes in `app/server.py`:
   ```python
   from app.chains.your_chain import your_chain
   add_routes(app, your_chain, path="/your-chain")
   ```

## Production Deployment

### Kubernetes

Use the health endpoints for probes:

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

### Scaling

- The application is stateless and can be horizontally scaled
- Consider using Redis for caching (uncomment in docker-compose.yml)
- Use a managed vector database for production RAG workloads

## License

MIT
