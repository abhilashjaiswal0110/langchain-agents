# Deployment Guide

> **Last Updated**: 2025-12-19
> **Version**: 2.0.0

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Local Development](#local-development)
4. [Docker Deployment](#docker-deployment)
5. [External Exposure (ngrok)](#external-exposure-ngrok)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Configuration Management](#configuration-management)
8. [Rollback Procedures](#rollback-procedures)
9. [Health Checks & Monitoring](#health-checks--monitoring)

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10+ | 3.11+ |
| Memory | 512 MB | 2 GB |
| CPU | 1 core | 2 cores |
| Disk | 1 GB | 5 GB |
| Docker | 24.0+ | Latest |
| Docker Compose | 2.0+ | Latest |

### Required API Keys

| Service | Required | Purpose |
|---------|----------|---------|
| OpenAI API Key | Yes* | Primary LLM provider |
| Anthropic API Key | No | Alternative LLM provider |
| LangSmith API Key | No | Observability and tracing |
| Tavily API Key | No | Web search functionality |

*At least one LLM provider API key is required.

### Development Tools

```bash
# Check Python version
python --version  # Should be 3.10+

# Check Docker
docker --version
docker compose version

# Check uv (optional but recommended)
uv --version
```

---

## Environment Setup

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd langchain-agents/deployment
```

### Step 2: Create Environment File

```bash
# Copy example environment file
cp .env.example .env
```

### Step 3: Configure Environment Variables

Edit `.env` file with your credentials:

```ini
# =============================================================================
# LLM PROVIDERS (at least one required)
# =============================================================================
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# =============================================================================
# LANGSMITH TRACING (recommended for production)
# =============================================================================
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_your_langsmith_key
LANGCHAIN_PROJECT=langchain-enterprise-agents

# =============================================================================
# OPTIONAL SERVICES
# =============================================================================
TAVILY_API_KEY=tvly-your-tavily-key

# =============================================================================
# SERVER CONFIGURATION
# =============================================================================
PORT=8000

# =============================================================================
# SECURITY (required for external exposure)
# =============================================================================
API_KEY_ENABLED=false
API_KEY=your-secure-api-key-here
CORS_ORIGINS=*
```

---

## Local Development

### Option A: Using pip

```bash
cd deployment

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -e .

# Run server
python -m uvicorn app.server:app --reload --host 0.0.0.0 --port 8000
```

### Option B: Using uv (Recommended)

```bash
cd deployment

# Create virtual environment and install
uv venv
uv pip install -e .

# Run server
uv run uvicorn app.server:app --reload --host 0.0.0.0 --port 8000
```

### Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "version": "1.0.0",
#   "chains_loaded": true,
#   "enterprise_agents_loaded": true,
#   ...
# }
```

---

## Docker Deployment

### Build and Run

```bash
cd deployment

# Build image
docker compose build

# Start container
docker compose up -d

# View logs
docker compose logs -f
```

### Docker Compose Configuration

```yaml
# docker-compose.yml
services:
  langchain-platform:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: langchain-platform
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
      - LANGCHAIN_TRACING_V2=${LANGCHAIN_TRACING_V2:-false}
      - LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY:-}
      - LANGCHAIN_PROJECT=${LANGCHAIN_PROJECT:-langchain-platform}
      - API_KEY_ENABLED=${API_KEY_ENABLED:-false}
      - API_KEY=${API_KEY:-}
    env_file:
      - .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Multi-Stage Dockerfile

```dockerfile
# Build stage
FROM python:3.11-slim AS builder

RUN pip install uv
WORKDIR /build
COPY pyproject.toml README.md ./
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install --no-cache .

# Production stage
FROM python:3.11-slim AS production

RUN useradd --create-home --shell /bin/bash appuser
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY --chown=appuser:appuser ./app ./app
USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## External Exposure (ngrok)

### Step 1: Enable Security

```bash
# Edit .env
API_KEY_ENABLED=true
API_KEY=lcp-secure-key-$(openssl rand -hex 16)
```

### Step 2: Rebuild Container

```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

### Step 3: Start ngrok

```bash
# Start ngrok tunnel
ngrok http 8000

# Note the https URL (e.g., https://abc123.ngrok-free.dev)
```

### Step 4: Test External Access

```bash
# Test without API key (should return 401)
curl https://your-ngrok-url.ngrok-free.dev/api/enterprise/agents

# Test with API key (should return 200)
curl https://your-ngrok-url.ngrok-free.dev/api/enterprise/agents \
  -H "X-API-Key: your-api-key"
```

### ngrok Configuration (Optional)

Create `ngrok.yml` for persistent configuration:

```yaml
version: "2"
tunnels:
  langchain:
    proto: http
    addr: 8000
    inspect: true
```

---

## Kubernetes Deployment

### Kubernetes Manifests

#### Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langchain-platform
  labels:
    app: langchain-platform
spec:
  replicas: 2
  selector:
    matchLabels:
      app: langchain-platform
  template:
    metadata:
      labels:
        app: langchain-platform
    spec:
      containers:
      - name: langchain-platform
        image: langchain-platform:latest
        ports:
        - containerPort: 8000
        envFrom:
        - secretRef:
            name: langchain-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2"
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

#### Service

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: langchain-platform
spec:
  selector:
    app: langchain-platform
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

#### Secrets

```yaml
# k8s/secrets.yaml (DO NOT commit to git)
apiVersion: v1
kind: Secret
metadata:
  name: langchain-secrets
type: Opaque
stringData:
  OPENAI_API_KEY: "sk-your-key"
  LANGCHAIN_API_KEY: "lsv2-your-key"
  API_KEY: "your-secure-key"
```

### Apply Configuration

```bash
# Create namespace
kubectl create namespace langchain

# Apply secrets (from external secret manager in production)
kubectl apply -f k8s/secrets.yaml -n langchain

# Deploy application
kubectl apply -f k8s/deployment.yaml -n langchain
kubectl apply -f k8s/service.yaml -n langchain

# Verify deployment
kubectl get pods -n langchain
kubectl logs -f deployment/langchain-platform -n langchain
```

---

## Configuration Management

### Environment-Specific Configuration

```
config/
├── dev/
│   └── .env.dev
├── staging/
│   └── .env.staging
└── prod/
    └── .env.prod.template  # Never commit actual prod secrets
```

### Configuration Hierarchy

1. Environment variables (highest priority)
2. `.env` file
3. `.env.example` defaults

### Required vs Optional Variables

| Variable | Required | Default |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes* | - |
| `ANTHROPIC_API_KEY` | No | - |
| `LANGCHAIN_TRACING_V2` | No | `false` |
| `LANGCHAIN_API_KEY` | No | - |
| `PORT` | No | `8000` |
| `API_KEY_ENABLED` | No | `false` |
| `API_KEY` | Conditional** | - |
| `CORS_ORIGINS` | No | `*` |

*At least one LLM provider required
**Required if `API_KEY_ENABLED=true`

---

## Rollback Procedures

### Docker Rollback

```bash
# List available images
docker images langchain-platform

# Tag current as backup
docker tag langchain-platform:latest langchain-platform:backup-$(date +%Y%m%d)

# Rollback to previous version
docker compose down
docker tag langchain-platform:previous langchain-platform:latest
docker compose up -d
```

### Kubernetes Rollback

```bash
# View rollout history
kubectl rollout history deployment/langchain-platform -n langchain

# Rollback to previous revision
kubectl rollout undo deployment/langchain-platform -n langchain

# Rollback to specific revision
kubectl rollout undo deployment/langchain-platform --to-revision=2 -n langchain

# Verify rollback
kubectl rollout status deployment/langchain-platform -n langchain
```

### Git-Based Rollback

```bash
# List recent tags
git tag -l "v*" --sort=-v:refname | head -5

# Checkout previous version
git checkout v1.0.0

# Rebuild and deploy
docker compose build
docker compose up -d
```

---

## Health Checks & Monitoring

### Health Endpoints

| Endpoint | Purpose | Response |
|----------|---------|----------|
| `GET /health` | Full health status | Components and tracing status |
| `GET /ready` | Kubernetes readiness | Simple ready check |

### Health Response Schema

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "chains_loaded": true,
  "langgraph_loaded": true,
  "doc_rag_loaded": true,
  "it_support_loaded": true,
  "enterprise_agents_loaded": true,
  "tracing_enabled": true,
  "langsmith_project": "langchain-platform"
}
```

### Monitoring Checklist

| Check | Command | Expected |
|-------|---------|----------|
| Container status | `docker ps` | Running, healthy |
| Health endpoint | `curl /health` | status: healthy |
| Logs | `docker logs langchain-platform` | No errors |
| API response | `curl /api/enterprise/agents` | 200 OK |
| Memory usage | `docker stats` | < 2GB |

### LangSmith Monitoring

1. Navigate to https://smith.langchain.com
2. Select project from `LANGCHAIN_PROJECT`
3. View traces, latency, and error rates

### Alerting Recommendations

| Metric | Threshold | Action |
|--------|-----------|--------|
| Health check failure | 3 consecutive | Restart container |
| Response time | > 30s | Investigate LLM latency |
| Error rate | > 5% | Check logs |
| Memory usage | > 80% | Scale or optimize |

---

## Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs
docker compose logs langchain-platform

# Common causes:
# - Missing API keys
# - Port 8000 already in use
# - Insufficient memory
```

#### Agents Not Loading

```bash
# Verify API keys
docker exec langchain-platform env | grep -E "OPENAI|ANTHROPIC"

# Check health endpoint
curl localhost:8000/health | jq '.enterprise_agents_loaded'
```

#### Slow Responses

```bash
# Check LangSmith traces
# - High latency in LLM calls
# - Tool execution taking too long
# - Network issues
```

#### Memory Issues

```bash
# Monitor memory
docker stats langchain-platform

# Increase limits in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G
```

---

## Related Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md) - System architecture
- [SECURITY.md](./SECURITY.md) - Security guidelines
- [OPERATIONS.md](./OPERATIONS.md) - Operations runbook
