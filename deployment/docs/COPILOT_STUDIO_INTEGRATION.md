# Microsoft Copilot Studio Integration Guide

**Document Version:** 1.0
**Last Updated:** January 12, 2026
**Status:** Production Ready

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Available Agent Endpoints](#available-agent-endpoints)
- [Prerequisites](#prerequisites)
- [Part 1: Platform Configuration](#part-1-platform-configuration)
- [Part 2: Copilot Studio Setup](#part-2-copilot-studio-setup)
- [Part 3: Testing & Validation](#part-3-testing--validation)
- [Part 4: Production Deployment](#part-4-production-deployment)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

---

## Overview

This guide provides step-by-step instructions for integrating the LangChain Platform agents with Microsoft Copilot Studio. After setup, your Copilot Studio agents can invoke:

- **Enterprise Agents** (8 specialized agents)
- **IT Support Agents** (conversational helpdesk)
- **Deep Agents** (IT Operations with 6 subagents)

All agents are accessible via public ngrok URL or your production domain.

### Integration Benefits

✅ **No Code Required** - Use Copilot Studio's HTTP action connector
✅ **Streaming Support** - Real-time responses with SSE
✅ **Multi-Agent** - Access all 15+ agents from one platform
✅ **Secure** - API key authentication, CORS protection
✅ **Scalable** - Docker-based deployment with health checks

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Microsoft Copilot Studio                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Copilot Agent (Dialog Flow)                               │ │
│  │  ├── User Input Trigger                                    │ │
│  │  ├── HTTP Action: Call LangChain API                       │ │
│  │  └── Response Handling                                     │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTPS (API Key)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Ngrok Tunnel                             │
│   https://your-subdomain.ngrok-free.dev                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               LangChain Platform (Docker Container)              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  FastAPI Server (Port 8000)                              │  │
│  │  ├── CORS Middleware (Origin Validation)                 │  │
│  │  ├── API Key Middleware (Authentication)                 │  │
│  │  └── Agent Router                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Agent Layer                                             │  │
│  │  ├── Enterprise Agents (8 agents)                        │  │
│  │  ├── IT Support Agents (conversational)                  │  │
│  │  └── Deep Agents (IT Operations)                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Available Agent Endpoints

### 1. Enterprise Agents

| Agent | Endpoint | Description |
|-------|----------|-------------|
| **Research** | `/api/enterprise/research/invoke` | Web search, information synthesis, competitive analysis |
| **Content** | `/api/enterprise/content/invoke` | LinkedIn, X (Twitter), blog post generation |
| **Data Analyst** | `/api/enterprise/data-analyst/invoke` | Excel/CSV analysis, data visualization |
| **Document** | `/api/enterprise/documents/invoke` | SOP/WLI/Policy document generation |
| **Multilingual RAG** | `/api/enterprise/rag/invoke` | Document Q&A in 50+ languages |
| **HITL Support** | `/api/enterprise/support/invoke` | Human-in-the-loop IT support with approval workflow |
| **Code Assistant** | `/api/enterprise/code/invoke` | Application modernization, code generation |
| **Document Intelligence** | `/api/enterprise/document-intelligence/invoke` | Multi-format document analysis, OCR, translation |

### 2. IT Support Agents (Conversational)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/conversation/start` | POST | Start conversation session |
| `/api/conversation/chat` | POST | Send message to agent |
| `/api/conversation/{session_id}` | GET | Get conversation history |
| `/api/agents` | GET | List available IT agents |

### 3. Deep Agents (IT Operations)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/deepagent/start` | POST | Start Deep Agent session |
| `/api/deepagent/chat` | POST | Send message (returns full response) |
| `/api/deepagent/chat/stream` | POST | Stream responses with SSE |
| `/api/deepagent/context/{session_id}` | GET | Get session context (todos, files) |

**Deep Agent Subagents:**
- Incident Management
- Change Management
- Problem Management
- Asset/CMDB Management
- SLA Management
- Knowledge Management

### 4. Dedicated Webhook (Recommended)

**Endpoint:** `/api/webhooks/copilot-studio`
**Method:** POST
**Purpose:** Optimized for Copilot Studio with simplified request/response

---

## Prerequisites

### Platform Requirements

- [x] Docker installed and running
- [x] Ngrok installed (or production domain)
- [x] LangChain Platform deployed (see [NGROK_SETUP.md](NGROK_SETUP.md))
- [x] Active ngrok tunnel to localhost:8000
- [x] API key generated and configured

### Copilot Studio Requirements

- Microsoft 365 subscription with Copilot Studio license
- Admin access to Copilot Studio portal
- Power Platform environment

### API Keys Required

```bash
# Required in .env file
OPENAI_API_KEY=sk-...           # For LLM operations
ANTHROPIC_API_KEY=sk-ant-...    # Alternative LLM provider
LANGCHAIN_API_KEY=lsv2_...      # For tracing
API_KEY=your-secure-key         # For webhook authentication
```

---

## Part 1: Platform Configuration

### Step 1.1: Update CORS Settings

Add your ngrok URL to allowed origins:

```bash
# In deployment/.env
CORS_ORIGINS=http://localhost:8000,http://localhost:3000,https://your-subdomain.ngrok-free.dev
```

**Important:** Replace `your-subdomain.ngrok-free.dev` with your actual ngrok URL.

### Step 1.2: Verify API Key

```bash
# In deployment/.env
API_KEY_ENABLED=true
API_KEY=your-secure-api-key-here

# Generate secure key (run in terminal):
openssl rand -hex 32
```

### Step 1.3: Restart Docker Container

```bash
cd deployment
docker compose down
docker compose up -d

# Verify health
curl http://localhost:8000/health
```

### Step 1.4: Verify Ngrok Tunnel

```bash
# Get active tunnel info
curl http://localhost:4040/api/tunnels | ConvertFrom-Json | Select-Object -ExpandProperty tunnels | Select-Object public_url

# Test health endpoint via ngrok
curl -H "ngrok-skip-browser-warning: true" https://your-subdomain.ngrok-free.dev/health
```

---

## Part 2: Copilot Studio Setup

### Step 2.1: Create New Topic (Recommended Approach)

1. **Open Copilot Studio**
   - Go to https://copilotstudio.microsoft.com
   - Select your environment
   - Create or select existing Copilot

2. **Create New Topic**
   - Click **Topics** → **+ New topic**
   - Name: "LangChain Agent Integration"
   - Add trigger phrases:
     - "research information"
     - "analyze data"
     - "generate document"
     - "help with code"

3. **Add Question Node**
   - Add node → **Ask a question**
   - Question text: "What would you like me to help with?"
   - Save response to variable: `UserQuery`

4. **Add Action - HTTP Request**
   - Add node → **Call an action** → **Create a flow**
   - Or use existing **HTTP** action

### Step 2.2: Configure HTTP Action

**Method:** POST
**URL:** `https://your-subdomain.ngrok-free.dev/api/webhooks/copilot-studio`

**Headers:**
```json
{
  "Content-Type": "application/json",
  "X-API-Key": "your-secure-api-key-here",
  "ngrok-skip-browser-warning": "true"
}
```

**Request Body:**
```json
{
  "query": "{UserQuery}",
  "agent_type": "research",
  "user_id": "{System.User.Id}",
  "conversation_id": "{System.Conversation.Id}",
  "session_id": null,
  "channel": "copilot-studio"
}
```

**Response Schema:**
```json
{
  "success": "boolean",
  "response": "string",
  "error": "string",
  "session_id": "string",
  "agent_type": "string",
  "source": "string",
  "metadata": "object"
}
```

### Step 2.3: Handle Response

Add **Condition** node:
- **If** `response.success` equals `true`:
  - **Show message:** `{response.response}`
- **Else:**
  - **Show message:** "I encountered an error: {response.error}"

### Step 2.4: Save and Test

1. Click **Save**
2. Click **Test your copilot**
3. Try trigger phrases:
   - "research information about quantum computing"
   - "analyze data trends"

---

## Part 3: Testing & Validation

### Test 1: Health Check

```bash
curl -H "ngrok-skip-browser-warning: true" \
     https://your-subdomain.ngrok-free.dev/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "it_support_loaded": true,
  "enterprise_agents_loaded": true,
  "deep_agent_loaded": true
}
```

### Test 2: List Available Agents

```bash
curl -H "ngrok-skip-browser-warning: true" \
     https://your-subdomain.ngrok-free.dev/api/enterprise/agents
```

### Test 3: Copilot Studio Webhook (Direct API Test)

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secure-api-key-here" \
  -H "ngrok-skip-browser-warning: true" \
  -d '{
    "query": "What are the latest AI trends?",
    "agent_type": "research",
    "user_id": "test_user",
    "conversation_id": "test_conv_001",
    "channel": "copilot-studio"
  }' \
  https://your-subdomain.ngrok-free.dev/api/webhooks/copilot-studio
```

**Expected Response:**
```json
{
  "success": true,
  "response": "### Executive Summary\n\nThe latest AI trends include...",
  "error": null,
  "session_id": "copilot-test_conv_001",
  "agent_type": "research",
  "source": "copilot-studio",
  "metadata": {
    "channel": "copilot-studio",
    "user_id": "test_user",
    "conversation_id": "test_conv_001"
  }
}
```

### Test 4: Enterprise Agent Direct Invocation

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secure-api-key-here" \
  -H "ngrok-skip-browser-warning: true" \
  -d '{
    "query": "Analyze the top 5 AI companies in 2026",
    "session_id": "test_session_001"
  }' \
  https://your-subdomain.ngrok-free.dev/api/enterprise/research/invoke
```

### Test 5: Deep Agent IT Operations

**Start Session:**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secure-api-key-here" \
  -H "ngrok-skip-browser-warning: true" \
  -d '{"user_id": "copilot_user"}' \
  https://your-subdomain.ngrok-free.dev/api/deepagent/start
```

**Send Chat:**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secure-api-key-here" \
  -H "ngrok-skip-browser-warning: true" \
  -d '{
    "session_id": "SESSION_ID_FROM_START",
    "message": "Show me all P1 incidents from last 7 days"
  }' \
  https://your-subdomain.ngrok-free.dev/api/deepagent/chat
```

---

## Part 4: Production Deployment

### Option 1: Persistent Ngrok (Development/Testing)

**Pros:** Quick setup, no infrastructure needed
**Cons:** URL changes if ngrok restarts (use paid plan for static domain)

**Setup:**
1. Get ngrok paid plan for reserved domain
2. Configure reserved domain in ngrok.yml
3. Run: `ngrok http 8000 --domain=your-reserved-domain.ngrok.app`
4. Update Copilot Studio with permanent URL

### Option 2: Azure App Service (Recommended Production)

**Pros:** Managed, scalable, permanent URL, SSL included
**Cons:** Azure costs

**Setup:**
1. Deploy Docker image to Azure Container Registry
2. Create Azure App Service with container
3. Configure environment variables
4. Use App Service URL in Copilot Studio

**See:** [DEPLOYMENT.md](DEPLOYMENT.md) for detailed Azure deployment

### Option 3: Azure Container Apps (Serverless)

**Pros:** Auto-scaling, cost-effective for variable load
**Cons:** Cold start latency

**Setup:**
1. Push Docker image to ACR
2. Create Container App
3. Enable HTTPS ingress
4. Configure in Copilot Studio

### Option 4: On-Premises with Public IP

**Pros:** Full control, no cloud costs
**Cons:** Network configuration, SSL certificates

**Setup:**
1. Obtain static public IP
2. Configure firewall/NAT
3. Install SSL certificate
4. Update DNS records

---

## Security Considerations

### 🔐 Authentication

**API Key Protection:**
```bash
# Generate strong key (32+ chars)
openssl rand -hex 32

# Store in .env (NEVER commit to git)
API_KEY=your-generated-key-here
```

**Header Required:**
```
X-API-Key: your-generated-key-here
```

### 🛡️ CORS Configuration

```bash
# In .env - NEVER use "*" in production
CORS_ORIGINS=https://your-subdomain.ngrok-free.dev,https://copilot.microsoft.com

# Restart after changes
docker compose restart
```

### 🔒 HTTPS Only

- ✅ Always use HTTPS (ngrok provides automatically)
- ❌ Never use HTTP in production
- ✅ Validate SSL certificates

### 🚫 Rate Limiting

Add rate limiting for production:

```python
# In server.py (future enhancement)
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/webhooks/copilot-studio")
@limiter.limit("100/minute")
async def copilot_studio_webhook(...):
    ...
```

### 📊 Monitoring

**Enable LangSmith Tracing:**
```bash
# In .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=copilot-studio-integration
```

**Check Logs:**
```bash
# Docker logs
docker logs langchain-platform --tail=100 -f

# Specific to Copilot Studio requests
docker logs langchain-platform 2>&1 | grep "copilot-studio"
```

---

## Troubleshooting

### Issue 1: 401 Unauthorized

**Symptoms:** Copilot Studio receives 401 error

**Solutions:**
1. Verify API key in Copilot Studio HTTP headers matches `.env` file
2. Check `API_KEY_ENABLED=true` in `.env`
3. Ensure header name is `X-API-Key` (case-sensitive)

**Test:**
```bash
curl -H "X-API-Key: wrong-key" https://your-subdomain.ngrok-free.dev/health
# Should return 401
```

### Issue 2: CORS Error

**Symptoms:** Browser console shows CORS error

**Solutions:**
1. Add ngrok URL to `CORS_ORIGINS` in `.env`
2. Restart Docker container: `docker compose restart`
3. Verify with: `docker exec langchain-platform env | grep CORS`

**Correct Format:**
```bash
CORS_ORIGINS=https://your-subdomain.ngrok-free.dev,https://copilot.microsoft.com
```

### Issue 3: Ngrok Tunnel Not Found

**Symptoms:** ERR_NGROK_3200 or "Tunnel not found"

**Solutions:**
1. Check ngrok process: `Get-Process ngrok`
2. Verify URL: `curl http://localhost:4040/api/tunnels`
3. Restart ngrok: `ngrok http 8000`

### Issue 4: Agent Returns Error

**Symptoms:** `success: false` in response

**Solutions:**
1. Check agent is loaded: `curl https://your-url/api/enterprise/agents`
2. Verify API keys (OPENAI_API_KEY or ANTHROPIC_API_KEY)
3. Check Docker logs: `docker logs langchain-platform --tail=50`

**Common Errors:**
- "Agent 'research' not available" → Agent not loaded (missing API keys)
- "No API keys found" → Check .env file location

### Issue 5: Slow Responses

**Symptoms:** Copilot Studio timeout or slow responses

**Solutions:**
1. Use streaming endpoint for real-time feedback
2. Check Docker resource limits: `docker stats langchain-platform`
3. Increase timeout in Copilot Studio HTTP action (default: 30s)
4. Monitor LangSmith for bottlenecks

**Resource Check:**
```bash
docker stats langchain-platform --no-stream
```

### Issue 6: Session Not Persisting

**Symptoms:** Deep Agent loses context between messages

**Solutions:**
1. Verify `deepagent_data` volume exists: `docker volume ls`
2. Check session_id is being passed in subsequent requests
3. Inspect storage: `docker exec langchain-platform ls -la /app/data/deepagent_context`

---

## API Reference

### Copilot Studio Webhook

**Endpoint:** `POST /api/webhooks/copilot-studio`

**Request:**
```json
{
  "query": "string (required)",
  "agent_type": "string (required)",
  "user_id": "string (optional)",
  "conversation_id": "string (optional)",
  "session_id": "string (optional)",
  "channel": "string (optional)"
}
```

**Agent Types:**
- `research` - Web search and synthesis
- `content` - Social media content generation
- `data-analyst` - Data analysis
- `document` - Document generation
- `multilingual-rag` - Document Q&A
- `hitl-support` - IT support with human approval
- `code-assistant` - Code generation and modernization

**Response:**
```json
{
  "success": true,
  "response": "Agent response text",
  "error": null,
  "session_id": "copilot-conversation_id",
  "agent_type": "research",
  "source": "copilot-studio",
  "metadata": {
    "channel": "copilot-studio",
    "user_id": "user123",
    "conversation_id": "conv456"
  }
}
```

### Enterprise Agent Direct Invocation

**Endpoints:**
- `POST /api/enterprise/research/invoke`
- `POST /api/enterprise/content/invoke`
- `POST /api/enterprise/data-analyst/invoke`
- `POST /api/enterprise/documents/invoke`
- `POST /api/enterprise/rag/invoke`
- `POST /api/enterprise/support/invoke`
- `POST /api/enterprise/code/invoke`
- `POST /api/enterprise/document-intelligence/invoke`

**Request:**
```json
{
  "query": "string (required)",
  "session_id": "string (optional)"
}
```

**Response:**
```json
{
  "success": true,
  "response": "Agent response in markdown format",
  "session_id": "generated-or-provided-session-id",
  "agent_type": "research",
  "tool_calls": null,
  "error": null
}
```

### Deep Agent Endpoints

#### Start Session
**Endpoint:** `POST /api/deepagent/start`

**Request:**
```json
{
  "user_id": "string (required)"
}
```

**Response:**
```json
{
  "session_id": "uuid",
  "message": "Deep Agent session started",
  "agent_type": "it_operations_deep",
  "available_tools": ["search_incidents", "create_incident", ...]
}
```

#### Chat (Non-Streaming)
**Endpoint:** `POST /api/deepagent/chat`

**Request:**
```json
{
  "session_id": "uuid (required)",
  "message": "string (required)"
}
```

**Response:**
```json
{
  "response": "Agent response",
  "session_id": "uuid",
  "tool_calls": [],
  "todos_updated": false,
  "files_updated": false
}
```

#### Chat (Streaming)
**Endpoint:** `POST /api/deepagent/chat/stream`

**Request:** Same as non-streaming

**Response:** Server-Sent Events (SSE)
```
event: thinking
data: {"content": "Analyzing incidents..."}

event: tool_call
data: {"tool": "search_incidents", "args": {...}}

event: tool_result
data: {"result": "Found 3 incidents"}

event: content
data: {"content": "Here are the results..."}

event: done
data: {"session_id": "uuid"}
```

---

## Quick Reference

### Essential Commands

```bash
# Start platform
cd deployment && docker compose up -d

# Start ngrok
ngrok http 8000

# Get ngrok URL
curl http://localhost:4040/api/tunnels

# Test health
curl -H "ngrok-skip-browser-warning: true" https://your-url.ngrok-free.dev/health

# View logs
docker logs langchain-platform -f

# Restart after config changes
docker compose restart
```

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...
API_KEY=your-secure-key
CORS_ORIGINS=https://your-ngrok-url

# Optional
ANTHROPIC_API_KEY=sk-ant-...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
```

### Copilot Studio HTTP Action Template

```
Method: POST
URL: https://your-subdomain.ngrok-free.dev/api/webhooks/copilot-studio

Headers:
  Content-Type: application/json
  X-API-Key: your-secure-api-key-here
  ngrok-skip-browser-warning: true

Body:
{
  "query": "{UserQuery}",
  "agent_type": "research",
  "user_id": "{System.User.Id}",
  "conversation_id": "{System.Conversation.Id}"
}

Response Handling:
  if response.success == true:
    Show: response.response
  else:
    Show: response.error
```

---

## Next Steps

1. ✅ Complete platform configuration (Part 1)
2. ✅ Set up Copilot Studio integration (Part 2)
3. ✅ Test all endpoints (Part 3)
4. ⏭️ Deploy to production (Part 4)
5. 🔄 Monitor with LangSmith
6. 📊 Set up logging and analytics

## Support & Resources

- **Platform Documentation:** [KNOWLEDGE.md](../KNOWLEDGE.md)
- **Deployment Guide:** [DEPLOYMENT.md](DEPLOYMENT.md)
- **Ngrok Setup:** [NGROK_SETUP.md](NGROK_SETUP.md)
- **API Reference:** [docs/api/README.md](api/README.md)
- **LangSmith:** https://smith.langchain.com
- **Copilot Studio:** https://copilotstudio.microsoft.com

---

**Document Status:** Production Ready
**Tested With:** Copilot Studio (January 2026), Ngrok Free/Paid, Docker 24.0+
**Compatibility:** Microsoft 365, Power Platform, Azure
