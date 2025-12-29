# API Reference

> **Last Updated**: 2025-12-19
> **Version**: 2.0.0
> **Base URL**: `http://localhost:8000`

---

## Table of Contents

1. [Authentication](#authentication)
2. [Health Endpoints](#health-endpoints)
3. [Enterprise Agent Endpoints](#enterprise-agent-endpoints)
4. [Webhook Endpoints](#webhook-endpoints)
5. [Legacy Endpoints](#legacy-endpoints)
6. [Error Handling](#error-handling)

---

## Authentication

### API Key Authentication

When `API_KEY_ENABLED=true`, all protected endpoints require the `X-API-Key` header:

```bash
curl -X GET "http://localhost:8000/api/enterprise/agents" \
  -H "X-API-Key: your-api-key"
```

### Public Endpoints

The following endpoints do not require authentication:
- `GET /` - Redirect to docs
- `GET /docs` - OpenAPI documentation
- `GET /redoc` - ReDoc documentation
- `GET /health` - Health check
- `GET /ready` - Readiness check
- `GET /chat` - Web UI

---

## Health Endpoints

### GET /health

Full health status with component information.

**Response:**
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

### GET /ready

Kubernetes readiness probe.

**Response:**
```json
{
  "status": "ready"
}
```

---

## Enterprise Agent Endpoints

### GET /api/enterprise/agents

List all available enterprise agents.

**Response:**
```json
{
  "status": "available",
  "agents": {
    "research": {
      "loaded": true,
      "description": "AI Research Agent for web search and information synthesis",
      "endpoint": "/api/enterprise/research/invoke"
    },
    "content": { ... },
    "data_analyst": { ... },
    "document": { ... },
    "multilingual_rag": { ... },
    "hitl_support": { ... },
    "code_assistant": { ... }
  }
}
```

### POST /api/enterprise/research/invoke

Invoke the Research Agent for web search and synthesis.

**Request:**
```json
{
  "query": "What are the latest trends in AI?",
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "success": true,
  "response": "### Executive Summary\n...",
  "session_id": "research-session-123",
  "agent_type": "research",
  "tool_calls": null,
  "error": null
}
```

### POST /api/enterprise/content/invoke

Invoke the Content Generation Agent.

**Request:**
```json
{
  "topic": "AI in enterprise",
  "platform": "linkedin",
  "tone": "professional",
  "audience": "IT leaders",
  "session_id": "optional"
}
```

**Response:**
```json
{
  "success": true,
  "response": "Generated content...",
  "session_id": "content-session-123",
  "agent_type": "content",
  "tool_calls": null,
  "error": null
}
```

### POST /api/enterprise/documents/invoke

Invoke the Document Generation Agent.

**Request:**
```json
{
  "doc_type": "sop",
  "title": "Password Reset Procedure",
  "description": "Standard procedure for password resets",
  "sections": ["Prerequisites", "Steps", "Verification"],
  "session_id": "optional"
}
```

**Response:**
```json
{
  "success": true,
  "response": "# SOP: Password Reset Procedure\n...",
  "session_id": "doc-session-123",
  "agent_type": "document",
  "error": null
}
```

### POST /api/enterprise/code/invoke

Invoke the Code Assistant Agent.

**Request:**
```json
{
  "code": "def get_user(id):\n    query = f\"SELECT * FROM users WHERE id = {id}\"\n    return db.execute(query)",
  "language": "python",
  "action": "analyze",
  "include_security": true,
  "session_id": "optional"
}
```

**Response:**
```json
{
  "success": true,
  "response": "## Code Analysis\n### Security Issues\n- SQL Injection vulnerability...",
  "agent_type": "code_assistant",
  "error": null
}
```

### POST /api/enterprise/support/invoke

Invoke the HITL IT Support Agent.

**Request:**
```json
{
  "message": "I need to reset my password",
  "session_id": "optional",
  "user_id": "user-123"
}
```

**Response:**
```json
{
  "success": true,
  "response": "I can help with password reset...",
  "session_id": "support-session-123",
  "agent_type": "hitl_support",
  "error": null
}
```

---

## Webhook Endpoints

### POST /api/webhooks/copilot-studio

Microsoft Copilot Studio integration webhook.

**Request:**
```json
{
  "query": "What is LangChain?",
  "agent_type": "research",
  "session_id": "optional",
  "user_id": "copilot-user-123",
  "conversation_id": "conv-123",
  "channel": "teams",
  "metadata": {}
}
```

**Response:**
```json
{
  "success": true,
  "response": "LangChain is...",
  "session_id": "copilot-conv-123",
  "agent_type": "research",
  "source": "copilot-studio",
  "metadata": {
    "channel": "teams",
    "user_id": "copilot-user-123",
    "conversation_id": "conv-123"
  },
  "error": null
}
```

### POST /api/webhooks/azure-ai

Azure AI Agent integration webhook.

**Request:**
```json
{
  "query": "Generate a report",
  "agent_type": "document",
  "session_id": "optional",
  "deployment_id": "deployment-123",
  "resource_group": "rg-ai",
  "subscription_id": "sub-123",
  "metadata": {}
}
```

**Response:**
```json
{
  "success": true,
  "response": "Report content...",
  "session_id": "azure-deployment-123",
  "agent_type": "document",
  "source": "azure-ai",
  "metadata": {
    "deployment_id": "deployment-123",
    "resource_group": "rg-ai"
  },
  "error": null
}
```

### POST /api/webhooks/aws-lex

AWS Lex integration webhook.

**Request:**
```json
{
  "query": "Search for network incidents",
  "agent_type": "research",
  "session_id": "optional",
  "bot_id": "bot-123",
  "bot_alias_id": "alias-1",
  "locale_id": "en_US",
  "session_attributes": {},
  "request_attributes": {}
}
```

**Response:**
```json
{
  "success": true,
  "response": "Found 5 network incidents...",
  "session_id": "lex-bot-123",
  "agent_type": "research",
  "source": "aws-lex",
  "metadata": {
    "bot_id": "bot-123",
    "locale_id": "en_US"
  },
  "error": null
}
```

---

## Legacy Endpoints

### LangServe Endpoints

These endpoints use LangServe format (wrapped input):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat/invoke` | POST | Simple chat |
| `/rag/invoke` | POST | RAG query |
| `/agent/invoke` | POST | LangGraph agent |

**Request Format:**
```json
{
  "input": {
    "input": "Your message here"
  }
}
```

### Conversation API

For IT Support agents with session management:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /api/conversation/start` | Start conversation |
| `POST /api/conversation/chat` | Send message |
| `GET /api/conversation/{session_id}` | Get session |
| `DELETE /api/conversation/{session_id}` | End session |

---

## Error Handling

### Error Response Format

```json
{
  "success": false,
  "response": null,
  "error": "Error message here",
  "session_id": null,
  "agent_type": "research"
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 401 | Unauthorized - Invalid/missing API key |
| 404 | Not Found - Endpoint not found |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Agent not loaded |

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Invalid or missing API key" | Missing X-API-Key header | Add API key header |
| "Agent not available" | Agent failed to load | Check API keys |
| "No API key found" | LLM provider not configured | Set OPENAI_API_KEY |

---

## Testing with VSCode REST Client

Create `api-tests.http` file:

```http
### Variables
@baseUrl = http://localhost:8000
@apiKey = your-api-key

### Health Check
GET {{baseUrl}}/health

### List Agents
GET {{baseUrl}}/api/enterprise/agents
X-API-Key: {{apiKey}}

### Research Agent
POST {{baseUrl}}/api/enterprise/research/invoke
Content-Type: application/json
X-API-Key: {{apiKey}}

{
  "query": "What is LangChain?",
  "session_id": "test-001"
}
```

---

## Rate Limits

Current implementation does not enforce rate limits. For production:
- Consider implementing rate limiting middleware
- Use API gateway for traffic management
- Monitor usage via LangSmith

---

## Related Documentation

- [ARCHITECTURE.md](../ARCHITECTURE.md) - System design
- [SECURITY.md](../SECURITY.md) - Authentication details
