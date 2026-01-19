# Copilot Studio Integration - Quick Reference

**Last Updated:** January 12, 2026

> **⚠️ Important:** Replace `your-subdomain` in all URLs below with your actual ngrok subdomain. Get your ngrok URL by running `curl http://localhost:4040/api/tunnels` or checking the ngrok dashboard.

## 🚀 Essential Information

### Ngrok URL (Public Endpoint)
```
https://your-subdomain.ngrok-free.dev
```

### API Key Location
```
deployment/.env
Look for: API_KEY=your-key-here
```

### CORS Configuration
```
deployment/.env
CORS_ORIGINS=http://localhost:8000,http://localhost:3000,https://your-subdomain.ngrok-free.dev
```

---

## 📋 Available Endpoints

### 1. Copilot Studio Webhook (Recommended)

**URL:** `https://your-subdomain.ngrok-free.dev/api/webhooks/copilot-studio`

**Method:** POST

**Headers:**
```json
{
  "Content-Type": "application/json",
  "X-API-Key": "your-api-key-from-env-file",
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
  "channel": "copilot-studio"
}
```

**Agent Types:**
- `research` - Web search and analysis
- `content` - Social media content
- `data-analyst` - Excel/CSV analysis
- `document` - SOP/Policy generation
- `multilingual-rag` - Document Q&A
- `hitl-support` - IT support with approval
- `code-assistant` - Code generation

### 2. Enterprise Agents (Direct)

**Base URL Pattern:**
```
https://your-subdomain.ngrok-free.dev/api/enterprise/{agent}/invoke
```

**Examples:**
- Research: `/api/enterprise/research/invoke`
- Content: `/api/enterprise/content/invoke`
- Data Analyst: `/api/enterprise/data-analyst/invoke`

**Request:**
```json
{
  "query": "Your question here",
  "session_id": "optional-session-id"
}
```

### 3. Deep Agent (IT Operations)

**Start Session:**
```
POST /api/deepagent/start
Body: {"user_id": "copilot_user"}
```

**Chat:**
```
POST /api/deepagent/chat
Body: {"session_id": "uuid", "message": "Show P1 incidents"}
```

**Streaming:**
```
POST /api/deepagent/chat/stream
(Server-Sent Events)
```

### 4. IT Support Agents

**Start Conversation:**
```
POST /api/conversation/start
Body: {"agent_type": "it_helpdesk"}
```

**Send Message:**
```
POST /api/conversation/chat
Body: {"session_id": "uuid", "message": "Help with password"}
```

---

## 🧪 Quick Tests

### Test 1: Health Check
```powershell
curl -H "ngrok-skip-browser-warning: true" `
  https://your-subdomain.ngrok-free.dev/health
```

### Test 2: List Agents
```powershell
curl -H "ngrok-skip-browser-warning: true" `
  https://your-subdomain.ngrok-free.dev/api/enterprise/agents
```

### Test 3: Copilot Studio Webhook
```powershell
$apiKey = (Get-Content deployment\.env | Select-String "^API_KEY=").ToString().Split("=")[1]
$body = @{
    query = "What are AI trends in 2026?"
    agent_type = "research"
    user_id = "test_user"
    conversation_id = "test_001"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "https://your-subdomain.ngrok-free.dev/api/webhooks/copilot-studio" `
  -Method POST `
  -Headers @{
      "Content-Type" = "application/json"
      "X-API-Key" = $apiKey
      "ngrok-skip-browser-warning" = "true"
  } `
  -Body $body
```

---

## 🔧 Copilot Studio Setup (Step-by-Step)

### Step 1: Create HTTP Action

1. Go to https://copilotstudio.microsoft.com
2. Select your Copilot
3. Add node → **Call an action** → **HTTP**

### Step 2: Configure Action

**Method:** POST
**URL:**
```
https://your-subdomain.ngrok-free.dev/api/webhooks/copilot-studio
```

**Headers (Add 3 headers):**
1. `Content-Type: application/json`
2. `X-API-Key: your-api-key-from-env-file`
3. `ngrok-skip-browser-warning: true`

**Body:**
```json
{
  "query": "{UserQuery}",
  "agent_type": "research",
  "user_id": "{System.User.Id}",
  "conversation_id": "{System.Conversation.Id}",
  "channel": "copilot-studio"
}
```

### Step 3: Handle Response

**Add Condition:**
- If `response.success == true`:
  - Show message: `{response.response}`
- Else:
  - Show message: `Error: {response.error}`

---

## 🎯 Agent Selection Guide

| Use Case | Agent Type | Why |
|----------|-----------|-----|
| Web research, market analysis | `research` | Uses Tavily search, synthesizes findings |
| LinkedIn/X posts | `content` | Optimized for social media formats |
| Excel data analysis | `data-analyst` | Pandas, visualization |
| Create SOPs, policies | `document` | Structured document generation |
| Document Q&A (multilingual) | `multilingual-rag` | 50+ languages, RAG-based |
| IT support with approval | `hitl-support` | Human-in-the-loop workflow |
| Code generation | `code-assistant` | Application modernization |
| Incident management | **Use Deep Agent** | `/api/deepagent/*` |

---

## 🔒 Security Checklist

- [x] API Key configured in `.env`
- [x] CORS includes ngrok URL
- [x] HTTPS only (ngrok provides)
- [x] API key in Copilot Studio headers
- [ ] Consider rate limiting for production
- [ ] Monitor LangSmith traces

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| 401 Unauthorized | Add `X-API-Key` header with correct value |
| CORS error | Add ngrok URL to `CORS_ORIGINS` in `.env`, restart container |
| Ngrok tunnel not found | Check `curl http://localhost:4040/api/tunnels` |
| Agent not loaded | Verify `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in `.env` |
| Slow responses | Use streaming endpoints, increase Copilot Studio timeout |

---

## 📊 Monitoring

### Check Logs
```powershell
docker logs langchain-platform -f
```

### Filter for Copilot Studio Requests
```powershell
docker logs langchain-platform 2>&1 | Select-String "copilot-studio"
```

### LangSmith Tracing
```
https://smith.langchain.com
Project: langchain-platform
```

---

## 🔄 Maintenance Commands

### Restart Container
```powershell
cd deployment
docker compose restart
```

### Update CORS
```powershell
# Edit deployment/.env
docker compose restart
```

### Check Ngrok Status
```powershell
curl http://localhost:4040/api/tunnels | ConvertFrom-Json
```

---

## 📚 Full Documentation

- **Complete Guide:** [COPILOT_STUDIO_INTEGRATION.md](COPILOT_STUDIO_INTEGRATION.md)
- **Platform Docs:** [../KNOWLEDGE.md](../KNOWLEDGE.md)
- **Ngrok Setup:** [NGROK_SETUP.md](NGROK_SETUP.md)
- **API Reference:** [api/README.md](api/README.md)

---

**Status:** ✅ Ready for Copilot Studio Integration
**Tested:** January 12, 2026
**Ngrok URL Valid Until:** Session ends (use paid plan for persistent URL)
