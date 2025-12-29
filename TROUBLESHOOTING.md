# Agent Troubleshooting & Fix Guide

**Last Updated**: 2025-12-19
**Version**: 1.0

---

## ROOT CAUSE IDENTIFIED

### The Problem
**API Key Middleware** is blocking ALL API requests even with `API_KEY_ENABLED=false` because:
1. The server must be **completely stopped and restarted** after changing `.env`
2. Uvicorn's hot reload **DOES NOT reload environment variables**
3. Old server instance keeps running with old `API_KEY_ENABLED=true` setting

### Symptoms
- ✗ IT Helpdesk: "Failed to start session: Unknown error"
- ✗ Enterprise Agents: "No response received. The agent may not be available."
- ✗ API calls return: `401 Unauthorized - Invalid or missing API key`

---

## COMPLETE FIX - Step by Step

### Step 1: Stop ALL Python Processes

**Windows PowerShell**:
```powershell
# Find process using port 8000
netstat -ano | findstr ":8000.*LISTENING"

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F

# OR kill all Python processes
Get-Process python | Stop-Process -Force
```

**Verify port is free**:
```bash
netstat -ano | findstr ":8000"
# Should return nothing
```

### Step 2: Verify Environment Configuration

```bash
cd deployment
grep "API_KEY_ENABLED" .env
```

**Expected output**:
```
API_KEY_ENABLED=false
```

**If it says `true`**, edit `.env` and change line 79:
```bash
# FROM:
API_KEY_ENABLED=true

# TO:
API_KEY_ENABLED=false
```

### Step 3: Start Fresh Server

```bash
cd deployment
python -m app.server
```

**Wait for startup messages** (should take 10-20 seconds):
```
============================================================
LangChain Platform Starting...
============================================================
[OK] LangSmith tracing enabled
[OK] LangChain chains loaded (OpenAI)
[OK] LangGraph agent loaded (OpenAI)
[OK] Document RAG chain loaded (OpenAI)
[OK] IT Support agents loaded: it_helpdesk, servicenow
[OK] Enterprise agents loaded: research, content, data_analyst...
============================================================
Platform ready!
  - API Docs: http://localhost:8000/docs
  - Chat UI:  http://localhost:8000/chat
============================================================
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### Step 4: Test Health Endpoint

```bash
curl http://localhost:8000/health
```

**Expected response**:
```json
{
    "status": "healthy",
    "it_support_loaded": true,
    "enterprise_agents_loaded": true
}
```

### Step 5: Test IT Support Agent

```bash
curl -X POST http://localhost:8000/api/conversation/start \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "it_helpdesk"}'
```

**Expected response** (NOT 401 error):
```json
{
    "session_id": "abc123-...",
    "agent_type": "it_helpdesk",
    "welcome_message": "Hello! I'm your IT Helpdesk Agent..."
}
```

### Step 6: Test Enterprise Agent

```bash
curl -X POST http://localhost:8000/api/enterprise/research/invoke \
  -H "Content-Type: application/json" \
  -d '{"query": "What is LangChain?"}'
```

**Expected response** (NOT 401 error):
```json
{
    "success": true,
    "response": "LangChain is a framework...",
    "agent_type": "research"
}
```

### Step 7: Test Web UI

1. Open browser: `http://localhost:8000/chat`
2. **Hard refresh**: Press `CTRL + SHIFT + R` (clears cache)
3. Select "IT Helpdesk Agent"
4. Type: "I need help resetting my password"
5. ✅ Should get response WITHOUT errors

6. Select "AI Research Agent"
7. Type: "What is LangGraph?"
8. ✅ Should get response WITHOUT "No response received" error

---

## Common Issues & Solutions

### Issue: Still Getting "401 Unauthorized"

**Cause**: Server not fully restarted

**Solution**:
1. Check if server is ACTUALLY running:
   ```bash
   curl http://localhost:8000/health
   ```
2. If connection refused, server didn't start - check for errors
3. If you get 401, server is running with OLD config:
   - Kill process completely (see Step 1)
   - Restart (see Step 3)
   - **DO NOT** rely on hot reload!

### Issue: Server Won't Start

**Symptom**: `curl: Failed to connect to localhost port 8000`

**Possible Causes**:
1. **Import Error**: Check for Python import issues
   ```bash
   cd deployment
   python -c "from app.server import app; print('OK')"
   ```

2. **Missing Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Port Already in Use**:
   ```bash
   # Find what's using port 8000
   netstat -ano | findstr ":8000"
   # Kill that process
   taskkill /PID <PID> /F
   ```

### Issue: "No response received" in UI

**Cause**: Browser cached old API responses with 401 errors

**Solution**:
1. Hard refresh: `CTRL + SHIFT + R`
2. Open DevTools (`F12`), go to Network tab
3. Check actual API responses - should NOT be 401
4. If still 401, server not restarted properly (see above)
5. Try incognito/private browsing mode

### Issue: IT Agent Says "Connected" But Won't Start

**Symptom**: Green "Connected" badge but dialog says "Failed to start session"

**Cause**: JavaScript received 401 error when calling `/api/conversation/start`

**Solution**:
1. Open browser DevTools (`F12`)
2. Go to Console tab
3. Look for errors
4. Go to Network tab
5. Try starting session again
6. Click on `/api/conversation/start` request
7. Check Response tab - should NOT be 401

**If you see 401**: Server needs restart (see Step 1-3 above)

---

### Issue: GraphRecursionError - "Recursion limit of 25/50/100 reached"

**Symptom**: Agent fails with error message like:
```
langgraph.errors.GraphRecursionError: Recursion limit of 50 reached without hitting a stop condition
```

**Common Scenarios**:
- Research Agent performing multiple web searches
- Complex multi-step agent workflows
- Verbose system prompts causing over-iteration

**Root Cause**: LangGraph agents making too many sequential tool calls, exceeding the configured recursion limit

**Solution 1: Increase Recursion Limit** (For Complex Workflows)

Modify the agent's `compile()` and `invoke()` methods:

```python
# In app/agents/your_agent/your_agent.py

def compile(self) -> None:
    """Compile the agent's graph with increased recursion limit."""
    from langgraph.checkpoint.memory import MemorySaver

    self._graph = self._build_graph()
    checkpointer = self.config.checkpointer or MemorySaver()
    self._compiled_graph = self._graph.compile(checkpointer=checkpointer)

    # Store increased recursion limit for complex workflows
    self._recursion_limit = 200  # Adjust based on your needs

def invoke(self, message: str, session_id: str | None = None, **kwargs) -> dict:
    """Invoke the agent with increased recursion limit."""
    if self._compiled_graph is None:
        self.compile()

    # Get recursion limit (default to 100)
    recursion_limit = getattr(self, '_recursion_limit', 100)

    # Configure with recursion limit
    config = {
        "configurable": {"thread_id": session_id or "default"},
        "recursion_limit": recursion_limit,  # Pass at invoke time
    }

    result = self._compiled_graph.invoke(input_state, config=config)
    return result
```

**Solution 2: Optimize System Prompt** (Reduce Iterations)

Add efficiency guidelines to your agent's system prompt:

```python
def _get_system_prompt(self) -> str:
    return """You are an AI agent...

## IMPORTANT: Efficiency Guidelines
- Use MAXIMUM 2-3 tool calls per query
- After gathering information, provide your final answer immediately
- Do NOT keep searching for more information - be decisive
- If you have enough information, synthesize and respond

[Rest of your system prompt...]
"""
```

**Solution 3: Clear Python Cache** (If Changes Not Applying)

If you've made code changes but still see the error:

```bash
# Windows PowerShell
Get-Process python* | Stop-Process -Force

# Clear all cache directories
cd deployment
rm -rf app/**/__pycache__

# Restart server WITHOUT hot reload
python -c "import uvicorn; from app.server import app; uvicorn.run(app, host='0.0.0.0', port=8000, reload=False)"
```

**IMPORTANT Configuration Rules**:
- ❌ **DO NOT** pass `recursion_limit` to `compile()` - it's not a valid parameter
- ✅ **DO** pass `recursion_limit` in config dict during `invoke()`
- ❌ **DO NOT** use `.with_config()` after compilation - can cause issues
- ✅ **DO** store limit as instance variable in `compile()`, apply in `invoke()`

**Test Your Fix**:
```bash
cd deployment
python test_agents_quick.py
```

**Reference**:
- Working example: [app/agents/research/research_agent.py](deployment/app/agents/research/research_agent.py)
- LangGraph docs: https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT
- Knowledge base: [deployment/KNOWLEDGE.md](deployment/KNOWLEDGE.md#issue-graphrecursionerror)

---

## Files Changed (For Reference)

### v3.2 Changes (2025-12-29)

| File | Lines | Change | Reason |
|------|-------|--------|--------|
| `deployment/app/agents/research/research_agent.py` | 229-271 | Override `compile()` with `_recursion_limit = 200` | Fix GraphRecursionError |
| `deployment/app/agents/research/research_agent.py` | 322-357 | Override `invoke()` with recursion config | Apply limit at invoke time |
| `deployment/app/agents/research/research_agent.py` | 247-271 | Optimized system prompt | Reduce tool iterations |
| `deployment/app/agents/it_helpdesk.py` | 483-490 | Changed LLM provider preference | Standardize to OpenAI |
| `deployment/app/agents/servicenow_agent.py` | 548-555 | Changed LLM provider preference | Standardize to OpenAI |
| `deployment/app/server.py` | 1461-1475 | Fixed method call to `create_document()` | Correct API usage |

### v3.1 Changes (2025-12-19)

| File | Line | Change | Reason |
|------|------|--------|--------|
| `deployment/app/agents/it_helpdesk.py` | 641 | Removed global instantiation | Load AFTER .env |
| `deployment/app/agents/servicenow_agent.py` | 653 | Removed global instantiation | Load AFTER .env |
| `deployment/.env` | 79 | `API_KEY_ENABLED=false` | Disable for local dev |

---

## Quick Reference Commands

```bash
# 1. Kill server
netstat -ano | findstr ":8000.*LISTENING"
taskkill /PID <PID> /F

# 2. Verify config
cd deployment && grep "API_KEY_ENABLED" .env

# 3. Start server
python -m app.server

# 4. Test (in new terminal)
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/conversation/start -H "Content-Type: application/json" -d '{"agent_type": "it_helpdesk"}'

# 5. Open UI
# Browse to: http://localhost:8000/chat
```

---

##Production Deployment

When deploying to production or exposing via ngrok:

1. **Enable API Key**:
```bash
# In .env
API_KEY_ENABLED=true
API_KEY=your-secure-random-key-here-min-32-chars
```

2. **Update Chat UI** (`app/static/chat.html`):
```javascript
// Add to all fetch requests
headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'your-secure-random-key-here'
}
```

3. **Security Checklist**:
- [ ] Use HTTPS (not HTTP)
- [ ] Set CORS_ORIGINS to specific domains (not `*`)
- [ ] Use strong API key (32+ random characters)
- [ ] Enable rate limiting
- [ ] Monitor logs for suspicious activity

---

## Need Help?

1. **Check server logs** for detailed errors
2. **Review**: [deployment/KNOWLEDGE.md](deployment/KNOWLEDGE.md)
3. **Development standards**: [.claude/CLAUDE.md](.claude/CLAUDE.md)
4. **GitHub issues**: https://github.com/anthropics/claude-code/issues

---

## Change Log

### 2025-12-29 - v3.2
- Added GraphRecursionError troubleshooting guidance
- Documented LangGraph recursion limit configuration patterns
- Added system prompt optimization best practices
- Documented Python bytecode caching issues and solutions
- Added IT Support agents OpenAI provider standardization
- Fixed Document Generator method call error

**Key Fixes**:
- Research Agent recursion limit properly configured (200 steps)
- All IT Support agents now use OpenAI as primary provider
- Document Generator uses correct `create_document()` method
- All agents tested and verified working

**Reference**: See [deployment/KNOWLEDGE.md](deployment/KNOWLEDGE.md) for detailed technical documentation

---

### 2025-12-19 - v1.0
- Initial comprehensive troubleshooting guide
- Root cause: API key middleware blocking local development
- Solution: Complete server restart after .env changes
- Consolidated all fixes into single document

---

*Keep this document updated as issues are discovered and resolved.*
