# Deep Agent Docker Deployment with Ngrok

This guide provides step-by-step instructions to rebuild the Docker image with the latest Deep Agent changes and expose it via ngrok.

## Prerequisites

- Docker Desktop installed and running
- Ngrok account and authtoken (from https://dashboard.ngrok.com/get-started/your-authtoken)
- `.env` file configured with API keys in `deployment/` folder

---

## Part 1: Rebuild Docker Image with Deep Agent

### Step 1: Navigate to Deployment Folder

```powershell
cd "c:\Users\a833555\OneDrive - ATOS\Gitwork\langchain-agents\deployment"
```

### Step 2: Stop Running Containers (if any)

```powershell
docker-compose down
```

### Step 3: Rebuild Docker Image

**Option A: Clean rebuild (recommended for major changes)**
```powershell
# Remove old image
docker-compose down --rmi all

# Rebuild from scratch
docker-compose build --no-cache
```

**Option B: Quick rebuild (faster)**
```powershell
docker-compose build
```

### Step 4: Verify Image Built Successfully

```powershell
docker images | Select-String "langchain-platform"
```

Expected output should show the newly built image with recent timestamp.

### Step 5: Start the Container

```powershell
docker-compose up -d
```

### Step 6: Verify Container is Running

```powershell
# Check container status
docker ps | Select-String "langchain-platform"

# Check logs
docker-compose logs -f --tail=50
```

**Expected logs should show:**
- `[OK] LangSmith tracing enabled`
- `[OK] LangChain chains loaded (OpenAI)`
- `[OK] Deep Agent loaded successfully`
- `Platform ready!`

Press `Ctrl+C` to stop following logs.

### Step 7: Test Deep Agent Locally

```powershell
# Test health endpoint
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET

# Test Deep Agent start
$body = @{
    user_id = "test_user"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/deepagent/start" -Method POST -Headers @{"Content-Type"="application/json"} -Body $body
```

---

## Part 2: Expose with Ngrok

### Step 1: Install Ngrok (if not installed)

Download from: https://ngrok.com/download

Or use Chocolatey:
```powershell
choco install ngrok
```

### Step 2: Authenticate Ngrok

```powershell
ngrok config add-authtoken YOUR_AUTHTOKEN_HERE
```

Replace `YOUR_AUTHTOKEN_HERE` with your actual authtoken from https://dashboard.ngrok.com/get-started/your-authtoken

### Step 3: Start Ngrok Tunnel

**Basic tunnel (HTTP only):**
```powershell
ngrok http 8000
```

**Advanced tunnel with custom subdomain (requires paid plan):**
```powershell
ngrok http 8000 --subdomain=langchain-deepagent
```

**Tunnel with additional features:**
```powershell
ngrok http 8000 `
  --log=stdout `
  --log-level=info `
  --region=us
```

### Step 4: Note the Forwarding URLs

Ngrok will display output like:
```
Forwarding    https://abc123-xyz.ngrok-free.app -> http://localhost:8000
```

Copy the HTTPS URL (e.g., `https://abc123-xyz.ngrok-free.app`)

### Step 5: Test Deep Agent via Ngrok

**PowerShell:**
```powershell
# Start session
$ngrokUrl = "https://YOUR-NGROK-URL.ngrok-free.app"
$headers = @{
    "ngrok-skip-browser-warning" = "true"
    "Content-Type" = "application/json"
}
$body = @{
    user_id = "ops_user"
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "$ngrokUrl/api/deepagent/start" -Method POST -Headers $headers -Body $body
$sessionId = $response.session_id
Write-Host "Session ID: $sessionId"

# Send message
$chatBody = @{
    session_id = $sessionId
    message = "Show me all P1 incidents from last 7 days"
} | ConvertTo-Json

Invoke-RestMethod -Uri "$ngrokUrl/api/deepagent/chat" -Method POST -Headers $headers -Body $chatBody
```

**Curl (from bash/Git Bash):**
```bash
# Start session
curl -X POST https://YOUR-NGROK-URL.ngrok-free.app/api/deepagent/start \
  -H "ngrok-skip-browser-warning: true" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "ops_user"}'

# Chat with agent (use session_id from above)
curl -X POST https://YOUR-NGROK-URL.ngrok-free.app/api/deepagent/chat \
  -H "ngrok-skip-browser-warning: true" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "deepagent-abc123...",
    "message": "Investigate INC0010001 and check for related incidents"
  }'
```

---

## Part 3: Test Streaming Endpoint

### PowerShell with SSE Streaming

```powershell
$ngrokUrl = "https://YOUR-NGROK-URL.ngrok-free.app"
$sessionId = "YOUR-SESSION-ID"

# Note: PowerShell doesn't natively support SSE, use curl instead
```

### Curl with SSE Streaming

```bash
# Stream Deep Agent response
curl -N -X POST https://YOUR-NGROK-URL.ngrok-free.app/api/deepagent/chat/stream \
  -H "ngrok-skip-browser-warning: true" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "deepagent-abc123...",
    "message": "Analyze P1 incidents from this week and identify patterns"
  }'
```

**Expected streaming events:**
```
event: thinking
data: {"content": "I'll analyze the P1 incidents..."}

event: tool_call
data: {"tool": "search_incidents", "args": {...}, "description": "Searching for P1 incidents"}

event: tool_result
data: {"tool": "search_incidents", "result": "Found 4 incidents"}

event: content
data: {"response": "Analysis complete. Found pattern..."}

event: done
data: {"session_id": "...", "todos_updated": true}
```

---

## Part 4: Access Web UI via Ngrok

### Open in Browser

Navigate to: `https://YOUR-NGROK-URL.ngrok-free.app/chat`

**Features available:**
- Select "IT Ops Deep Agent" from dropdown
- Real-time streaming responses
- Task progress panel (shows todos)
- Context files panel
- Quick action buttons

---

## Troubleshooting

### Issue: Container won't start

**Check logs:**
```powershell
docker-compose logs
```

**Common causes:**
- Missing API keys in `.env` file
- Port 8000 already in use
- Insufficient Docker resources

**Solution:**
```powershell
# Stop conflicting processes
Get-Process | Where-Object {$_.ProcessName -like "*python*"} | Stop-Process -Force

# Restart Docker Desktop
# Then rebuild and start
docker-compose down
docker-compose up -d
```

### Issue: Ngrok tunnel not accessible

**Check:**
- Firewall settings
- Docker container is running: `docker ps`
- Local URL works: `curl http://localhost:8000/health`

**Solution:**
```powershell
# Restart ngrok with debug logging
ngrok http 8000 --log=stdout --log-level=debug
```

### Issue: Deep Agent returns 503 Service Unavailable

**Cause:** API keys not loaded

**Solution:**
```powershell
# Verify .env file exists and contains:
Get-Content .env | Select-String "OPENAI_API_KEY"

# Restart container
docker-compose restart
docker-compose logs -f
```

### Issue: Ngrok shows "Visit Site" button

**Cause:** Ngrok free plan warning page

**Solution:**
Add header to bypass: `-H "ngrok-skip-browser-warning: true"`

---

## Production Deployment Checklist

Before deploying to production:

- [ ] Enable API key authentication
  ```env
  API_KEY_ENABLED=true
  API_KEY=your-secure-api-key-here
  ```

- [ ] Configure CORS for specific origins
  ```env
  CORS_ORIGINS=https://your-domain.com,https://app.your-domain.com
  ```

- [ ] Set up SSL/TLS with custom domain (ngrok paid plan)

- [ ] Configure resource limits in docker-compose.yml

- [ ] Set up monitoring and logging

- [ ] Configure persistent volume backups
  ```powershell
  docker volume ls
  docker run --rm -v deployment_deepagent_data:/data -v ${PWD}/backup:/backup alpine tar czf /backup/deepagent_backup.tar.gz /data
  ```

- [ ] Document incident response procedures

---

## Useful Commands

### Docker Management

```powershell
# View container logs
docker-compose logs -f langchain-platform

# Restart container
docker-compose restart

# Stop and remove container
docker-compose down

# View container stats
docker stats langchain-platform

# Execute command in container
docker-compose exec langchain-platform python -c "import sys; print(sys.version)"

# Inspect Deep Agent volume
docker volume inspect deployment_deepagent_data
```

### Ngrok Management

```powershell
# List active tunnels
ngrok tunnels list

# View tunnel details
ngrok diagnose

# Stop tunnel
# Press Ctrl+C in ngrok terminal
```

---

## Architecture Overview

```
┌─────────────────┐
│   Web Browser   │
│   or Client     │
└────────┬────────┘
         │ HTTPS
         ▼
┌─────────────────┐
│  Ngrok Tunnel   │
│  (Public URL)   │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│  Docker Container│
│  langchain-      │
│  platform:8000   │
│  ┌─────────────┐│
│  │ FastAPI     ││
│  │ Server      ││
│  └──────┬──────┘│
│         │       │
│  ┌──────▼──────┐│
│  │ Deep Agent  ││
│  │ + Subagents ││
│  └──────┬──────┘│
│         │       │
│  ┌──────▼──────┐│
│  │ Persistent  ││
│  │ Storage     ││
│  └─────────────┘│
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  OpenAI API     │
│  LangSmith      │
│  ServiceNow     │
└─────────────────┘
```

---

## Security Best Practices

1. **Never expose .env file**
   - Ensure `.env` is in `.dockerignore`
   - Use Docker secrets for production

2. **Enable authentication**
   - Set `API_KEY_ENABLED=true`
   - Use strong, unique API keys
   - Rotate keys regularly

3. **Restrict CORS**
   - Don't use `CORS_ORIGINS=*` in production
   - Whitelist specific domains only

4. **Monitor access logs**
   ```powershell
   docker-compose logs | Select-String "POST /api/deepagent"
   ```

5. **Use HTTPS only**
   - Ngrok provides HTTPS by default
   - Never use HTTP for production

6. **Implement rate limiting**
   - Configure in FastAPI middleware
   - Use ngrok rate limiting (paid plans)

---

## Support

For issues or questions:
- Check logs: `docker-compose logs -f`
- Review documentation: `deployment/docs/`
- Check GitHub issues
- Contact: support@your-organization.com
