# Operations Runbook

> **Last Updated**: 2025-12-19
> **Version**: 2.0.0
> **Audience**: Operations Team, SRE, DevOps

---

## Table of Contents

1. [System Monitoring](#system-monitoring)
2. [Common Issues & Solutions](#common-issues--solutions)
3. [Maintenance Procedures](#maintenance-procedures)
4. [Backup & Recovery](#backup--recovery)
5. [Scaling Guidelines](#scaling-guidelines)
6. [Performance Tuning](#performance-tuning)
7. [Incident Response](#incident-response)
8. [Runbook Procedures](#runbook-procedures)

---

## System Monitoring

### Key Metrics

| Metric | Source | Threshold | Alert Level |
|--------|--------|-----------|-------------|
| Health status | `/health` | status != "healthy" | Critical |
| Response time | LangSmith | > 30s | Warning |
| Error rate | LangSmith | > 5% | Warning |
| Memory usage | Docker stats | > 80% | Warning |
| Container restarts | Docker | > 3 in 5 min | Critical |
| API latency (P99) | LangSmith | > 60s | Critical |

### Monitoring Commands

```bash
# Container health
docker ps --filter name=langchain-platform --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Real-time stats
docker stats langchain-platform --no-stream

# Recent logs
docker logs langchain-platform --tail 100 --since 5m

# Health check
curl -s http://localhost:8000/health | jq .

# API status
curl -s http://localhost:8000/api/enterprise/agents | jq '.status'
```

### LangSmith Dashboard

1. Navigate to https://smith.langchain.com
2. Select project: `langchain-platform` (or configured project name)
3. Monitor:
   - **Traces**: Individual request traces
   - **Latency**: Response time distribution
   - **Errors**: Failed requests
   - **Tokens**: LLM token usage

---

## Common Issues & Solutions

### Issue 1: Container Fails to Start

**Symptoms:**
- Container exits immediately
- Status shows "Exited (1)"

**Diagnosis:**
```bash
docker compose logs langchain-platform
```

**Common Causes & Solutions:**

| Cause | Log Message | Solution |
|-------|-------------|----------|
| Missing API key | "No API key found" | Add OPENAI_API_KEY or ANTHROPIC_API_KEY to .env |
| Port in use | "Address already in use" | `lsof -i :8000` then kill process |
| Python syntax | "SyntaxError" | Check code changes, rebuild image |
| Memory issue | "Killed" | Increase memory limits |

### Issue 2: Agents Not Loading

**Symptoms:**
- Health check shows `enterprise_agents_loaded: false`
- 503 errors on agent endpoints

**Diagnosis:**
```bash
curl -s http://localhost:8000/health | jq '{enterprise_agents_loaded, chains_loaded}'
docker logs langchain-platform | grep -i "failed to load"
```

**Solutions:**

| Cause | Solution |
|-------|----------|
| No API keys | Add at least one LLM provider key |
| Import error | Check agent module imports |
| Dependency issue | Rebuild Docker image |

### Issue 3: Slow Response Times

**Symptoms:**
- Requests taking > 30 seconds
- Timeouts on webhook calls

**Diagnosis:**
```bash
# Check LangSmith traces for slow spans
# Look for:
# - LLM call latency
# - Tool execution time
# - Network delays
```

**Solutions:**

| Cause | Solution |
|-------|----------|
| LLM latency | Use faster model (gpt-4o-mini vs gpt-4) |
| Tool timeout | Implement timeout on external calls |
| Large context | Reduce message history size |

### Issue 4: 401 Unauthorized Errors

**Symptoms:**
- All API calls return 401
- "Invalid or missing API key" message

**Diagnosis:**
```bash
# Check if API key auth is enabled
docker exec langchain-platform printenv | grep API_KEY

# Test with API key header
curl -H "X-API-Key: your-key" http://localhost:8000/api/enterprise/agents
```

**Solutions:**

| Cause | Solution |
|-------|----------|
| Wrong API key | Verify API_KEY in .env matches request header |
| Auth disabled | Set API_KEY_ENABLED=true |
| Header missing | Add X-API-Key header to requests |

### Issue 5: Memory Exhaustion

**Symptoms:**
- OOM (Out of Memory) kills
- Container restarts
- Slow performance

**Diagnosis:**
```bash
docker stats langchain-platform --no-stream
docker inspect langchain-platform | jq '.[0].HostConfig.Memory'
```

**Solutions:**
```yaml
# Increase memory in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G
    reservations:
      memory: 1G
```

---

## Maintenance Procedures

### Routine Maintenance Checklist

**Daily:**
- [ ] Check health endpoint status
- [ ] Review error logs
- [ ] Verify LangSmith traces

**Weekly:**
- [ ] Review container resource usage
- [ ] Check for pending updates
- [ ] Review LangSmith metrics

**Monthly:**
- [ ] Update dependencies (security patches)
- [ ] Review and rotate API keys
- [ ] Clean up old Docker images
- [ ] Review access logs

### Container Restart Procedure

```bash
# Graceful restart
docker compose restart langchain-platform

# Full restart (if issues persist)
docker compose down
docker compose up -d

# Verify health
sleep 10
curl -s http://localhost:8000/health | jq '.status'
```

### Image Update Procedure

```bash
# Pull latest code
git pull origin main

# Build new image
docker compose build --no-cache

# Tag current as backup
docker tag deployment-langchain-platform:latest deployment-langchain-platform:backup-$(date +%Y%m%d)

# Deploy new image
docker compose down
docker compose up -d

# Verify deployment
sleep 10
curl -s http://localhost:8000/health
```

### Log Rotation

```bash
# Docker handles log rotation by default
# Configure in daemon.json or docker-compose.yml:

# Option 1: Docker Compose logging
services:
  langchain-platform:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

---

## Backup & Recovery

### What to Backup

| Component | Location | Frequency |
|-----------|----------|-----------|
| Environment file | `.env` | On change |
| Docker images | Registry | On build |
| Configuration | `docker-compose.yml` | On change |
| Conversation state* | MemorySaver | N/A (in-memory) |

*For production, use persistent checkpointer (Redis/PostgreSQL)

### Backup Procedures

```bash
# Backup environment (excluding secrets)
cp .env .env.backup.$(date +%Y%m%d)

# Backup Docker image
docker save deployment-langchain-platform:latest | gzip > backup-$(date +%Y%m%d).tar.gz

# Upload to secure storage (example: Azure Blob)
az storage blob upload -f backup-*.tar.gz -c backups -n langchain/
```

### Recovery Procedures

```bash
# Restore from image backup
gunzip -c backup-20251219.tar.gz | docker load

# Restore environment
cp .env.backup.20251219 .env

# Start container
docker compose up -d
```

---

## Scaling Guidelines

### Horizontal Scaling

**When to Scale:**
- Request queue building up
- Response latency increasing
- CPU > 80% sustained

**Docker Compose Scaling:**
```bash
# Scale to 3 replicas
docker compose up -d --scale langchain-platform=3
```

**Kubernetes Scaling:**
```bash
# Manual scaling
kubectl scale deployment/langchain-platform --replicas=3 -n langchain

# Autoscaling
kubectl autoscale deployment/langchain-platform \
  --min=2 --max=10 --cpu-percent=70 -n langchain
```

### Vertical Scaling

**Resource Adjustment:**
```yaml
# docker-compose.yml
deploy:
  resources:
    limits:
      cpus: "4"
      memory: 4G
    reservations:
      cpus: "1"
      memory: 1G
```

### Scaling Considerations

| Factor | Impact | Recommendation |
|--------|--------|----------------|
| LLM API limits | Rate limiting | Use multiple API keys |
| Stateless design | Easy scaling | Ensure no server state |
| Memory per instance | Container sizing | Start with 2GB, adjust |
| Network latency | External tools | Cache when possible |

---

## Performance Tuning

### LLM Optimization

| Setting | Impact | Recommendation |
|---------|--------|----------------|
| Model selection | Latency vs quality | gpt-4o-mini for speed |
| Temperature | Response variability | 0.7 for balance |
| Max tokens | Response length | Set appropriate limits |

### Connection Pool Tuning

```python
# The LLM clients handle connection pooling automatically
# For custom HTTP clients, configure:
import httpx
client = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
)
```

### Memory Optimization

- Limit conversation history length
- Clear completed sessions
- Use streaming for large responses

---

## Incident Response

### Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| P1 - Critical | Complete outage | 15 min | All agents down |
| P2 - High | Major degradation | 1 hour | One agent failing |
| P3 - Medium | Minor impact | 4 hours | Slow responses |
| P4 - Low | Minimal impact | 24 hours | Non-critical warning |

### Incident Response Procedure

```
1. DETECT
   - Alert received or user report
   - Verify issue exists

2. ASSESS
   - Check health endpoint
   - Review logs
   - Identify scope

3. CONTAIN
   - Isolate affected component if needed
   - Enable maintenance mode if required

4. RESOLVE
   - Apply fix or workaround
   - Restart services if needed

5. VERIFY
   - Confirm resolution
   - Test affected functionality

6. DOCUMENT
   - Update incident log
   - Create post-mortem if P1/P2
```

### Emergency Contacts

| Role | Responsibility |
|------|----------------|
| On-call Engineer | First response |
| Platform Lead | Escalation |
| Security Team | Security incidents |

---

## Runbook Procedures

### RB-001: Restart Service

```bash
#!/bin/bash
# Purpose: Restart the langchain-platform service
# Use when: Performance degradation or non-critical issues

echo "Stopping service..."
docker compose stop langchain-platform

echo "Starting service..."
docker compose start langchain-platform

echo "Waiting for health check..."
sleep 10

echo "Checking health..."
curl -s http://localhost:8000/health | jq '.status'
```

### RB-002: Emergency Rollback

```bash
#!/bin/bash
# Purpose: Rollback to previous known-good version
# Use when: Critical issue after deployment

echo "Stopping current version..."
docker compose down

echo "Restoring backup image..."
docker tag deployment-langchain-platform:backup deployment-langchain-platform:latest

echo "Starting service..."
docker compose up -d

echo "Verifying health..."
sleep 15
curl -s http://localhost:8000/health | jq '.'
```

### RB-003: Clear Stuck Sessions

```bash
#!/bin/bash
# Purpose: Clear in-memory sessions
# Use when: Sessions not responding

echo "Restarting container to clear memory..."
docker compose restart langchain-platform

echo "Waiting for startup..."
sleep 15

echo "Verifying agents loaded..."
curl -s http://localhost:8000/api/enterprise/agents | jq '.status'
```

### RB-004: Enable Debug Logging

```bash
#!/bin/bash
# Purpose: Enable verbose logging for troubleshooting
# Use when: Need detailed diagnostics

# Restart with debug environment
docker compose down

# Add debug flag to .env temporarily
echo "LOG_LEVEL=DEBUG" >> .env

docker compose up -d

# View debug logs
docker logs -f langchain-platform
```

### RB-005: API Key Rotation

```bash
#!/bin/bash
# Purpose: Rotate API keys without downtime
# Use when: Security policy or key compromise

echo "Updating .env with new keys..."
# Edit .env with new API keys

echo "Rebuilding container..."
docker compose build

echo "Restarting with new keys..."
docker compose up -d

echo "Verifying..."
curl -s http://localhost:8000/health | jq '.enterprise_agents_loaded'
```

---

## Related Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md) - System architecture
- [DEPLOYMENT.md](./DEPLOYMENT.md) - Deployment guide
- [SECURITY.md](./SECURITY.md) - Security guidelines
