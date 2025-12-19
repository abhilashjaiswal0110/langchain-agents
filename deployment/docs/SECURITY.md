# Security Guidelines

> **Last Updated**: 2025-12-19
> **Version**: 2.0.0
> **Classification**: Internal Use

---

## Table of Contents

1. [Security Overview](#security-overview)
2. [Authentication & Authorization](#authentication--authorization)
3. [Data Protection](#data-protection)
4. [Secrets Management](#secrets-management)
5. [Network Security](#network-security)
6. [Input Validation](#input-validation)
7. [Security Best Practices](#security-best-practices)
8. [Compliance Requirements](#compliance-requirements)
9. [Incident Response](#incident-response)
10. [Security Checklist](#security-checklist)

---

## Security Overview

### Security Principles

This platform follows the **Defense in Depth** approach:

1. **API Key Authentication** - First line of defense for external access
2. **Input Validation** - Pydantic models for all request/response
3. **Secrets Management** - Environment variables only, never hardcoded
4. **Least Privilege** - Non-root container user
5. **Observability** - Full tracing for security auditing

### Threat Model

| Threat | Risk Level | Mitigation |
|--------|------------|------------|
| Unauthorized API access | High | API key authentication |
| Prompt injection | Medium | Input sanitization |
| Data leakage | Medium | No PII logging |
| Denial of service | Medium | Rate limiting |
| Supply chain attack | Low | Dependency scanning |

---

## Authentication & Authorization

### API Key Authentication

The platform implements middleware-based API key authentication for protected endpoints:

```python
# Middleware validates X-API-Key header
class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not API_KEY_ENABLED:
            return await call_next(request)

        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key != API_KEY:
            return HTMLResponse(
                content='{"detail": "Invalid or missing API key"}',
                status_code=401
            )

        return await call_next(request)
```

### Public vs Protected Endpoints

| Endpoint Category | Authentication | Examples |
|-------------------|----------------|----------|
| Health/Status | Public | `/health`, `/ready`, `/docs` |
| Static Assets | Public | `/chat`, `/static/*` |
| Enterprise APIs | Protected | `/api/enterprise/*` |
| Webhooks | Protected | `/api/webhooks/*` |
| Conversation | Protected | `/api/conversation/*` |

### Enabling Authentication

```bash
# Required environment variables
API_KEY_ENABLED=true
API_KEY=your-secure-api-key-here

# Generate secure key (recommended)
openssl rand -hex 32
```

### Request Authentication

```bash
# All protected endpoints require X-API-Key header
curl -X POST "https://your-endpoint/api/enterprise/research/invoke" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"query": "What is LangChain?"}'
```

---

## Data Protection

### Data Classification

| Data Type | Classification | Handling |
|-----------|----------------|----------|
| API Keys | Secret | Environment variables only |
| User queries | Internal | Not persisted, traced in LangSmith |
| Agent responses | Internal | Transient, in-memory only |
| Session data | Internal | Cleared on restart (MemorySaver) |
| Logs | Internal | No PII, rotate regularly |

### Data in Transit

- All external traffic should use HTTPS (via ngrok or load balancer)
- Internal Docker network traffic is isolated
- LLM API calls use provider's TLS encryption

### Data at Rest

- No persistent storage by default (MemorySaver is in-memory)
- For production persistence:
  - Use encrypted database connections
  - Enable encryption at rest
  - Implement key rotation

### PII Handling

```python
# Tool outputs are sanitized
from app.agents.base.tools import sanitize_output

@tool
@tool_error_handler
def search_knowledge_base(query: str) -> str:
    result = perform_search(query)
    return sanitize_output(result)  # Removes potential PII
```

---

## Secrets Management

### Environment Variables

All secrets MUST be provided via environment variables:

```ini
# .env (NEVER commit to git)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LANGCHAIN_API_KEY=lsv2_...
API_KEY=your-secure-key
TAVILY_API_KEY=tvly-...
```

### Git Exclusions

```gitignore
# .gitignore - CRITICAL
.env
.env.*
!.env.example
*.pem
*.key
credentials.json
secrets.json
```

### Secret Rotation

| Secret | Rotation Frequency | Procedure |
|--------|-------------------|-----------|
| API_KEY | Monthly | Update .env, restart container |
| OPENAI_API_KEY | Quarterly | Rotate in OpenAI dashboard |
| LANGCHAIN_API_KEY | Quarterly | Rotate in LangSmith |

### Production Secret Management

For production environments, use:
- **Azure Key Vault**
- **AWS Secrets Manager**
- **HashiCorp Vault**
- **Kubernetes Secrets** (with encryption)

```yaml
# Kubernetes secret reference
envFrom:
  - secretRef:
      name: langchain-secrets
```

---

## Network Security

### CORS Configuration

```python
# Configured in server.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "X-API-Key"],
)
```

### Production CORS Recommendations

```bash
# Restrict to specific origins
CORS_ORIGINS=https://app.example.com,https://admin.example.com
```

### Container Security

```dockerfile
# Run as non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Minimal base image
FROM python:3.11-slim
```

### Network Isolation

```yaml
# Docker network isolation
networks:
  default:
    name: langchain-network
    # Only langchain containers on this network
```

### Firewall Recommendations

| Port | Direction | Source | Purpose |
|------|-----------|--------|---------|
| 8000 | Inbound | Load balancer | API traffic |
| 443 | Outbound | Internet | LLM API calls |

---

## Input Validation

### Pydantic Models

All API inputs are validated using Pydantic models:

```python
class ResearchAgentRequest(BaseModel):
    query: str  # Required, must be string
    session_id: str | None = None  # Optional

class ContentAgentRequest(BaseModel):
    topic: str
    platform: Literal["linkedin", "x", "blog"] = "linkedin"
    tone: str = "professional"
```

### Tool Input Validation

```python
@tool
@tool_error_handler
def create_support_ticket(
    title: str,
    description: str,
    priority: Literal["low", "medium", "high"] = "medium"
) -> str:
    """Tool with validated inputs."""
    # Pydantic validates all inputs before execution
    ...
```

### Prompt Injection Mitigation

- System prompts are separate from user inputs
- User inputs are treated as data, not instructions
- Tool descriptions limit what agents can do

---

## Security Best Practices

### Development Guidelines

1. **Never hardcode secrets**
   ```python
   # BAD
   api_key = "sk-1234567890"

   # GOOD
   api_key = os.getenv("OPENAI_API_KEY")
   if not api_key:
       raise ValueError("OPENAI_API_KEY not configured")
   ```

2. **Validate all inputs**
   ```python
   # All request models use Pydantic
   class UserRequest(BaseModel):
       message: str = Field(..., min_length=1, max_length=10000)
   ```

3. **Handle errors securely**
   ```python
   # Don't expose internal errors
   try:
       result = risky_operation()
   except Exception as e:
       logger.error("Operation failed", exc_info=True)
       return {"error": "Operation failed. Please try again."}
   ```

4. **Use type hints**
   ```python
   def process_input(data: str) -> dict[str, Any]:
       """Type hints catch errors at development time."""
       ...
   ```

### Deployment Guidelines

1. **Enable API key auth for external access**
2. **Use HTTPS in production**
3. **Implement rate limiting**
4. **Monitor for anomalies**
5. **Keep dependencies updated**

### Code Review Security Checklist

- [ ] No hardcoded secrets
- [ ] Input validation on all user inputs
- [ ] Error messages don't expose internals
- [ ] No `eval()` or `exec()` on user data
- [ ] Dependencies are up to date
- [ ] Logging doesn't include sensitive data

---

## Compliance Requirements

### Data Privacy (GDPR/CCPA)

| Requirement | Implementation |
|-------------|----------------|
| Data minimization | No persistent storage by default |
| Purpose limitation | Clear API purposes |
| Right to erasure | Session data clears on restart |
| Data portability | API-based access |

### Security Standards

| Standard | Relevance |
|----------|-----------|
| OWASP Top 10 | Input validation, auth, error handling |
| SOC 2 | Audit logging via LangSmith |
| ISO 27001 | Security controls documented |

### Audit Logging

LangSmith provides comprehensive audit logging:
- All API calls traced
- Tool invocations recorded
- Response content captured
- Latency and errors tracked

---

## Incident Response

### Security Incident Classification

| Level | Description | Response Time | Example |
|-------|-------------|---------------|---------|
| Critical | Active breach | Immediate | Leaked API keys |
| High | Vulnerability | 4 hours | Auth bypass |
| Medium | Suspicious activity | 24 hours | Unusual traffic |
| Low | Potential issue | 72 hours | Failed auth attempts |

### Incident Response Steps

1. **Detect**
   - Monitor alerts
   - Review LangSmith traces
   - Check error logs

2. **Contain**
   - Revoke compromised credentials
   - Block suspicious IPs
   - Disable affected endpoints

3. **Eradicate**
   - Rotate all secrets
   - Patch vulnerabilities
   - Update dependencies

4. **Recover**
   - Restore from known-good state
   - Enable enhanced monitoring
   - Verify functionality

5. **Document**
   - Create incident report
   - Update security procedures
   - Conduct post-mortem

### Key Rotation Procedure (Emergency)

```bash
# 1. Generate new keys
NEW_API_KEY=$(openssl rand -hex 32)

# 2. Update environment
echo "API_KEY=$NEW_API_KEY" >> .env.new
# Update other keys as needed

# 3. Deploy with new keys
docker compose down
mv .env .env.compromised
mv .env.new .env
docker compose up -d

# 4. Verify
curl -H "X-API-Key: $NEW_API_KEY" http://localhost:8000/health
```

---

## Security Checklist

### Pre-Deployment Checklist

- [ ] All secrets in environment variables
- [ ] `.env` excluded from git
- [ ] API key authentication enabled
- [ ] CORS configured for production
- [ ] Container runs as non-root
- [ ] Dependencies scanned for vulnerabilities
- [ ] Input validation on all endpoints
- [ ] Error handling doesn't expose internals
- [ ] Logging configured without PII
- [ ] Health check endpoints working

### Production Readiness Checklist

- [ ] HTTPS enabled (via load balancer or ngrok)
- [ ] Rate limiting implemented
- [ ] Monitoring and alerting configured
- [ ] Incident response plan documented
- [ ] Secret rotation schedule defined
- [ ] Backup and recovery tested
- [ ] Access logs enabled
- [ ] Security headers configured

### Regular Review Checklist

- [ ] Dependency updates applied (weekly)
- [ ] Access logs reviewed (weekly)
- [ ] Secret rotation completed (monthly)
- [ ] Penetration testing (quarterly)
- [ ] Security training updated (annually)

---

## Related Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md) - Security architecture details
- [DEPLOYMENT.md](./DEPLOYMENT.md) - Secure deployment procedures
- [OPERATIONS.md](./OPERATIONS.md) - Security operations
