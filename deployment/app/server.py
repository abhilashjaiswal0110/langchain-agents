"""LangChain Platform API Server.

This FastAPI application serves multiple LangChain chains and LangGraph agents
as REST API endpoints using LangServe with LangSmith tracing enabled.
"""

import json
import os
import re
import secrets
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi import Path as FastAPIPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from langserve import add_routes
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# Load environment variables from .env file (explicit path for reliability)
_ENV_FILE = Path(__file__).parent.parent / ".env"
_env_loaded = load_dotenv(_ENV_FILE, override=True)
print(f"[Startup] Loading .env from: {_ENV_FILE}")
print(f"[Startup] .env file exists: {_ENV_FILE.exists()}, loaded: {_env_loaded}")

# Import AIMessage for response extraction
from langchain_core.messages import AIMessage

from app.agents.evals.eval_middleware import submit_for_evaluation

# Response cache (opt-in via CACHE_ENABLED=true; default off for backward compat)
from app.cache.response_cache import CACHE_TTL_SECONDS, MAX_CACHE_SIZE, get_cache, is_cache_enabled
from app.governance.cost_estimator import get_cost_estimator

# ============================================================================
# Agent Response Helper
# ============================================================================


def extract_agent_response(result: dict) -> str:
    """Extract the AI response from agent result state.

    LangGraph agents return state with 'messages' array containing the conversation.
    This helper extracts the last AI message content as the response.

    Args:
        result: Agent invoke result (state dict or Pydantic model)

    Returns:
        The last AI message content, or empty string if not found
    """
    # Handle both dict and Pydantic model
    messages = []
    if hasattr(result, "messages"):
        messages = result.messages
    elif isinstance(result, dict):
        messages = result.get("messages", [])

    # Find last AI message
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content

    # Fallback to output key if present (for non-LangGraph agents)
    if isinstance(result, dict):
        return result.get("output", result.get("document", ""))

    return ""


def _serialize_sse(event: dict) -> str:
    """Serialize an SSE event dict to a JSON string.

    Handles common non-serializable types such as datetime objects,
    Pydantic models, and arbitrary objects with a ``__dict__`` attribute.

    Args:
        event: SSE event dict with ``type`` and ``data`` keys.

    Returns:
        JSON-encoded string representation of the event.
    """

    def _default(obj: object) -> object:
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    return json.dumps(event, default=_default)


# ============================================================================
# LangSmith Tracing Configuration
# ============================================================================


def _suppress_langsmith_noise() -> None:
    """Suppress repetitive LangSmith 403/connection error log noise.

    When LangSmith tracing is enabled but the API key is invalid/expired the
    background trace-upload thread emits a noisy "Failed to send compressed
    multipart ingest" warning every few seconds.  This helper configures the
    relevant loggers to WARNING level (once) so the first failure is still
    visible but repeated failures are silenced.
    """
    import logging

    # Suppress the background batch-ingest thread that spams on auth failure
    logging.getLogger("langsmith.client").setLevel(logging.ERROR)
    logging.getLogger("langsmith").setLevel(logging.ERROR)


def _verify_langsmith_key(api_key: str, endpoint: str, project: str) -> bool:
    """Verify a LangSmith API key is valid by probing the projects endpoint.

    Args:
        api_key: LangSmith API key to test.
        endpoint: LangSmith API endpoint URL.
        project: Project name to use if key is valid.

    Returns:
        True if the key is accepted (HTTP 200), False otherwise.
    """
    try:
        import urllib.request

        url = f"{endpoint.rstrip('/')}/projects?name={project}"
        req = urllib.request.Request(url, headers={"x-api-key": api_key})
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            return resp.status == 200
    except Exception:
        return False


def setup_langsmith_tracing() -> bool:
    """Configure LangSmith tracing if enabled.

    Probes the LangSmith API to verify the key is valid before enabling
    tracing.  If the key returns 403 tracing is disabled automatically and
    a clear diagnostic message is printed so the problem is obvious without
    flooding the log with repeated background-upload failures.

    Returns:
        True if tracing is enabled and the key verified, False otherwise.
    """
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    langsmith_api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    project = os.getenv("LANGCHAIN_PROJECT", "langchain-platform")

    if not tracing_enabled:
        return False

    if not langsmith_api_key:
        print("Warning: LANGCHAIN_TRACING_V2=true but LANGCHAIN_API_KEY not set")
        print("  → Tracing disabled. Get your API key from https://smith.langchain.com")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

    # Verify the key before enabling — avoids endless 403 background noise
    print(f"[LangSmith] Verifying API key for project '{project}'...")
    key_valid = _verify_langsmith_key(langsmith_api_key, endpoint, project)

    if not key_valid:
        print("[LangSmith] ⚠  API key verification FAILED (403 Forbidden).")
        print("  → The key may be expired or revoked.")
        print(f"  → Visit https://smith.langchain.com to generate a new key.")
        print(f"  → Set LANGCHAIN_API_KEY in deployment/.env and restart.")
        print("  → Tracing is now DISABLED to prevent log flooding.")
        # Disable tracing in-process so the background uploader never starts
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        _suppress_langsmith_noise()
        return False

    # Key is valid — activate tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = project
    os.environ.setdefault("LANGCHAIN_ENDPOINT", endpoint)
    print(f"[LangSmith] ✓  Tracing enabled → project: '{project}' @ {endpoint}")
    return True


# Initialize tracing early
tracing_enabled = setup_langsmith_tracing()

# ============================================================================
# API Key Security (for ngrok/external exposure)
# ============================================================================

API_KEY_ENABLED = os.getenv("API_KEY_ENABLED", "true").lower() == "true"
API_KEY = os.getenv("API_KEY", "")
API_KEY_HEADER = "X-API-Key"

# Log API key configuration at startup for debugging
print(f"[Security] API_KEY_ENABLED={API_KEY_ENABLED} (env: '{os.getenv('API_KEY_ENABLED', 'not set')}')")
if API_KEY_ENABLED:
    print("[Security] API key authentication is ENABLED - protected endpoints require X-API-Key header")
else:
    print("[Security] API key authentication is DISABLED - all endpoints are open")

# Paths that don't require authentication
PUBLIC_PATHS = {"/", "/docs", "/redoc", "/openapi.json", "/health", "/ready", "/chat", "/chatui"}

# API paths that use LangServe (no API key required for demo purposes)
LANGSERVE_PREFIXES = ("/api/langserve/",)

# API paths accessible from internal web UI (no API key required)
# These are secured by same-origin policy since they're accessed from /chat
UI_API_PREFIXES = (
    "/api/conversation",
    "/api/deepagent",
    "/api/sales-agent",
    "/api/recruitment-agent",
    "/api/enterprise",
)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware to validate API key for protected endpoints."""

    async def dispatch(self, request: Request, call_next):
        import sys

        # Normalize path (remove trailing slash for comparison)
        path = request.url.path.rstrip("/") or "/"

        # Debug: Log every request through middleware
        sys.stdout.write(f"[Middleware] Processing: {path}, API_KEY_ENABLED={API_KEY_ENABLED}\n")
        sys.stdout.flush()

        # Skip if API key auth is disabled
        if not API_KEY_ENABLED:
            sys.stdout.write(f"[Middleware] Auth disabled, allowing: {path}\n")
            sys.stdout.flush()
            return await call_next(request)

        # Skip public paths (check both with and without trailing slash)
        if path in PUBLIC_PATHS or request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # Skip static files
        if path.startswith("/static"):
            return await call_next(request)

        # Skip UI API endpoints (accessed from internal web UI)
        if path.startswith(UI_API_PREFIXES):
            return await call_next(request)

        # Skip LangServe API endpoints (demo/testing)
        if path.startswith(LANGSERVE_PREFIXES):
            return await call_next(request)

        # Validate API key (use constant-time comparison to prevent timing attacks)
        api_key = request.headers.get(API_KEY_HEADER)
        if not api_key or not API_KEY or not secrets.compare_digest(api_key, API_KEY):
            # Log the rejection for debugging
            print(f"[Security] Rejected request to {path} - API key missing/invalid")
            return HTMLResponse(
                content='{"detail": "Invalid or missing API key"}',
                status_code=401,
                media_type="application/json",
            )

        return await call_next(request)


# ============================================================================
# Chain Loading
# ============================================================================

chains_loaded = False
langgraph_loaded = False
doc_rag_loaded = False
it_support_loaded = False
enterprise_agents_loaded = False
deep_agent_loaded = False
chat_chain = None
rag_chain = None
agent_executor = None
langgraph_agent = None
doc_rag_chain = None
conversation_manager = None

# Enterprise Agents (new)
research_agent = None
content_agent = None
data_analyst_agent = None
document_agent = None
multilingual_rag_agent = None
hitl_support_agent = None
code_assistant_agent = None
document_intelligence_agent = None

# Deep Agent
it_operations_deep_agent = None
sales_intelligence_deep_agent = None
recruitment_deep_agent = None


def _is_azure_openai_configured() -> bool:
    """Check if Azure OpenAI is fully configured.

    Returns:
        True if all required Azure OpenAI env vars are set.
    """
    return all(
        [
            os.getenv("AZURE_OPENAI_API_KEY"),
            os.getenv("AZURE_OPENAI_ENDPOINT"),
            os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        ]
    )


def _is_any_llm_configured() -> bool:
    """Check if any LLM provider is configured.

    Returns:
        True if Azure OpenAI, OpenAI (enabled), or Anthropic is configured.
    """
    has_azure = _is_azure_openai_configured()
    has_openai = bool(os.getenv("OPENAI_API_KEY")) and os.getenv("OPENAI_ENABLED", "false").lower() == "true"
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    return has_azure or has_openai or has_anthropic


def load_chains() -> bool:
    """Load LangChain chains if any LLM provider is available.

    Returns:
        True if chains loaded successfully, False otherwise.
    """
    global chains_loaded, chat_chain, rag_chain, agent_executor

    if not _is_any_llm_configured():
        return False

    try:
        from app.chains.agent import agent_executor as _agent_executor
        from app.chains.chat import chat_chain as _chat_chain
        from app.chains.rag import rag_chain as _rag_chain

        chat_chain = _chat_chain
        rag_chain = _rag_chain
        agent_executor = _agent_executor
        chains_loaded = True
        return True
    except Exception as e:
        print(f"Failed to load LangChain chains: {e}")
        return False


def load_langgraph_agent() -> bool:
    """Load LangGraph agent if any LLM provider is available.

    Returns:
        True if LangGraph agent loaded successfully, False otherwise.
    """
    global langgraph_loaded, langgraph_agent

    if not _is_any_llm_configured():
        return False

    try:
        from app.chains.langgraph_agent import LangGraphAgentRunnable

        langgraph_agent = LangGraphAgentRunnable(model_provider="auto")
        if langgraph_agent.agent is not None:
            langgraph_loaded = True
            return True
        return False
    except Exception as e:
        print(f"Failed to load LangGraph agent: {e}")
        return False


def load_doc_rag() -> bool:
    """Load Document RAG chain if any LLM provider is available.

    Returns:
        True if Document RAG chain loaded successfully, False otherwise.
    """
    global doc_rag_loaded, doc_rag_chain

    if not _is_any_llm_configured():
        return False

    try:
        from app.chains.doc_rag import doc_rag_chain as _doc_rag_chain

        doc_rag_chain = _doc_rag_chain
        doc_rag_loaded = True
        return True
    except Exception as e:
        print(f"Failed to load Document RAG chain: {e}")
        return False


def load_it_support_agents() -> bool:
    """Load IT Support agents and conversation manager.

    Returns:
        True if IT Support agents loaded successfully, False otherwise.
    """
    global it_support_loaded, conversation_manager

    if not _is_any_llm_configured():
        return False

    try:
        from app.agents.conversation_manager import ConversationManager

        conversation_manager = ConversationManager()
        it_support_loaded = True
        return True
    except Exception as e:
        print(f"Failed to load IT Support agents: {e}")
        return False


def load_enterprise_agents() -> dict[str, bool]:
    """Load enterprise IT agents.

    Returns:
        Dictionary with load status for each agent.
    """
    global enterprise_agents_loaded
    global research_agent, content_agent, data_analyst_agent
    global document_agent, multilingual_rag_agent, hitl_support_agent
    global code_assistant_agent, document_intelligence_agent

    if not _is_any_llm_configured():
        return {"loaded": False, "reason": "No API keys configured"}

    status = {}

    # Research Agent
    try:
        from app.agents.research import ResearchAgent

        research_agent = ResearchAgent()
        status["research"] = True
    except Exception as e:
        print(f"Failed to load Research Agent: {e}")
        status["research"] = False

    # Content Agent (with auto_approve=True for API usage - skip HITL review)
    try:
        from app.agents.content import ContentAgent

        content_agent = ContentAgent(auto_approve=True)
        status["content"] = True
    except Exception as e:
        print(f"Failed to load Content Agent: {e}")
        status["content"] = False

    # Data Analyst Agent
    try:
        from app.agents.data_analyst import DataAnalystAgent

        data_analyst_agent = DataAnalystAgent()
        status["data_analyst"] = True
    except Exception as e:
        print(f"Failed to load Data Analyst Agent: {e}")
        status["data_analyst"] = False

    # Document Agent
    try:
        from app.agents.documents import DocumentAgent

        document_agent = DocumentAgent()
        status["document"] = True
    except Exception as e:
        print(f"Failed to load Document Agent: {e}")
        status["document"] = False

    # Multilingual RAG Agent
    try:
        from app.agents.rag import MultilingualRAGAgent

        multilingual_rag_agent = MultilingualRAGAgent()
        status["multilingual_rag"] = True
    except Exception as e:
        print(f"Failed to load Multilingual RAG Agent: {e}")
        status["multilingual_rag"] = False

    # HITL Support Agent
    try:
        from app.agents.it_support import HITLSupportAgent

        hitl_support_agent = HITLSupportAgent()
        status["hitl_support"] = True
    except Exception as e:
        print(f"Failed to load HITL Support Agent: {e}")
        status["hitl_support"] = False

    # Code Assistant Agent
    try:
        from app.agents.code_assistant import CodeAssistantAgent

        code_assistant_agent = CodeAssistantAgent()
        status["code_assistant"] = True
    except Exception as e:
        print(f"Failed to load Code Assistant Agent: {e}")
        status["code_assistant"] = False

    # Document Intelligence Agent
    try:
        from app.agents.document_intelligence import DocumentIntelligenceAgent

        document_intelligence_agent = DocumentIntelligenceAgent()
        status["document_intelligence"] = True
    except Exception as e:
        print(f"Failed to load Document Intelligence Agent: {e}")
        status["document_intelligence"] = False

    enterprise_agents_loaded = any(status.values())
    status["loaded"] = enterprise_agents_loaded
    return status


def load_deep_agent() -> bool:
    """Load the IT Operations Deep Agent.

    Returns:
        True if Deep Agent loaded successfully, False otherwise.
    """
    global deep_agent_loaded, it_operations_deep_agent, sales_intelligence_deep_agent, recruitment_deep_agent

    if not _is_any_llm_configured():
        print(
            "[DEBUG] Deep Agent: No LLM provider configured (Azure OpenAI, OpenAI with OPENAI_ENABLED=true, or Anthropic)"
        )
        return False

    try:
        from app.deepagents import create_it_operations_agent
        from app.deepagents.recruitment_agent import create_recruitment_agent
        from app.deepagents.sales_intelligence_agent import create_sales_intelligence_agent

        # Get model configuration from environment
        provider = os.getenv("DEEP_AGENT_PROVIDER", "auto")
        model_name = os.getenv("DEEP_AGENT_MODEL") or None

        it_operations_deep_agent = create_it_operations_agent(
            model_provider=provider,
            model_name=model_name,
            storage_path="./data/deepagent_context",
        )

        sales_intelligence_deep_agent = create_sales_intelligence_agent(
            model_provider=provider,
            model_name=model_name,
            storage_path="./data/deepagent_context",
        )

        recruitment_deep_agent = create_recruitment_agent(
            model_provider=provider,
            model_name=model_name,
            storage_path="./data/recruitment_context",
        )

        deep_agent_loaded = True

        # Log model info
        model_info = model_name or "default"
        print(f"[DEBUG] Deep Agents using provider={provider}, model={model_info}")
        print("[DEBUG] Loaded: IT Operations, Sales Intelligence, Recruitment")

        return True
    except Exception as e:
        import traceback

        print(f"[ERROR] Failed to load Deep Agent: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# Application Lifespan
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup and shutdown."""
    # Startup
    print("=" * 60)
    print("LangChain Platform Starting...")
    print("=" * 60)

    # Report tracing status
    if tracing_enabled:
        print("[OK] LangSmith tracing enabled")
        print(f"     Project: {os.getenv('LANGCHAIN_PROJECT')}")
        print(f"     Endpoint: {os.getenv('LANGCHAIN_ENDPOINT')}")
    else:
        print("[--] LangSmith tracing disabled")

    # Load chains and setup LangServe routes
    if load_chains():
        print("[OK] LangChain chains loaded (OpenAI)")
        setup_langchain_routes()
        print("[OK] LangServe routes registered")
    else:
        print("[--] LangChain chains not loaded (OPENAI_API_KEY not set)")

    # Load LangGraph agent
    if load_langgraph_agent():
        provider = "Anthropic" if os.getenv("ANTHROPIC_API_KEY") else "OpenAI"
        print(f"[OK] LangGraph agent loaded ({provider})")
    else:
        print("[--] LangGraph agent not loaded (no API keys set)")

    # Load Document RAG chain
    if load_doc_rag():
        print("[OK] Document RAG chain loaded (OpenAI)")
    else:
        print("[--] Document RAG chain not loaded (OPENAI_API_KEY not set)")

    # Load IT Support agents
    if load_it_support_agents():
        agents = list(conversation_manager.get_available_agents().keys())
        print(f"[OK] IT Support agents loaded: {', '.join(agents)}")
    else:
        print("[--] IT Support agents not loaded (no API keys set)")

    # Load Enterprise Agents
    enterprise_status = load_enterprise_agents()
    if enterprise_status.get("loaded"):
        loaded_agents = [k for k, v in enterprise_status.items() if v is True and k != "loaded"]
        print(f"[OK] Enterprise agents loaded: {', '.join(loaded_agents)}")
    else:
        print("[--] Enterprise agents not loaded (no API keys set)")

    # Load Deep Agent
    if load_deep_agent():
        print("[OK] IT Operations Deep Agent loaded")
    else:
        print("[--] Deep Agent not loaded (no API keys set)")

    # Initialise response cache singleton (no-op when CACHE_ENABLED=false)
    _agent_cache = get_cache()
    if is_cache_enabled():
        print(f"[OK] Response cache enabled (TTL={CACHE_TTL_SECONDS}s, max_size={MAX_CACHE_SIZE})")
    else:
        print("[--] Response cache disabled (set CACHE_ENABLED=true to enable)")

    print("=" * 60)
    print("Platform ready!")
    print("  - Chat UI:  http://localhost:8000/chat")
    print("  - API Docs: http://localhost:8000/docs")
    print("  - Health:   http://localhost:8000/health")
    print("=" * 60)

    yield

    # Shutdown
    print("Shutting down LangChain Platform...")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="LangChain Platform API",
    version="1.0.0",
    description="""
## LangChain Platform with LangGraph Integration

A production-ready API platform serving LangChain chains and LangGraph agents
with full LangSmith tracing support.

### Available Endpoints

#### LangChain Chains (requires OPENAI_API_KEY)
- **Chat Chain** (`/chat`): Simple conversational AI
- **RAG Chain** (`/rag`): Retrieval-Augmented Generation
- **Agent** (`/agent`): AI agent with tools

#### LangGraph Agents (requires OPENAI_API_KEY or ANTHROPIC_API_KEY)
- **LangGraph Agent** (`/langgraph`): Stateful agent with tool calling

### Tracing & Observability

Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` to enable
LangSmith tracing for full observability.

### Documentation

- **API Docs**: `/docs` (Swagger UI)
- **ReDoc**: `/redoc`
- **OpenAPI**: `/openapi.json`
    """,
    lifespan=lifespan,
)

# Configure CORS (restrict origins in production, never use "*")
_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:8000,http://localhost:3000")
_allowed_origins = [origin.strip() for origin in _cors_origins.split(",") if origin.strip()]

# SECURITY: Reject wildcard origins in production
if any("*" in origin for origin in _allowed_origins):
    print("[Security] WARNING: Wildcard CORS origins detected - not recommended for production")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept", API_KEY_HEADER],
)

# Add API key authentication middleware
app.add_middleware(APIKeyMiddleware)

# Include integration routes (Teams, Slack webhooks)
try:
    from app.integrations.routes import router as integrations_router

    app.include_router(integrations_router)
    print("[OK] Integration routes loaded (Teams, Slack)")
except ImportError as e:
    print(f"[--] Integration routes not loaded: {e}")

# Include Software Development Deep Agent routes
try:
    from app.deepagents.software_dev.routes import router as software_dev_router

    app.include_router(software_dev_router)
    print("[OK] Software Development Deep Agent routes loaded")
except ImportError as e:
    print(f"[--] Software Development Deep Agent routes not loaded: {e}")

# Include Domain Agent routes (8 specialised domain agents)
try:
    from app.agents.domains.routes import router as domain_router

    app.include_router(domain_router)
    print("[OK] Domain Agent routes loaded (marcom, hr, lnd, presales, datacenter, cloud, cybersecurity, data_ai)")
except ImportError as e:
    print(f"[--] Domain Agent routes not loaded: {e}")

# Include Analytics routes
try:
    from app.analytics.metrics_api import router as analytics_router

    app.include_router(analytics_router)
    print("[OK] Analytics routes loaded")
except ImportError as e:
    print(f"[--] Analytics routes not loaded: {e}")

# Mount Prometheus /metrics endpoint
try:
    from app.monitoring.prometheus import setup_metrics

    setup_metrics(app)
    print("[OK] Prometheus /metrics endpoint mounted")
except Exception as e:  # noqa: BLE001
    print(f"[--] Prometheus metrics not mounted: {e}")


# ============================================================================
# Response Models
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    version: str
    chains_loaded: bool
    langgraph_loaded: bool
    doc_rag_loaded: bool
    it_support_loaded: bool
    enterprise_agents_loaded: bool
    deep_agent_loaded: bool
    tracing_enabled: bool
    langsmith_project: str | None


class LangGraphRequest(BaseModel):
    """LangGraph agent request model."""

    input: str


class LangGraphResponse(BaseModel):
    """LangGraph agent response model."""

    output: str


class DocRagQueryRequest(BaseModel):
    """Document RAG query request model."""

    question: str
    k: int = 4


class DocRagQueryResponse(BaseModel):
    """Document RAG query response model."""

    status: str
    answer: str | None = None
    sources: list[dict] | None = None
    num_sources: int | None = None
    error: str | None = None


class DocRagUploadResponse(BaseModel):
    """Document RAG upload response model."""

    status: str
    file_name: str | None = None
    original_filename: str | None = None
    chunks_created: int | None = None
    total_documents: int | None = None
    error: str | None = None


class DocRagInfoResponse(BaseModel):
    """Document RAG info response model."""

    total_documents: int
    total_chunks: int
    documents: dict
    vector_store_initialized: bool


# Conversation API Models
class ConversationStartRequest(BaseModel):
    """Request to start a new conversation."""

    agent_type: Literal["it_helpdesk", "servicenow", "document_intelligence", "employee_experience"]
    user_id: str | None = None
    metadata: dict | None = None


class ConversationStartResponse(BaseModel):
    """Response from starting a conversation."""

    session_id: str | None = None
    agent_type: str | None = None
    welcome_message: str | None = None
    available_commands: list[str] | None = None
    error: str | None = None


class ConversationChatRequest(BaseModel):
    """Request to send a message in a conversation."""

    session_id: str
    message: str


class ConversationChatResponse(BaseModel):
    """Response from conversation chat."""

    session_id: str | None = None
    response: str | None = None
    agent_type: str | None = None
    tool_calls: list | None = None
    is_command: bool = False
    error: str | None = None


# Integration API Models (for external platforms)
class WebhookPayload(BaseModel):
    """Webhook payload for external integrations."""

    event_type: str
    session_id: str | None = None
    agent_type: str | None = None
    message: str | None = None
    user_id: str | None = None
    metadata: dict | None = None


class IntegrationResponse(BaseModel):
    """Standard response for integrations."""

    success: bool
    message: str | None = None
    data: dict | None = None
    session_id: str | None = None


# 3rd Party Platform-specific Webhook Models
class CopilotStudioRequest(BaseModel):
    """Microsoft Copilot Studio webhook request."""

    query: str
    agent_type: str = "research"
    session_id: str | None = None
    user_id: str | None = None
    conversation_id: str | None = None
    channel: str = "copilot-studio"
    metadata: dict | None = None


class AzureAIRequest(BaseModel):
    """Azure AI Agent webhook request."""

    query: str
    agent_type: str = "research"
    session_id: str | None = None
    deployment_id: str | None = None
    resource_group: str | None = None
    subscription_id: str | None = None
    metadata: dict | None = None


class AWSLexRequest(BaseModel):
    """AWS Lex webhook request."""

    query: str
    agent_type: str = "research"
    session_id: str | None = None
    bot_id: str | None = None
    bot_alias_id: str | None = None
    locale_id: str = "en_US"
    session_attributes: dict | None = None
    request_attributes: dict | None = None


class ThirdPartyResponse(BaseModel):
    """Standardized response for 3rd party integrations."""

    success: bool
    response: str | None = None
    session_id: str | None = None
    agent_type: str | None = None
    source: str | None = None
    metadata: dict | None = None
    error: str | None = None


# ============================================================================
# Enterprise Agent Models
# ============================================================================


class EnterpriseAgentRequest(BaseModel):
    """Base request for enterprise agents."""

    message: str
    session_id: str | None = None


class EnterpriseAgentResponse(BaseModel):
    """Base response from enterprise agents."""

    success: bool
    response: str | None = None
    session_id: str | None = None
    agent_type: str | None = None
    tool_calls: list | None = None
    error: str | None = None
    cached: bool = False


class ResearchAgentRequest(BaseModel):
    """Research agent request."""

    query: str
    session_id: str | None = None


class ContentAgentRequest(BaseModel):
    """Content generation agent request."""

    topic: str
    platform: Literal["linkedin", "x", "blog"] = "linkedin"
    tone: str = "professional"
    audience: str = "general"
    session_id: str | None = None


class DataAnalystRequest(BaseModel):
    """Data analyst agent request."""

    message: str
    session_id: str | None = None


class DocumentAgentRequest(BaseModel):
    """Document generation agent request."""

    doc_type: Literal["sop", "wli", "policy"]
    title: str
    description: str
    sections: list[str] | None = None
    session_id: str | None = None


class RAGAgentRequest(BaseModel):
    """Multilingual RAG agent request."""

    query: str
    language: str | None = None
    session_id: str | None = None


class CodeAssistantRequest(BaseModel):
    """Code assistant agent request."""

    code: str
    language: str = "python"
    action: Literal["analyze", "modernize"] = "analyze"
    include_security: bool = True
    session_id: str | None = None


class HITLSupportRequest(BaseModel):
    """Human-in-the-loop support agent request."""

    message: str
    session_id: str | None = None
    user_id: str | None = None


class HITLApprovalRequest(BaseModel):
    """HITL approval request."""

    session_id: str
    action_id: str
    approved: bool
    approved_by: str | None = None


class DocumentIntelligenceRequest(BaseModel):
    """Document Intelligence agent request."""

    message: str
    session_id: str | None = None
    target_language: str | None = None


class DocumentIntelligenceUploadResponse(BaseModel):
    """Document Intelligence upload response."""

    success: bool
    document_id: str | None = None
    filename: str | None = None
    file_type: str | None = None
    chunks_created: int | None = None
    detected_language: str | None = None
    message: str | None = None
    error: str | None = None


# ============================================================================
# Deep Agent Models
# ============================================================================


class DeepAgentStartRequest(BaseModel):
    """Request to start a Deep Agent session."""

    user_id: str | None = None
    metadata: dict | None = None


class DeepAgentStartResponse(BaseModel):
    """Response from starting a Deep Agent session."""

    success: bool
    session_id: str | None = None
    message: str | None = None
    error: str | None = None


class DeepAgentChatRequest(BaseModel):
    """Request to chat with Deep Agent."""

    message: str
    session_id: str | None = None
    user_id: str | None = None


class DeepAgentChatResponse(BaseModel):
    """Response from Deep Agent chat."""

    success: bool
    response: str | None = None
    session_id: str | None = None
    todos: list[dict] | None = None
    files: list[str] | None = None
    tool_calls: list | None = None
    iteration_count: int | None = None
    error: str | None = None


class DeepAgentContextResponse(BaseModel):
    """Response with Deep Agent session context."""

    success: bool
    session_id: str
    todos: list[dict] | None = None
    files: list[str] | None = None
    metadata: dict | None = None
    error: str | None = None


class DeepAgentUploadResponse(BaseModel):
    """Response from Deep Agent document upload."""

    success: bool
    document_id: str | None = None
    filename: str | None = None
    file_type: str | None = None
    chunks_created: int | None = None
    detected_language: str | None = None
    session_id: str | None = None
    message: str | None = None
    error: str | None = None


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/")
async def root() -> RedirectResponse:
    """Redirect root to API documentation."""
    return RedirectResponse("/docs")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint with detailed status."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        chains_loaded=chains_loaded,
        langgraph_loaded=langgraph_loaded,
        doc_rag_loaded=doc_rag_loaded,
        it_support_loaded=it_support_loaded,
        enterprise_agents_loaded=enterprise_agents_loaded,
        deep_agent_loaded=deep_agent_loaded,
        tracing_enabled=tracing_enabled,
        langsmith_project=os.getenv("LANGCHAIN_PROJECT") if tracing_enabled else None,
    )


@app.get("/ready")
async def readiness_check() -> dict:
    """Readiness check for Kubernetes."""
    if not (chains_loaded or langgraph_loaded):
        raise HTTPException(
            status_code=503,
            detail="Service not ready: no chains or agents loaded",
        )
    return {"status": "ready"}


@app.post("/langgraph/invoke", response_model=LangGraphResponse)
async def langgraph_invoke(request: LangGraphRequest) -> LangGraphResponse:
    """Invoke the LangGraph agent.

    Args:
        request: The input request with user message.

    Returns:
        The agent's response.

    Raises:
        HTTPException: If LangGraph agent is not available.
    """
    if not langgraph_loaded or langgraph_agent is None:
        raise HTTPException(
            status_code=503,
            detail="LangGraph agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.",
        )

    result = await langgraph_agent.ainvoke({"input": request.input})
    return LangGraphResponse(output=result["output"])


# ============================================================================
# Document RAG Endpoints
# ============================================================================

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".doc"}


@app.post("/doc-rag/upload", response_model=DocRagUploadResponse)
async def doc_rag_upload(file: UploadFile = File(...)) -> DocRagUploadResponse:
    """Upload a document for RAG processing.

    Supports PDF, Word (.docx), and plain text (.txt) files.

    Args:
        file: The document file to upload.

    Returns:
        Upload status and document information.

    Raises:
        HTTPException: If Document RAG is not available or file type unsupported.
    """
    if not doc_rag_loaded or doc_rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="Document RAG not available. Set OPENAI_API_KEY.",
        )

    # Validate file extension
    filename = file.filename or "unknown"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read file content
    content = await file.read()

    # Process document
    result = doc_rag_chain.load_from_bytes(content, filename)

    return DocRagUploadResponse(**result)


@app.post("/doc-rag/query", response_model=DocRagQueryResponse)
async def doc_rag_query(request: DocRagQueryRequest) -> DocRagQueryResponse:
    """Query the uploaded documents.

    Args:
        request: The query request with question and optional k value.

    Returns:
        Answer and source information.

    Raises:
        HTTPException: If Document RAG is not available.
    """
    if not doc_rag_loaded or doc_rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="Document RAG not available. Set OPENAI_API_KEY.",
        )

    result = doc_rag_chain.query(request.question, k=request.k)

    return DocRagQueryResponse(**result)


@app.get("/doc-rag/info", response_model=DocRagInfoResponse)
async def doc_rag_info() -> DocRagInfoResponse:
    """Get information about loaded documents.

    Returns:
        Document statistics and metadata.

    Raises:
        HTTPException: If Document RAG is not available.
    """
    if not doc_rag_loaded or doc_rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="Document RAG not available. Set OPENAI_API_KEY.",
        )

    info = doc_rag_chain.get_document_info()
    return DocRagInfoResponse(**info)


@app.delete("/doc-rag/clear")
async def doc_rag_clear() -> dict:
    """Clear all loaded documents from memory.

    Returns:
        Status message.

    Raises:
        HTTPException: If Document RAG is not available.
    """
    if not doc_rag_loaded or doc_rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="Document RAG not available. Set OPENAI_API_KEY.",
        )

    result = doc_rag_chain.clear_documents()
    return result


# ============================================================================
# Conversation API Endpoints (IT Support Agents)
# ============================================================================

_TENANT_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


def _validate_tenant_id(tenant_id: str) -> str:
    """Validate and return the tenant ID.

    Args:
        tenant_id: Raw tenant identifier from the request header.

    Returns:
        The validated tenant identifier, unchanged.

    Raises:
        HTTPException: 400 if the value contains disallowed characters or
            exceeds the maximum length.
    """
    if not _TENANT_ID_RE.fullmatch(tenant_id):
        raise HTTPException(
            status_code=400,
            detail="X-Tenant-ID must be 1-64 alphanumeric, dash, or underscore characters.",
        )
    return tenant_id


@app.post("/api/conversation/start", response_model=ConversationStartResponse)
async def conversation_start(
    request: ConversationStartRequest,
    x_tenant_id: str = Header(default="default"),
) -> ConversationStartResponse:
    """Start a new conversation with an IT Support agent.

    The optional `X-Tenant-ID` request header scopes the session to a specific
    tenant. When the header is absent the session is created in the `"default"`
    tenant, preserving full backward compatibility.

    Args:
        request: The conversation start request with agent type.
        x_tenant_id: Tenant identifier extracted from the `X-Tenant-ID` header.

    Returns:
        Session ID, welcome message, and available commands.
    """
    tenant_id = _validate_tenant_id(x_tenant_id)

    if not it_support_loaded or conversation_manager is None:
        raise HTTPException(
            status_code=503,
            detail="IT Support agents not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.",
        )

    result = conversation_manager.start_conversation(
        agent_type=request.agent_type,
        user_id=request.user_id,
        metadata=request.metadata,
        tenant_id=tenant_id,
    )

    return ConversationStartResponse(**result)


@app.post("/api/conversation/chat", response_model=ConversationChatResponse)
async def conversation_chat(
    request: ConversationChatRequest,
    x_tenant_id: str = Header(default="default"),
) -> ConversationChatResponse:
    """Send a message in an existing conversation.

    The optional `X-Tenant-ID` request header must match the tenant used when
    the session was created. Omitting the header uses the `"default"` tenant.

    Args:
        request: The chat request with session ID and message.
        x_tenant_id: Tenant identifier extracted from the `X-Tenant-ID` header.

    Returns:
        Agent's response and metadata.
    """
    tenant_id = _validate_tenant_id(x_tenant_id)

    if not it_support_loaded or conversation_manager is None:
        raise HTTPException(
            status_code=503,
            detail="IT Support agents not available.",
        )

    result = await conversation_manager.achat(
        session_id=request.session_id,
        message=request.message,
        tenant_id=tenant_id,
    )

    return ConversationChatResponse(**result)


@app.get("/api/conversation/{session_id}")
async def conversation_info(
    session_id: str,
    x_tenant_id: str = Header(default="default"),
) -> dict:
    """Get information about a conversation session.

    Args:
        session_id: The session ID to query.
        x_tenant_id: Tenant identifier extracted from the `X-Tenant-ID` header.

    Returns:
        Session information.
    """
    tenant_id = _validate_tenant_id(x_tenant_id)

    if not it_support_loaded or conversation_manager is None:
        raise HTTPException(status_code=503, detail="IT Support agents not available.")

    info = conversation_manager.get_session_info(session_id, tenant_id=tenant_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found.")

    return info


@app.delete("/api/conversation/{session_id}")
async def conversation_end(
    session_id: str,
    x_tenant_id: str = Header(default="default"),
) -> dict:
    """End a conversation session.

    Args:
        session_id: The session ID to end.
        x_tenant_id: Tenant identifier extracted from the `X-Tenant-ID` header.

    Returns:
        Session summary.
    """
    tenant_id = _validate_tenant_id(x_tenant_id)

    if not it_support_loaded or conversation_manager is None:
        raise HTTPException(status_code=503, detail="IT Support agents not available.")

    return conversation_manager.end_conversation(session_id, tenant_id=tenant_id)


@app.get("/api/conversation/{session_id}/export")
async def export_conversation(
    session_id: str = FastAPIPath(..., pattern=r"^[0-9a-f\-]{36}$"),
    export_format: str = "json",
    x_tenant_id: str = Header(default="default"),
) -> Response:
    """Export a conversation session as JSON, plain text, or PDF.

    Args:
        session_id: The session ID to export (UUID format).
        export_format: Output format — one of ``json``, ``text``, or ``pdf``.
        x_tenant_id: Tenant identifier extracted from the `X-Tenant-ID` header.

    Returns:
        The exported conversation in the requested format.

    Raises:
        HTTPException: 400 if the tenant ID is invalid, 404 if the session is
            not found, 422 if the format is not supported, or 503 if IT Support
            agents are unavailable.
    """
    tenant_id = _validate_tenant_id(x_tenant_id)

    if not it_support_loaded or conversation_manager is None:
        raise HTTPException(status_code=503, detail="IT Support agents not available.")

    session = conversation_manager.session_store.get_session(session_id, tenant_id=tenant_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    supported_formats = {"json", "text", "pdf"}
    if export_format not in supported_formats:
        raise HTTPException(
            status_code=422,
            detail="Unsupported format. Choose one of: json, text, pdf",
        )

    from app.agents.export import ConversationExporter

    exporter = ConversationExporter()

    if export_format == "text":
        return PlainTextResponse(exporter.to_text(session))
    elif export_format == "pdf":
        try:
            content = exporter.to_pdf(session)
        except ImportError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        safe_id = re.sub(r"[^\w\-]", "_", session_id)
        return Response(
            content=content,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="conversation-{safe_id}.pdf"'},
        )
    # Default: JSON
    return JSONResponse(json.loads(exporter.to_json(session)))


@app.post("/api/conversation/{session_id}/handoff", tags=["IT Support"])
async def handoff_conversation(
    session_id: str = FastAPIPath(..., pattern=r"^[0-9a-f\-]{36}$"),
    body: dict = None,
    x_tenant_id: str = Header(default="default"),
) -> dict:
    """Transfer a conversation to a different agent, preserving context.

    Args:
        session_id: The session to transfer.
        body: JSON body with ``to_agent``, ``reason``, and optional
              ``conversation_summary`` / ``key_entities``.
        x_tenant_id: Tenant scope.

    Returns:
        Handoff result with ``success``, ``new_agent``, and optional ``error``.
    """
    if not it_support_loaded or conversation_manager is None:
        raise HTTPException(status_code=503, detail="IT Support agents not available")
    if body is None:
        body = {}

    from app.agents.handoff.handoff_manager import HandoffManager
    from app.agents.handoff.handoff_protocol import HandoffRequest

    try:
        session = conversation_manager.session_store.get_session(session_id, tenant_id=x_tenant_id)
    except Exception:
        session = None
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    req = HandoffRequest(
        from_agent=session.metadata.agent_type,
        to_agent=body.get("to_agent", ""),
        reason=body.get("reason", "User requested transfer"),
        session_id=session_id,
        conversation_summary=body.get("conversation_summary", ""),
        key_entities=body.get("key_entities", {}),
    )

    result = await HandoffManager().execute_handoff(req, conversation_manager)
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    return result.model_dump()


@app.get("/api/agents")
async def list_agents() -> dict:
    """Get list of available IT Support agents."""
    if not it_support_loaded or conversation_manager is None:
        return {"agents": {}, "status": "unavailable"}

    return {
        "agents": conversation_manager.get_available_agents(),
        "status": "available",
    }


# ============================================================================
# Master Orchestrator — unified single-entry-point endpoint
# ============================================================================

_orchestrator_instance = None


def _get_orchestrator():
    global _orchestrator_instance
    if _orchestrator_instance is None:
        from app.agents.supervisors.master_orchestrator import MasterOrchestrator

        _orchestrator_instance = MasterOrchestrator(
            conversation_manager=conversation_manager,
        )
    return _orchestrator_instance


@app.post("/api/orchestrate", tags=["Orchestration"])
async def orchestrate(body: dict) -> dict:
    """Route a message to the most appropriate agent automatically.

    Classifies the request and forwards it to one of:
    IT Support, Domain Agent, Deep Agent, or Research Agent.

    Args:
        body: JSON with ``message`` (required), ``session_id`` (optional),
              and ``user_context`` dict (optional).

    Returns:
        Dict with ``cluster``, ``agent_type``, ``session_id``, and ``response``.
    """
    message = body.get("message", "")
    if not message:
        raise HTTPException(status_code=422, detail="'message' field is required")

    result = await _get_orchestrator().route(
        message=message,
        session_id=body.get("session_id"),
        user_context=body.get("user_context", {}),
    )
    return result


# ============================================================================
# Integration Endpoints (for external platforms: Copilot Studio, Azure AI, etc.)
# ============================================================================


@app.post("/api/webhook/chat", response_model=IntegrationResponse)
async def webhook_chat(payload: WebhookPayload) -> IntegrationResponse:
    """Webhook endpoint for external platform integration.

    Supports: Microsoft Copilot Studio, Azure AI Agent, AWS AI, etc.

    Args:
        payload: The webhook payload with event type and message.

    Returns:
        Standardized response for integration.
    """
    if not it_support_loaded or conversation_manager is None:
        return IntegrationResponse(
            success=False,
            message="IT Support agents not available.",
        )

    try:
        # Handle different event types
        if payload.event_type == "conversation.start":
            agent_type = payload.agent_type or "it_helpdesk"
            result = conversation_manager.start_conversation(
                agent_type=agent_type,
                user_id=payload.user_id,
                metadata=payload.metadata,
            )
            return IntegrationResponse(
                success=True,
                message=result.get("welcome_message"),
                session_id=result.get("session_id"),
                data={"agent_type": agent_type},
            )

        elif payload.event_type == "conversation.message":
            if not payload.session_id or not payload.message:
                return IntegrationResponse(
                    success=False,
                    message="session_id and message are required.",
                )

            result = await conversation_manager.achat(
                session_id=payload.session_id,
                message=payload.message,
            )

            if "error" in result:
                return IntegrationResponse(
                    success=False,
                    message=result["error"],
                    session_id=payload.session_id,
                )

            return IntegrationResponse(
                success=True,
                message=result.get("response"),
                session_id=payload.session_id,
                data={"tool_calls": result.get("tool_calls", [])},
            )

        elif payload.event_type == "conversation.end":
            if not payload.session_id:
                return IntegrationResponse(
                    success=False,
                    message="session_id is required.",
                )

            result = conversation_manager.end_conversation(payload.session_id)
            return IntegrationResponse(
                success=True,
                message="Conversation ended.",
                session_id=payload.session_id,
                data=result,
            )

        else:
            return IntegrationResponse(
                success=False,
                message=f"Unknown event type: {payload.event_type}",
            )

    except Exception as e:
        return IntegrationResponse(
            success=False,
            message=str(e),
            session_id=payload.session_id,
        )


# ============================================================================
# 3rd Party Platform Webhook Endpoints
# ============================================================================


def _invoke_enterprise_agent(agent_type: str, query: str) -> tuple[bool, str | None]:
    """Helper to invoke an enterprise agent by type.

    Returns:
        Tuple of (success, response_or_error)
    """
    from langchain_core.messages import AIMessage

    agent_map = {
        "research": research_agent,
        "content": content_agent,
        "data-analyst": data_analyst_agent,
        "document": document_agent,
        "multilingual-rag": multilingual_rag_agent,
        "hitl-support": hitl_support_agent,
        "code-assistant": code_assistant_agent,
    }

    agent = agent_map.get(agent_type)
    if agent is None:
        return False, f"Agent '{agent_type}' not available or not loaded"

    try:
        # Use invoke (synchronous)
        result = agent.invoke(message=query)

        # Extract response - handle both dict and Pydantic model
        messages = []
        if hasattr(result, "messages"):
            messages = result.messages
        elif isinstance(result, dict):
            messages = result.get("messages", [])

        # Find last AI message
        response = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                response = msg.content
                break

        return True, response
    except Exception as e:
        return False, str(e)


@app.post("/api/webhooks/copilot-studio", response_model=ThirdPartyResponse)
async def copilot_studio_webhook(request: CopilotStudioRequest) -> ThirdPartyResponse:
    """Webhook endpoint for Microsoft Copilot Studio integration.

    Allows Copilot Studio to invoke enterprise agents via HTTP action.

    Example Copilot Studio configuration:
    - Action Type: HTTP Request
    - Method: POST
    - URL: https://your-server/api/webhooks/copilot-studio
    - Body: {"query": "user input", "agent_type": "research"}
    """
    if not enterprise_agents_loaded:
        return ThirdPartyResponse(
            success=False,
            error="Enterprise agents not loaded",
            source="copilot-studio",
        )

    session_id = request.session_id or f"copilot-{request.conversation_id or 'default'}"
    success, response = _invoke_enterprise_agent(request.agent_type, request.query)

    return ThirdPartyResponse(
        success=success,
        response=response if success else None,
        error=response if not success else None,
        session_id=session_id,
        agent_type=request.agent_type,
        source="copilot-studio",
        metadata={
            "channel": request.channel,
            "user_id": request.user_id,
            "conversation_id": request.conversation_id,
        },
    )


@app.post("/api/webhooks/azure-ai", response_model=ThirdPartyResponse)
async def azure_ai_webhook(request: AzureAIRequest) -> ThirdPartyResponse:
    """Webhook endpoint for Azure AI Agent integration.

    Allows Azure AI services to invoke enterprise agents.

    Example Azure AI configuration:
    - Create a custom skill or connector
    - Configure endpoint: https://your-server/api/webhooks/azure-ai
    - Map input/output schema to request/response models
    """
    if not enterprise_agents_loaded:
        return ThirdPartyResponse(
            success=False,
            error="Enterprise agents not loaded",
            source="azure-ai",
        )

    session_id = request.session_id or f"azure-{request.deployment_id or 'default'}"
    success, response = _invoke_enterprise_agent(request.agent_type, request.query)

    return ThirdPartyResponse(
        success=success,
        response=response if success else None,
        error=response if not success else None,
        session_id=session_id,
        agent_type=request.agent_type,
        source="azure-ai",
        metadata={
            "deployment_id": request.deployment_id,
            "resource_group": request.resource_group,
            "subscription_id": request.subscription_id,
        },
    )


@app.post("/api/webhooks/aws-lex", response_model=ThirdPartyResponse)
async def aws_lex_webhook(request: AWSLexRequest) -> ThirdPartyResponse:
    """Webhook endpoint for AWS Lex integration.

    Allows AWS Lex bots to invoke enterprise agents via Lambda fulfillment.

    Example AWS Lex configuration:
    1. Create Lambda function that calls this endpoint
    2. Configure Lex bot to use Lambda for fulfillment
    3. Map Lex slots to request parameters
    """
    if not enterprise_agents_loaded:
        return ThirdPartyResponse(
            success=False,
            error="Enterprise agents not loaded",
            source="aws-lex",
        )

    session_id = request.session_id or f"lex-{request.bot_id or 'default'}"
    success, response = _invoke_enterprise_agent(request.agent_type, request.query)

    return ThirdPartyResponse(
        success=success,
        response=response if success else None,
        error=response if not success else None,
        session_id=session_id,
        agent_type=request.agent_type,
        source="aws-lex",
        metadata={
            "bot_id": request.bot_id,
            "bot_alias_id": request.bot_alias_id,
            "locale_id": request.locale_id,
            "session_attributes": request.session_attributes,
        },
    )


# ============================================================================
# Enterprise Agent Endpoints
# ============================================================================


@app.get("/api/enterprise/agents")
async def list_enterprise_agents() -> dict:
    """List all available enterprise agents and their status."""
    return {
        "status": "available" if enterprise_agents_loaded else "unavailable",
        "agents": {
            "research": {
                "loaded": research_agent is not None,
                "description": "AI Research Agent for web search and information synthesis",
                "endpoint": "/api/enterprise/research/invoke",
            },
            "content": {
                "loaded": content_agent is not None,
                "description": "Content Generation Agent for LinkedIn, X, and blog posts",
                "endpoint": "/api/enterprise/content/invoke",
            },
            "data_analyst": {
                "loaded": data_analyst_agent is not None,
                "description": "Data Analyst Agent for Excel/CSV analysis",
                "endpoint": "/api/enterprise/data-analyst/invoke",
            },
            "document": {
                "loaded": document_agent is not None,
                "description": "IT Document Generator for SOP/WLI/Policy creation",
                "endpoint": "/api/enterprise/documents/invoke",
            },
            "multilingual_rag": {
                "loaded": multilingual_rag_agent is not None,
                "description": "Multilingual RAG Agent for document Q&A",
                "endpoint": "/api/enterprise/rag/invoke",
            },
            "hitl_support": {
                "loaded": hitl_support_agent is not None,
                "description": "Human-in-the-Loop IT Support Agent",
                "endpoint": "/api/enterprise/support/invoke",
            },
            "code_assistant": {
                "loaded": code_assistant_agent is not None,
                "description": "Code Assistant for application modernization",
                "endpoint": "/api/enterprise/code/invoke",
            },
            "document_intelligence": {
                "loaded": document_intelligence_agent is not None,
                "description": "Document Intelligence Agent for multi-format document analysis, RAG, translation",
                "endpoint": "/api/enterprise/document-intelligence/invoke",
            },
        },
    }


_VALID_AGENTS = {
    "research",
    "content",
    "data-analyst",
    "documents",
    "rag",
    "support",
    "code",
    "document-intelligence",
}


@app.get("/api/enterprise/{agent}/estimate", tags=["Enterprise Agents"])
async def estimate_agent_cost(
    agent: str,
    message: str,
) -> dict:
    """Estimate token count and USD cost before invoking an enterprise agent.

    This endpoint is free — it does **not** call the LLM.

    Args:
        agent: Agent type (research, content, data-analyst, …).
        message: The message you intend to send.

    Returns:
        Dict with ``input_tokens``, ``estimated_output_tokens``,
        ``estimated_cost_usd``, and ``model``.
    """
    if agent not in _VALID_AGENTS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown agent '{agent}'. Valid agents: {sorted(_VALID_AGENTS)}",
        )
    return get_cost_estimator().estimate(message, agent)


@app.post("/api/enterprise/research/invoke", response_model=EnterpriseAgentResponse, tags=["Enterprise Agents"])
async def research_agent_invoke(
    request: ResearchAgentRequest, background_tasks: BackgroundTasks
) -> EnterpriseAgentResponse:
    """Invoke the Research Agent for web search and information synthesis.

    Responses are served from the in-memory cache when ``CACHE_ENABLED=true``
    and an identical (whitespace-normalised) query has been answered before.
    Cache hits are indicated by ``cached: true`` in the response.

    Args:
        request: Research query and optional session ID.

    Returns:
        Research findings and synthesized information.
    """
    if not enterprise_agents_loaded or research_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Research Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.",
        )

    # Cache key is message-only (not session-scoped): research responses are stateless
    # with respect to session context — the same query always yields the same answer.
    # Check cache before invoking the LLM (no-op when CACHE_ENABLED=false)
    _agent_cache = get_cache()
    cached_response = _agent_cache.get("research", request.query)
    if cached_response is not None:
        return EnterpriseAgentResponse(
            success=True,
            response=cached_response,
            agent_type="research",
            cached=True,
        )

    try:
        result = research_agent.research(
            query=request.query,
            session_id=request.session_id,
        )
        # Extract response from LangGraph state messages
        response_text = extract_agent_response(result)

        # Persist result in cache for future identical queries
        _agent_cache.set("research", request.query, response_text)

        submit_for_evaluation(background_tasks, "research", request.query, response_text)

        return EnterpriseAgentResponse(
            success=True,
            response=response_text,
            session_id=result.get("session_id") if isinstance(result, dict) else getattr(result, "session_id", None),
            agent_type="research",
            cached=False,
        )
    except Exception as e:
        return EnterpriseAgentResponse(
            success=False,
            error=str(e),
            agent_type="research",
        )


@app.post("/api/enterprise/research/stream", tags=["Enterprise Agents"])
async def research_agent_stream(request: ResearchAgentRequest) -> StreamingResponse:
    """Stream responses from the Research Agent using Server-Sent Events.

    Streams incremental tokens, tool events, and a final ``complete`` event.

    Args:
        request: Research query and optional session ID.

    Returns:
        SSE stream of events.
    """
    if not enterprise_agents_loaded or research_agent is None:

        async def _unavailable():
            yield f"data: {_serialize_sse({'type': 'error', 'data': {'error': 'Research Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.'}})}\n\n"

        return StreamingResponse(_unavailable(), media_type="text/event-stream")

    async def _event_generator():
        try:
            async for event in research_agent.astream(
                message=request.query,
                session_id=request.session_id,
            ):
                yield f"data: {_serialize_sse(event)}\n\n"
        except GeneratorExit:
            pass
        except Exception:
            yield f"data: {_serialize_sse({'type': 'error', 'data': {'error': 'An error occurred processing your request'}})}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/api/enterprise/content/invoke", response_model=EnterpriseAgentResponse, tags=["Enterprise Agents"])
async def content_agent_invoke(request: ContentAgentRequest) -> EnterpriseAgentResponse:
    """Invoke the Content Generation Agent.

    Generates content for LinkedIn, X (Twitter), or blog posts.
    May require human approval for publishing.

    Args:
        request: Content topic, platform, tone, and audience.

    Returns:
        Generated content draft or published content.
    """
    if not enterprise_agents_loaded or content_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Content Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.",
        )

    try:
        # Ensure auto_approve mode for API usage (skip HITL review)
        result = content_agent.create_content(
            topic=request.topic,
            platform=request.platform,
            tone=request.tone,
            target_audience=request.audience,
            session_id=request.session_id,
            auto_approve=True,  # Skip HITL review for API calls
        )
        # Extract response from LangGraph state messages
        response_text = extract_agent_response(result)
        return EnterpriseAgentResponse(
            success=True,
            response=response_text,
            session_id=result.get("session_id") if isinstance(result, dict) else getattr(result, "session_id", None),
            agent_type="content",
            tool_calls=result.get("tool_calls") if isinstance(result, dict) else None,
        )
    except Exception as e:
        return EnterpriseAgentResponse(
            success=False,
            error=str(e),
            agent_type="content",
        )


@app.post("/api/enterprise/content/stream", tags=["Enterprise Agents"])
async def content_agent_stream(request: ContentAgentRequest) -> StreamingResponse:
    """Stream responses from the Content Generation Agent using Server-Sent Events.

    Args:
        request: Content topic, platform, tone, and audience.

    Returns:
        SSE stream of events.
    """
    if not enterprise_agents_loaded or content_agent is None:

        async def _unavailable():
            yield f"data: {_serialize_sse({'type': 'error', 'data': {'error': 'Content Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.'}})}\n\n"

        return StreamingResponse(_unavailable(), media_type="text/event-stream")

    message = f"Create {request.tone} content for {request.platform} about: {request.topic}. Target audience: {request.audience}."

    async def _event_generator():
        try:
            async for event in content_agent.astream(
                message=message,
                session_id=request.session_id,
            ):
                yield f"data: {_serialize_sse(event)}\n\n"
        except GeneratorExit:
            pass
        except Exception:
            yield f"data: {_serialize_sse({'type': 'error', 'data': {'error': 'An error occurred processing your request'}})}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/api/enterprise/data-analyst/invoke", response_model=EnterpriseAgentResponse, tags=["Enterprise Agents"])
async def data_analyst_invoke(request: DataAnalystRequest) -> EnterpriseAgentResponse:
    """Invoke the Data Analyst Agent.

    Analyzes Excel/CSV data and generates insights.

    Args:
        request: Analysis message and optional session ID.

    Returns:
        Data analysis results and insights.
    """
    if not enterprise_agents_loaded or data_analyst_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Data Analyst Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.",
        )

    try:
        # Use default_session for consistency with upload endpoint when not specified
        effective_session_id = request.session_id or "default_session"
        result = data_analyst_agent.invoke(
            message=request.message,
            session_id=effective_session_id,
        )
        # Extract response from LangGraph state messages
        response_text = extract_agent_response(result)
        return EnterpriseAgentResponse(
            success=True,
            response=response_text,
            session_id=effective_session_id,
            agent_type="data_analyst",
            tool_calls=result.get("tool_calls") if isinstance(result, dict) else None,
        )
    except Exception as e:
        return EnterpriseAgentResponse(
            success=False,
            error=str(e),
            agent_type="data_analyst",
        )


@app.post("/api/enterprise/data-analyst/stream", tags=["Enterprise Agents"])
async def data_analyst_stream(request: DataAnalystRequest) -> StreamingResponse:
    """Stream responses from the Data Analyst Agent using Server-Sent Events.

    Args:
        request: Analysis message and optional session ID.

    Returns:
        SSE stream of events.
    """
    if not enterprise_agents_loaded or data_analyst_agent is None:

        async def _unavailable():
            yield f"data: {_serialize_sse({'type': 'error', 'data': {'error': 'Data Analyst Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.'}})}\n\n"

        return StreamingResponse(_unavailable(), media_type="text/event-stream")

    effective_session_id = request.session_id or "default_session"

    async def _event_generator():
        try:
            async for event in data_analyst_agent.astream(
                message=request.message,
                session_id=effective_session_id,
            ):
                yield f"data: {_serialize_sse(event)}\n\n"
        except GeneratorExit:
            pass
        except Exception:
            yield f"data: {_serialize_sse({'type': 'error', 'data': {'error': 'An error occurred processing your request'}})}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/api/enterprise/data-analyst/upload", tags=["Enterprise Agents"])
async def data_analyst_upload(
    file: UploadFile = File(...),
    session_id: str | None = Form(None),
) -> dict:
    """Upload a file for data analysis.

    Supports Excel (.xlsx, .xls) and CSV files.

    Args:
        file: The data file to upload.
        session_id: Optional session ID for data isolation.

    Returns:
        Upload status and file information.
    """
    if not enterprise_agents_loaded or data_analyst_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Data Analyst Agent not available.",
        )

    filename = file.filename or "unknown"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    allowed = {".xlsx", ".xls", ".csv"}

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(allowed)}",
        )

    content = await file.read()

    # Save to temp location for analysis
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    # Automatically load the file into the agent's memory
    try:
        from app.agents.data_analyst.data_analyst_agent import _dataframes, load_csv_file, load_excel_file

        # Use provided session_id or default
        effective_session_id = session_id or "default_session"

        if ext in [".xlsx", ".xls"]:
            result = load_excel_file.invoke({"file_path": tmp_path, "session_id": effective_session_id})
        else:  # CSV
            result = load_csv_file.invoke({"file_path": tmp_path, "session_id": effective_session_id})

        # Check if the tool returned an error
        if result.startswith("Error"):
            return {
                "status": "error",
                "filename": filename,
                "message": result,
            }

        # Verify data was actually loaded
        if session_id not in _dataframes or "current" not in _dataframes[session_id]:
            return {
                "status": "error",
                "filename": filename,
                "message": "File was processed but data was not stored. Please try again.",
            }

        # Return success with session_id for subsequent requests
        df = _dataframes[session_id]["current"]
        return {
            "status": "success",
            "filename": filename,
            "session_id": session_id,
            "rows": len(df),
            "columns": len(df.columns),
            "message": f"File loaded successfully! Ready for analysis. ({len(df)} rows, {len(df.columns)} columns)",
        }
    except Exception as e:
        # Return actual error for debugging
        return {
            "status": "error",
            "filename": filename,
            "message": f"Failed to load file: {str(e)}",
        }


@app.get("/api/enterprise/data-analyst/status", tags=["Enterprise Agents"])
async def data_analyst_status() -> dict:
    """Check the current status of data loaded in the Data Analyst Agent.

    Returns:
        Status of loaded data including session info and data shape.
    """
    try:
        from app.agents.data_analyst.data_analyst_agent import _dataframes

        sessions = {}
        for session_id, data in _dataframes.items():
            if "current" in data:
                df = data["current"]
                sessions[session_id] = {
                    "has_data": True,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist()[:10],  # First 10 columns
                }
            else:
                sessions[session_id] = {"has_data": False}

        return {
            "status": "ok",
            "total_sessions": len(_dataframes),
            "sessions": sessions,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


@app.post("/api/enterprise/documents/invoke", response_model=EnterpriseAgentResponse, tags=["Enterprise Agents"])
async def document_agent_invoke(request: DocumentAgentRequest) -> EnterpriseAgentResponse:
    """Invoke the IT Document Generator Agent.

    Generates SOP, WLI, or Policy documents.

    Args:
        request: Document type, title, description, and sections.

    Returns:
        Generated document content.
    """
    if not enterprise_agents_loaded or document_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Document Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.",
        )

    try:
        result = document_agent.create_document(
            doc_type=request.doc_type,
            title=request.title,
            department=getattr(request, "department", ""),
            purpose=request.description,
            additional_context=str(getattr(request, "sections", [])),
            session_id=request.session_id,
        )
        # Extract response from LangGraph state messages
        response_text = extract_agent_response(result)
        return EnterpriseAgentResponse(
            success=True,
            response=response_text,
            session_id=result.get("session_id") if isinstance(result, dict) else getattr(result, "session_id", None),
            agent_type="document",
            tool_calls=result.get("tool_calls") if isinstance(result, dict) else None,
        )
    except Exception as e:
        return EnterpriseAgentResponse(
            success=False,
            error=str(e),
            agent_type="document",
        )


@app.post("/api/enterprise/documents/stream", tags=["Enterprise Agents"])
async def document_agent_stream(request: DocumentAgentRequest) -> StreamingResponse:
    """Stream responses from the IT Document Generator Agent using Server-Sent Events.

    Args:
        request: Document type, title, description, and sections.

    Returns:
        SSE stream of events.
    """
    if not enterprise_agents_loaded or document_agent is None:

        async def _unavailable():
            yield f"data: {_serialize_sse({'type': 'error', 'data': {'error': 'Document Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.'}})}\n\n"

        return StreamingResponse(_unavailable(), media_type="text/event-stream")

    sections_str = str(request.sections or [])
    message = f"Create a {request.doc_type} document titled '{request.title}'. Purpose: {request.description}. Sections: {sections_str}."

    async def _event_generator():
        try:
            async for event in document_agent.astream(
                message=message,
                session_id=request.session_id,
            ):
                yield f"data: {_serialize_sse(event)}\n\n"
        except GeneratorExit:
            pass
        except Exception:
            yield f"data: {_serialize_sse({'type': 'error', 'data': {'error': 'An error occurred processing your request'}})}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/api/enterprise/rag/invoke", response_model=EnterpriseAgentResponse, tags=["Enterprise Agents"])
async def rag_agent_invoke(request: RAGAgentRequest) -> EnterpriseAgentResponse:
    """Invoke the Multilingual RAG Agent.

    Answers questions based on uploaded documents with multilingual support.

    Args:
        request: Query, optional language, and session ID.

    Returns:
        Answer based on document context.
    """
    if not enterprise_agents_loaded or multilingual_rag_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Multilingual RAG Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.",
        )

    try:
        result = multilingual_rag_agent.query(
            question=request.query,
            language=request.language or "auto",  # Default to "auto" if None
            session_id=request.session_id,
        )
        # Extract response from LangGraph state messages
        response_text = extract_agent_response(result)
        return EnterpriseAgentResponse(
            success=True,
            response=response_text,
            session_id=result.get("session_id") if isinstance(result, dict) else getattr(result, "session_id", None),
            agent_type="multilingual_rag",
            tool_calls=result.get("tool_calls") if isinstance(result, dict) else None,
        )
    except Exception as e:
        return EnterpriseAgentResponse(
            success=False,
            error=str(e),
            agent_type="multilingual_rag",
        )


@app.post("/api/enterprise/rag/stream", tags=["Enterprise Agents"])
async def rag_agent_stream(request: RAGAgentRequest) -> StreamingResponse:
    """Stream responses from the Multilingual RAG Agent using Server-Sent Events.

    Args:
        request: Query, optional language, and session ID.

    Returns:
        SSE stream of events.
    """
    if not enterprise_agents_loaded or multilingual_rag_agent is None:

        async def _unavailable():
            yield f"data: {_serialize_sse({'type': 'error', 'data': {'error': 'RAG Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.'}})}\n\n"

        return StreamingResponse(_unavailable(), media_type="text/event-stream")

    language_hint = f" Answer in {request.language}." if request.language and request.language != "auto" else ""
    message = f"{request.query}{language_hint}"

    async def _event_generator():
        try:
            async for event in multilingual_rag_agent.astream(
                message=message,
                session_id=request.session_id,
            ):
                yield f"data: {_serialize_sse(event)}\n\n"
        except GeneratorExit:
            pass
        except Exception:
            yield f"data: {_serialize_sse({'type': 'error', 'data': {'error': 'An error occurred processing your request'}})}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/api/enterprise/rag/upload", tags=["Enterprise Agents"])
async def rag_upload_document(
    file: UploadFile = File(...),
    session_id: str | None = Form(None),
) -> dict:
    """Upload a document for RAG processing.

    Supports PDF, Word, and text files in multiple languages.

    Args:
        file: The document to upload.
        session_id: Optional session ID (for future session-scoped storage).

    Returns:
        Upload status and document information.
    """
    if not enterprise_agents_loaded or multilingual_rag_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Multilingual RAG Agent not available.",
        )

    filename = file.filename or "unknown"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    allowed = {".pdf", ".txt", ".docx", ".doc", ".md"}

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(allowed)}",
        )

    content_bytes = await file.read()

    # Extract text content based on file type
    try:
        if ext in [".txt", ".md"]:
            # Plain text files - decode as UTF-8
            text_content = content_bytes.decode("utf-8", errors="replace")
        elif ext == ".pdf":
            # PDF files - use PyPDF2 or pdfplumber
            try:
                import io

                from PyPDF2 import PdfReader

                pdf_reader = PdfReader(io.BytesIO(content_bytes))
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text() or ""
            except ImportError:
                raise HTTPException(
                    status_code=500,
                    detail="PDF processing requires PyPDF2. Install with: pip install PyPDF2",
                )
        elif ext in [".docx", ".doc"]:
            # Word files - use python-docx
            try:
                import io

                from docx import Document

                doc = Document(io.BytesIO(content_bytes))
                text_content = "\n".join([para.text for para in doc.paragraphs])
            except ImportError:
                raise HTTPException(
                    status_code=500,
                    detail="Word processing requires python-docx. Install with: pip install python-docx",
                )
        else:
            text_content = content_bytes.decode("utf-8", errors="replace")

        if not text_content.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from the document. The file may be empty or corrupted.",
            )

        # Upload extracted text to the RAG agent
        result = multilingual_rag_agent.upload(text_content, filename, ext.replace(".", ""))
        return {"status": "success", "message": result, "filename": filename}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.post("/api/enterprise/support/invoke", response_model=EnterpriseAgentResponse, tags=["Enterprise Agents"])
async def hitl_support_invoke(request: HITLSupportRequest) -> EnterpriseAgentResponse:
    """Invoke the Human-in-the-Loop IT Support Agent.

    Handles IT support requests with approval gates for sensitive actions.

    Args:
        request: Support message, session ID, and user ID.

    Returns:
        Support response or approval request.
    """
    if not enterprise_agents_loaded or hitl_support_agent is None:
        raise HTTPException(
            status_code=503,
            detail="HITL Support Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.",
        )

    try:
        result = hitl_support_agent.invoke(
            message=request.message,
            session_id=request.session_id,
            user_id=request.user_id,
        )
        # Extract response from LangGraph state messages
        response_text = extract_agent_response(result)
        return EnterpriseAgentResponse(
            success=True,
            response=response_text,
            session_id=result.get("session_id") if isinstance(result, dict) else getattr(result, "session_id", None),
            agent_type="hitl_support",
            tool_calls=result.get("tool_calls") if isinstance(result, dict) else None,
        )
    except Exception as e:
        return EnterpriseAgentResponse(
            success=False,
            error=str(e),
            agent_type="hitl_support",
        )


@app.post("/api/enterprise/support/stream", tags=["Enterprise Agents"])
async def hitl_support_stream(request: HITLSupportRequest) -> StreamingResponse:
    """Stream responses from the HITL IT Support Agent using Server-Sent Events.

    Args:
        request: Support message, session ID, and user ID.

    Returns:
        SSE stream of events.
    """
    if not enterprise_agents_loaded or hitl_support_agent is None:

        async def _unavailable():
            yield f"data: {_serialize_sse({'type': 'error', 'data': {'error': 'HITL Support Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.'}})}\n\n"

        return StreamingResponse(_unavailable(), media_type="text/event-stream")

    async def _event_generator():
        try:
            async for event in hitl_support_agent.astream(
                message=request.message,
                session_id=request.session_id,
                user_id=request.user_id,
            ):
                yield f"data: {_serialize_sse(event)}\n\n"
        except GeneratorExit:
            pass
        except Exception:
            yield f"data: {_serialize_sse({'type': 'error', 'data': {'error': 'An error occurred processing your request'}})}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/api/enterprise/support/approve", tags=["Enterprise Agents"])
async def hitl_approve_action(request: HITLApprovalRequest) -> dict:
    """Approve or reject a pending action in HITL Support.

    Args:
        request: Session ID, action ID, approval status, and approver.

    Returns:
        Approval result.
    """
    if not enterprise_agents_loaded or hitl_support_agent is None:
        raise HTTPException(
            status_code=503,
            detail="HITL Support Agent not available.",
        )

    try:
        result = hitl_support_agent.approve_action(
            session_id=request.session_id,
            action_id=request.action_id,
            approved=request.approved,
            approved_by=request.approved_by,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/enterprise/code/invoke", response_model=EnterpriseAgentResponse, tags=["Enterprise Agents"])
async def code_assistant_invoke(request: CodeAssistantRequest) -> EnterpriseAgentResponse:
    """Invoke the Code Assistant Agent.

    Analyzes code for modernization opportunities and security issues.

    Args:
        request: Code, language, action type, and security flag.

    Returns:
        Analysis results and recommendations.
    """
    if not enterprise_agents_loaded or code_assistant_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Code Assistant Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.",
        )

    try:
        if request.action == "analyze":
            result = code_assistant_agent.analyze(
                code=request.code,
                language=request.language,
                include_security=request.include_security,
                session_id=request.session_id,
            )
        else:  # modernize
            result = code_assistant_agent.modernize(
                code=request.code,
                language=request.language,
                session_id=request.session_id,
            )

        # Extract response from LangGraph state messages
        response_text = extract_agent_response(result)
        return EnterpriseAgentResponse(
            success=True,
            response=response_text,
            session_id=result.get("session_id") if isinstance(result, dict) else getattr(result, "session_id", None),
            agent_type="code_assistant",
            tool_calls=result.get("tool_calls") if isinstance(result, dict) else None,
        )
    except Exception as e:
        return EnterpriseAgentResponse(
            success=False,
            error=str(e),
            agent_type="code_assistant",
        )


@app.post("/api/enterprise/code/stream", tags=["Enterprise Agents"])
async def code_assistant_stream(request: CodeAssistantRequest) -> StreamingResponse:
    """Stream responses from the Code Assistant Agent using Server-Sent Events.

    Args:
        request: Code, language, action type, and security flag.

    Returns:
        SSE stream of events.
    """
    if not enterprise_agents_loaded or code_assistant_agent is None:

        async def _unavailable():
            yield f"data: {_serialize_sse({'type': 'error', 'data': {'error': 'Code Assistant Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.'}})}\n\n"

        return StreamingResponse(_unavailable(), media_type="text/event-stream")

    if request.action == "analyze":
        security_line = "3. Security vulnerabilities\n" if request.include_security else ""
        message = (
            f"Please analyze this {request.language} code:\n\n"
            f"```{request.language}\n{request.code}\n```\n\n"
            f"Provide:\n1. Code structure analysis\n2. Legacy patterns found\n"
            f"{security_line}4. Modernization suggestions\n5. Example improvements"
        )
    else:
        message = (
            f"Please help modernize this {request.language} code:\n\n"
            f"```{request.language}\n{request.code}\n```\n\n"
            f"Provide step-by-step modernization recommendations."
        )

    async def _event_generator():
        try:
            async for event in code_assistant_agent.astream(
                message=message,
                session_id=request.session_id,
            ):
                yield f"data: {_serialize_sse(event)}\n\n"
        except GeneratorExit:
            pass
        except Exception:
            yield f"data: {_serialize_sse({'type': 'error', 'data': {'error': 'An error occurred processing your request'}})}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ============================================================================
# Document Intelligence Agent Endpoints
# ============================================================================


@app.post(
    "/api/enterprise/document-intelligence/invoke", response_model=EnterpriseAgentResponse, tags=["Enterprise Agents"]
)
async def document_intelligence_invoke(request: DocumentIntelligenceRequest) -> EnterpriseAgentResponse:
    """Invoke the Document Intelligence Agent.

    Chat with the agent for document analysis, Q&A, translation, and web search.

    Args:
        request: Message, optional session ID, and target language.

    Returns:
        Agent response with document insights.
    """
    if not enterprise_agents_loaded or document_intelligence_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Document Intelligence Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.",
        )

    try:
        result = document_intelligence_agent.chat(
            message=request.message,
            session_id=request.session_id,
            target_language=request.target_language,
        )
        return EnterpriseAgentResponse(
            success=True,
            response=result.get("response", ""),
            session_id=result.get("session_id"),
            agent_type="document_intelligence",
        )
    except Exception as e:
        return EnterpriseAgentResponse(
            success=False,
            error=str(e),
            agent_type="document_intelligence",
        )


@app.post("/api/enterprise/document-intelligence/stream", tags=["Enterprise Agents"])
async def document_intelligence_stream(request: DocumentIntelligenceRequest) -> StreamingResponse:
    """Stream responses from the Document Intelligence Agent using Server-Sent Events.

    Args:
        request: Message, optional session ID, and target language.

    Returns:
        SSE stream of events.
    """
    if not enterprise_agents_loaded or document_intelligence_agent is None:

        async def _unavailable():
            yield f"data: {_serialize_sse({'type': 'error', 'data': {'error': 'Document Intelligence Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.'}})}\n\n"

        return StreamingResponse(_unavailable(), media_type="text/event-stream")

    language_hint = f" Respond in {request.target_language}." if request.target_language else ""
    message = f"{request.message}{language_hint}"

    async def _event_generator():
        try:
            async for event in document_intelligence_agent.astream(
                message=message,
                session_id=request.session_id,
            ):
                yield f"data: {_serialize_sse(event)}\n\n"
        except GeneratorExit:
            pass
        except Exception:
            yield f"data: {_serialize_sse({'type': 'error', 'data': {'error': 'An error occurred processing your request'}})}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post(
    "/api/enterprise/document-intelligence/upload",
    response_model=DocumentIntelligenceUploadResponse,
    tags=["Enterprise Agents"],
)
async def document_intelligence_upload(
    file: UploadFile = File(...),
    session_id: str | None = Form(None),
) -> DocumentIntelligenceUploadResponse:
    """Upload a document for analysis.

    Supports PDF, TXT, DOCX, PPTX, and images (PNG/JPG with OCR).

    Args:
        file: The document to upload.
        session_id: Optional session ID for document isolation.

    Returns:
        Upload status with document ID and metadata.
    """
    if not enterprise_agents_loaded or document_intelligence_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Document Intelligence Agent not available.",
        )

    filename = file.filename or "unknown"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    allowed = {".pdf", ".txt", ".docx", ".doc", ".pptx", ".ppt", ".png", ".jpg", ".jpeg"}

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(allowed)}",
        )

    content_bytes = await file.read()

    try:
        result = document_intelligence_agent.upload_document(
            content=content_bytes,
            filename=filename,
            session_id=session_id,
        )

        # Parse result message for metadata
        return DocumentIntelligenceUploadResponse(
            success=result.get("success", False),
            message=result.get("message", ""),
            filename=filename,
            file_type=ext.replace(".", ""),
        )
    except Exception as e:
        return DocumentIntelligenceUploadResponse(
            success=False,
            error=str(e),
            filename=filename,
        )


@app.get("/api/enterprise/document-intelligence/documents/{session_id}", tags=["Enterprise Agents"])
async def document_intelligence_list_documents(session_id: str) -> dict:
    """List all documents in a session.

    Args:
        session_id: Session identifier.

    Returns:
        List of documents with metadata.
    """
    if not enterprise_agents_loaded or document_intelligence_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Document Intelligence Agent not available.",
        )

    documents = document_intelligence_agent.get_documents(session_id)
    return {
        "success": True,
        "session_id": session_id,
        "documents": documents,
        "total": len(documents),
    }


@app.delete("/api/enterprise/document-intelligence/documents/{session_id}", tags=["Enterprise Agents"])
async def document_intelligence_clear_documents(
    session_id: str,
    document_ids: str | None = None,
) -> dict:
    """Clear documents from a session.

    Args:
        session_id: Session identifier.
        document_ids: Optional comma-separated list of document IDs to clear.

    Returns:
        Confirmation with count of cleared documents.
    """
    if not enterprise_agents_loaded or document_intelligence_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Document Intelligence Agent not available.",
        )

    doc_ids = None
    if document_ids:
        doc_ids = [d.strip() for d in document_ids.split(",")]

    count = document_intelligence_agent.clear_documents(session_id, doc_ids)
    return {
        "success": True,
        "session_id": session_id,
        "cleared_count": count,
    }


# ============================================================================
# Cache Management Endpoints
# ============================================================================


@app.get("/api/cache/stats", tags=["Cache"])
async def cache_stats() -> dict:
    """Return current response-cache statistics.

    The cache is opt-in via ``CACHE_ENABLED=true``.  When disabled the size
    will always be zero because all cache writes are suppressed.

    Returns:
        Dictionary with ``enabled`` (bool) and ``size`` (int) fields.
    """
    return {"enabled": is_cache_enabled(), "size": get_cache().size()}


@app.delete("/api/cache/clear", tags=["Cache"])
async def cache_clear() -> dict:
    """Clear all entries from the in-memory response cache.

    Returns:
        Confirmation dictionary with ``cleared: true``.
    """
    get_cache().clear()
    return {"cleared": True}


# ============================================================================
# Deep Agent Endpoints
# ============================================================================


@app.post("/api/deepagent/start", response_model=DeepAgentStartResponse, tags=["Deep Agent"])
async def deep_agent_start(request: DeepAgentStartRequest) -> DeepAgentStartResponse:
    """Start a new Deep Agent session.

    The IT Operations Deep Agent can handle complex IT managed services tasks
    including incident management, change management, problem management,
    asset management, SLA monitoring, and knowledge management.

    Returns:
        Session ID and welcome message.
    """
    if not deep_agent_loaded or it_operations_deep_agent is None:
        return DeepAgentStartResponse(
            success=False,
            error="Deep Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.",
        )

    import uuid

    session_id = str(uuid.uuid4())

    return DeepAgentStartResponse(
        success=True,
        session_id=session_id,
        message="IT Operations Deep Agent session started. I can help with incident management, change requests, problem analysis, CMDB queries, SLA tracking, and knowledge base operations.",
    )


@app.post("/api/deepagent/chat", response_model=DeepAgentChatResponse, tags=["Deep Agent"])
async def deep_agent_chat(request: DeepAgentChatRequest) -> DeepAgentChatResponse:
    """Chat with the IT Operations Deep Agent.

    The Deep Agent will:
    1. Plan complex tasks using todos
    2. Delegate to specialized subagents
    3. Query ServiceNow for ITSM data
    4. Store context in workspace files

    Args:
        request: Chat message and optional session ID.

    Returns:
        Agent response with todos, files, and tool calls.
    """
    if not deep_agent_loaded or it_operations_deep_agent is None:
        return DeepAgentChatResponse(
            success=False,
            error="Deep Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.",
        )

    try:
        result = await it_operations_deep_agent.achat(
            message=request.message,
            session_id=request.session_id,
            user_id=request.user_id,
        )

        return DeepAgentChatResponse(
            success=True,
            response=result.get("response"),
            session_id=result.get("session_id"),
            todos=result.get("todos"),
            files=result.get("files"),
            tool_calls=result.get("tool_calls"),
            iteration_count=result.get("iteration_count"),
        )
    except Exception as e:
        return DeepAgentChatResponse(
            success=False,
            session_id=request.session_id,
            error=str(e),
        )


@app.post("/api/deepagent/chat/stream", tags=["Deep Agent"])
async def deep_agent_chat_stream(request: DeepAgentChatRequest):
    """Stream chat with the IT Operations Deep Agent.

    Uses Server-Sent Events (SSE) to stream:
    - thinking: Agent's reasoning steps
    - tool_start: When a tool is being called
    - tool_result: Tool execution results
    - todo_update: Todo list changes
    - token: Streaming response tokens
    - complete: Final response with all context

    Args:
        request: Chat message and session ID.

    Returns:
        SSE stream of events.
    """
    import json
    import traceback

    def serialize_event(event: dict) -> str:
        """Serialize event to JSON, handling datetime and other types."""

        def default_serializer(obj):
            if hasattr(obj, "isoformat"):
                return obj.isoformat()
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "__dict__"):
                return str(obj)
            return str(obj)

        return json.dumps(event, default=default_serializer)

    if not deep_agent_loaded or it_operations_deep_agent is None:

        async def error_generator():
            yield f"data: {serialize_event({'type': 'error', 'data': {'error': 'Deep Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.'}})}\n\n"

        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream",
        )

    async def event_generator():
        try:
            print(f"[DEBUG] Starting stream for session: {request.session_id}")
            async for event in it_operations_deep_agent.astream_chat(
                message=request.message,
                session_id=request.session_id,
                user_id=request.user_id,
            ):
                try:
                    yield f"data: {serialize_event(event)}\n\n"
                except Exception as serialize_err:
                    print(f"[ERROR] Failed to serialize event: {serialize_err}")
                    yield f"data: {serialize_event({'type': 'error', 'data': {'error': f'Serialization error: {serialize_err}'}})}\n\n"
            print(f"[DEBUG] Stream completed for session: {request.session_id}")
        except GeneratorExit:
            print(f"[DEBUG] Client disconnected from stream: {request.session_id}")
        except Exception as e:
            print(f"[ERROR] Stream error for session {request.session_id}: {e}")
            traceback.print_exc()
            yield f"data: {serialize_event({'type': 'error', 'data': {'error': str(e)}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/deepagent/context/{session_id}", response_model=DeepAgentContextResponse, tags=["Deep Agent"])
async def deep_agent_context(session_id: str) -> DeepAgentContextResponse:
    """Get the current context for a Deep Agent session.

    Returns todos, workspace files, and session metadata.

    Args:
        session_id: The session identifier.

    Returns:
        Session context including todos and files.
    """
    if not deep_agent_loaded or it_operations_deep_agent is None:
        return DeepAgentContextResponse(
            success=False,
            session_id=session_id,
            error="Deep Agent not available.",
        )

    try:
        context = it_operations_deep_agent.get_session_context(session_id)
        return DeepAgentContextResponse(
            success=True,
            session_id=session_id,
            todos=context.get("todos"),
            files=context.get("files"),
            metadata=context.get("metadata"),
        )
    except Exception as e:
        return DeepAgentContextResponse(
            success=False,
            session_id=session_id,
            error=str(e),
        )


@app.get("/api/deepagent/todos/{session_id}", tags=["Deep Agent"])
async def deep_agent_todos(session_id: str) -> dict:
    """Get the todo list for a Deep Agent session.

    Args:
        session_id: The session identifier.

    Returns:
        List of todos with their status.
    """
    if not deep_agent_loaded or it_operations_deep_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Deep Agent not available.",
        )

    context = it_operations_deep_agent.get_session_context(session_id)
    todos = context.get("todos", [])

    summary = {
        "total": len(todos),
        "pending": len([t for t in todos if t.get("status") == "pending"]),
        "in_progress": len([t for t in todos if t.get("status") == "in_progress"]),
        "completed": len([t for t in todos if t.get("status") == "completed"]),
    }

    return {
        "success": True,
        "session_id": session_id,
        "todos": todos,
        "summary": summary,
    }


@app.get("/api/deepagent/files/{session_id}", tags=["Deep Agent"])
async def deep_agent_files(session_id: str) -> dict:
    """Get the workspace files for a Deep Agent session.

    Args:
        session_id: The session identifier.

    Returns:
        List of file paths in the session workspace.
    """
    if not deep_agent_loaded or it_operations_deep_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Deep Agent not available.",
        )

    context = it_operations_deep_agent.get_session_context(session_id)

    return {
        "success": True,
        "session_id": session_id,
        "files": context.get("files", []),
        "count": len(context.get("files", [])),
    }


@app.get("/api/deepagent/subagents", tags=["Deep Agent"])
async def deep_agent_subagents() -> dict:
    """Get available subagents for the Deep Agent.

    Returns:
        List of available subagents with descriptions.
    """
    if not deep_agent_loaded:
        raise HTTPException(
            status_code=503,
            detail="Deep Agent not available.",
        )

    subagents = [
        {
            "name": "incident-manager",
            "description": "Incident lifecycle management - create, update, escalate incidents",
        },
        {
            "name": "change-manager",
            "description": "Change request validation and risk assessment",
        },
        {
            "name": "problem-manager",
            "description": "Root cause analysis and known error management",
        },
        {
            "name": "asset-manager",
            "description": "CMDB queries and CI relationship mapping",
        },
        {
            "name": "sla-monitor",
            "description": "SLA tracking and breach prediction",
        },
        {
            "name": "knowledge-manager",
            "description": "Knowledge base search and article management",
        },
    ]

    return {
        "success": True,
        "subagents": subagents,
        "count": len(subagents),
    }


@app.post("/api/deepagent/upload", response_model=DeepAgentUploadResponse, tags=["Deep Agent"])
async def deep_agent_upload(
    file: UploadFile = File(...),
    session_id: str = Form(...),
) -> DeepAgentUploadResponse:
    """Upload a document to the IT Operations Deep Agent session.

    Supported file types: PDF, TXT, DOCX, DOC, PPTX, PPT, PNG, JPG, JPEG
    Documents are processed, chunked, and indexed for RAG-based search.
    The agent can then use search_attachments tool to query document content.

    Args:
        file: Document file to upload
        session_id: Session ID to associate the document with

    Returns:
        Upload response with document ID and metadata
    """
    if not deep_agent_loaded:
        return DeepAgentUploadResponse(
            success=False,
            error="Deep Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.",
        )

    from app.deepagents.tools.document_tools import process_and_store_document

    # Validate file extension
    allowed_extensions = {".pdf", ".txt", ".docx", ".doc", ".pptx", ".ppt", ".png", ".jpg", ".jpeg"}
    file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""

    if file_ext not in allowed_extensions:
        return DeepAgentUploadResponse(
            success=False,
            filename=file.filename,
            error=f"Unsupported file type: {file_ext}. Supported: {', '.join(allowed_extensions)}",
        )

    try:
        # Read file content
        content = await file.read()

        # Process and store document
        result = process_and_store_document(
            content=content,
            filename=file.filename,
            session_id=session_id,
        )

        return DeepAgentUploadResponse(
            success=True,
            document_id=result["doc_id"],
            filename=result["filename"],
            file_type=result["file_type"],
            chunks_created=result["chunk_count"],
            detected_language=result["language"],
            session_id=session_id,
            message=f"Document '{file.filename}' uploaded successfully with {result['chunk_count']} chunks. Use search_attachments tool to query content.",
        )

    except ValueError as ve:
        return DeepAgentUploadResponse(
            success=False,
            filename=file.filename,
            session_id=session_id,
            error=str(ve),
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return DeepAgentUploadResponse(
            success=False,
            filename=file.filename,
            session_id=session_id,
            error=f"Failed to process document: {e}",
        )


@app.get("/api/deepagent/attachments/{session_id}", tags=["Deep Agent"])
async def deep_agent_list_attachments(session_id: str) -> dict:
    """List all uploaded documents for a Deep Agent session.

    Args:
        session_id: Session identifier

    Returns:
        List of uploaded documents with metadata
    """
    if not deep_agent_loaded:
        raise HTTPException(
            status_code=503,
            detail="Deep Agent not available.",
        )

    from app.deepagents.tools.document_tools import _current_document, _document_metadata

    docs = _document_metadata.get(session_id, {})
    current_doc_id = _current_document.get(session_id)

    documents = []
    for doc_id, meta in docs.items():
        documents.append(
            {
                **meta,
                "is_current": doc_id == current_doc_id,
            }
        )

    return {
        "success": True,
        "session_id": session_id,
        "documents": documents,
        "count": len(documents),
        "current_document_id": current_doc_id,
    }


@app.delete("/api/deepagent/attachments/{session_id}", tags=["Deep Agent"])
async def deep_agent_clear_attachments(
    session_id: str,
    document_ids: str | None = None,
) -> dict:
    """Clear uploaded documents from a Deep Agent session.

    Args:
        session_id: Session identifier
        document_ids: Comma-separated document IDs to clear, or None for all

    Returns:
        Confirmation of cleared documents
    """
    if not deep_agent_loaded:
        raise HTTPException(
            status_code=503,
            detail="Deep Agent not available.",
        )

    from app.deepagents.tools.document_tools import clear_attachments

    doc_ids = document_ids.split(",") if document_ids else None
    result = clear_attachments.invoke({"session_id": session_id, "doc_ids": doc_ids})

    return {
        "success": True,
        "session_id": session_id,
        "message": result,
    }


# ============================================================================
# Sales Intelligence Deep Agent Endpoints
# ============================================================================


@app.post("/api/sales-agent/start", response_model=DeepAgentStartResponse, tags=["Sales Agent"])
async def sales_agent_start(request: DeepAgentStartRequest) -> DeepAgentStartResponse:
    """Start a new Sales Intelligence Deep Agent session.

    The Sales Intelligence Deep Agent assists with deal qualification,
    RFP/RFI responses, solution mapping, competitive positioning,
    and pricing optimization.

    Returns:
        Session ID and welcome message.
    """
    if not deep_agent_loaded or sales_intelligence_deep_agent is None:
        return DeepAgentStartResponse(
            success=False,
            error="Sales Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.",
        )

    import uuid

    session_id = str(uuid.uuid4())

    return DeepAgentStartResponse(
        success=True,
        session_id=session_id,
        message="Sales Intelligence Deep Agent session started. I can help with deal qualification, RFP responses, solution mapping, competitive analysis, and pricing optimization.",
    )


@app.post("/api/sales-agent/chat", response_model=DeepAgentChatResponse, tags=["Sales Agent"])
async def sales_agent_chat(request: DeepAgentChatRequest) -> DeepAgentChatResponse:
    """Chat with the Sales Intelligence Deep Agent.

    The Deep Agent will:
    1. Qualify deals using BANT/MEDDIC frameworks
    2. Draft RFP/RFI responses using templates
    3. Analyze competitors and develop win strategies
    4. Calculate pricing with margin analysis
    5. Assess win probability and deal risks

    Args:
        request: Chat message and optional session ID.

    Returns:
        Agent response with todos, files, and tool calls.
    """
    if not deep_agent_loaded or sales_intelligence_deep_agent is None:
        return DeepAgentChatResponse(
            success=False,
            error="Sales Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.",
        )

    try:
        result = await sales_intelligence_deep_agent.achat(
            message=request.message,
            session_id=request.session_id,
            user_id=request.user_id,
        )

        return DeepAgentChatResponse(
            success=True,
            response=result.get("response"),
            session_id=result.get("session_id"),
            todos=result.get("todos"),
            files=result.get("files"),
            tool_calls=result.get("tool_calls"),
            iteration_count=result.get("iteration_count"),
        )
    except Exception as e:
        return DeepAgentChatResponse(
            success=False,
            session_id=request.session_id,
            error=str(e),
        )


@app.post("/api/sales-agent/chat/stream", tags=["Sales Agent"])
async def sales_agent_chat_stream(request: DeepAgentChatRequest):
    """Stream chat with the Sales Intelligence Deep Agent.

    Uses Server-Sent Events (SSE) to stream:
    - thinking: Agent's reasoning steps
    - tool_start: When a tool is being called
    - tool_result: Tool execution results
    - todo_update: Todo list changes
    - token: Streaming response tokens
    - complete: Final response with all context

    Args:
        request: Chat message and session ID.

    Returns:
        SSE stream of events.
    """
    import json

    def serialize_event(event: dict) -> str:
        """Serialize event to JSON, handling datetime and other types."""

        def default_serializer(obj):
            if hasattr(obj, "isoformat"):
                return obj.isoformat()
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "__dict__"):
                return str(obj)
            return str(obj)

        return json.dumps(event, default=default_serializer)

    async def event_generator():
        if not deep_agent_loaded or sales_intelligence_deep_agent is None:
            yield f"data: {serialize_event({'type': 'error', 'data': {'error': 'Sales Agent not available'}})}\n\n"
            return

        try:
            print(f"[DEBUG] Starting Sales Agent stream for session: {request.session_id}")
            async for event in sales_intelligence_deep_agent.astream_chat(
                message=request.message,
                session_id=request.session_id,
                user_id=request.user_id,
            ):
                try:
                    yield f"data: {serialize_event(event)}\n\n"
                except Exception as serialize_err:
                    print(f"[ERROR] Failed to serialize event: {serialize_err}")
                    yield f"data: {serialize_event({'type': 'error', 'data': {'error': f'Serialization error: {serialize_err}'}})}\n\n"
            print(f"[DEBUG] Sales Agent stream completed for session: {request.session_id}")
        except GeneratorExit:
            print(f"[DEBUG] Client disconnected from Sales Agent stream: {request.session_id}")
        except Exception as e:
            print(f"[ERROR] Sales Agent stream error for session {request.session_id}: {e}")
            import traceback

            traceback.print_exc()
            yield f"data: {serialize_event({'type': 'error', 'data': {'error': str(e)}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/sales-agent/context/{session_id}", response_model=DeepAgentContextResponse, tags=["Sales Agent"])
async def sales_agent_context(session_id: str) -> DeepAgentContextResponse:
    """Get context for a Sales Agent session.

    Args:
        session_id: Session identifier.

    Returns:
        Session context including todos and files.
    """
    if not deep_agent_loaded or sales_intelligence_deep_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Sales Agent not available.",
        )

    try:
        context = sales_intelligence_deep_agent.get_session_context(session_id)

        return DeepAgentContextResponse(
            success=True,
            session_id=session_id,
            todos=context.get("todos", []),
            files=context.get("files", []),
            metadata=context.get("metadata"),
        )
    except Exception as e:
        return DeepAgentContextResponse(
            success=False,
            session_id=session_id,
            error=str(e),
        )


@app.get("/api/sales-agent/todos/{session_id}", tags=["Sales Agent"])
async def sales_agent_todos(session_id: str) -> dict:
    """Get the todo list for a Sales Agent session.

    Args:
        session_id: The session identifier.

    Returns:
        List of todos with their status.
    """
    if not deep_agent_loaded or sales_intelligence_deep_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Sales Agent not available.",
        )

    context = sales_intelligence_deep_agent.get_session_context(session_id)
    todos = context.get("todos", [])

    summary = {
        "total": len(todos),
        "pending": len([t for t in todos if t.get("status") == "pending"]),
        "in_progress": len([t for t in todos if t.get("status") == "in_progress"]),
        "completed": len([t for t in todos if t.get("status") == "completed"]),
    }

    return {
        "success": True,
        "session_id": session_id,
        "todos": todos,
        "summary": summary,
    }


@app.get("/api/sales-agent/files/{session_id}", tags=["Sales Agent"])
async def sales_agent_files(session_id: str) -> dict:
    """Get the workspace files for a Sales Agent session.

    Args:
        session_id: The session identifier.

    Returns:
        List of file paths in the session workspace.
    """
    if not deep_agent_loaded or sales_intelligence_deep_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Sales Agent not available.",
        )

    context = sales_intelligence_deep_agent.get_session_context(session_id)

    return {
        "success": True,
        "session_id": session_id,
        "files": context.get("files", []),
        "count": len(context.get("files", [])),
    }


@app.get("/api/sales-agent/subagents", tags=["Sales Agent"])
async def sales_agent_subagents() -> dict:
    """Get available subagents for the Sales Intelligence Agent.

    Returns:
        List of available subagents with descriptions.
    """
    if not deep_agent_loaded:
        raise HTTPException(
            status_code=503,
            detail="Sales Agent not available.",
        )

    subagents = [
        {
            "name": "deal-qualifier",
            "description": "Lead qualification using BANT/MEDDIC frameworks",
        },
        {
            "name": "solution-architect",
            "description": "Requirement mapping and solution design by business line",
        },
        {
            "name": "proposal-writer",
            "description": "RFP/RFI response drafting and executive summaries",
        },
        {
            "name": "pricing-analyst",
            "description": "Pricing strategy, margin analysis, and commercial modeling",
        },
        {
            "name": "competitive-strategist",
            "description": "Competitive positioning and objection handling",
        },
    ]

    return {
        "success": True,
        "subagents": subagents,
        "count": len(subagents),
    }


@app.post("/api/sales-agent/upload", response_model=DeepAgentUploadResponse, tags=["Sales Agent"])
async def sales_agent_upload(
    file: UploadFile = File(...),
    session_id: str = Form(...),
) -> DeepAgentUploadResponse:
    """Upload a document to the Sales Intelligence Deep Agent session.

    Supported file types: PDF, TXT, DOCX, DOC, PPTX, PPT, PNG, JPG, JPEG
    Documents are processed, chunked, and indexed for RAG-based search.
    The agent can then use search_attachments tool to query document content.
    Useful for RFP documents, customer requirements, competitive intel, etc.

    Args:
        file: Document file to upload
        session_id: Session ID to associate the document with

    Returns:
        Upload response with document ID and metadata
    """
    if not deep_agent_loaded:
        return DeepAgentUploadResponse(
            success=False,
            error="Sales Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.",
        )

    from app.deepagents.tools.document_tools import process_and_store_document

    # Validate file extension
    allowed_extensions = {".pdf", ".txt", ".docx", ".doc", ".pptx", ".ppt", ".png", ".jpg", ".jpeg"}
    file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""

    if file_ext not in allowed_extensions:
        return DeepAgentUploadResponse(
            success=False,
            filename=file.filename,
            error=f"Unsupported file type: {file_ext}. Supported: {', '.join(allowed_extensions)}",
        )

    try:
        # Read file content
        content = await file.read()

        # Process and store document
        result = process_and_store_document(
            content=content,
            filename=file.filename,
            session_id=session_id,
        )

        return DeepAgentUploadResponse(
            success=True,
            document_id=result["doc_id"],
            filename=result["filename"],
            file_type=result["file_type"],
            chunks_created=result["chunk_count"],
            detected_language=result["language"],
            session_id=session_id,
            message=f"Document '{file.filename}' uploaded successfully with {result['chunk_count']} chunks. Use search_attachments tool to query content.",
        )

    except ValueError as ve:
        return DeepAgentUploadResponse(
            success=False,
            filename=file.filename,
            session_id=session_id,
            error=str(ve),
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return DeepAgentUploadResponse(
            success=False,
            filename=file.filename,
            session_id=session_id,
            error=f"Failed to process document: {e}",
        )


@app.get("/api/sales-agent/attachments/{session_id}", tags=["Sales Agent"])
async def sales_agent_list_attachments(session_id: str) -> dict:
    """List all uploaded documents for a Sales Agent session.

    Args:
        session_id: Session identifier

    Returns:
        List of uploaded documents with metadata
    """
    if not deep_agent_loaded:
        raise HTTPException(
            status_code=503,
            detail="Sales Agent not available.",
        )

    from app.deepagents.tools.document_tools import _current_document, _document_metadata

    docs = _document_metadata.get(session_id, {})
    current_doc_id = _current_document.get(session_id)

    documents = []
    for doc_id, meta in docs.items():
        documents.append(
            {
                **meta,
                "is_current": doc_id == current_doc_id,
            }
        )

    return {
        "success": True,
        "session_id": session_id,
        "documents": documents,
        "count": len(documents),
        "current_document_id": current_doc_id,
    }


@app.delete("/api/sales-agent/attachments/{session_id}", tags=["Sales Agent"])
async def sales_agent_clear_attachments(
    session_id: str,
    document_ids: str | None = None,
) -> dict:
    """Clear uploaded documents from a Sales Agent session.

    Args:
        session_id: Session identifier
        document_ids: Comma-separated document IDs to clear, or None for all

    Returns:
        Confirmation of cleared documents
    """
    if not deep_agent_loaded:
        raise HTTPException(
            status_code=503,
            detail="Sales Agent not available.",
        )

    from app.deepagents.tools.document_tools import clear_attachments

    doc_ids = document_ids.split(",") if document_ids else None
    result = clear_attachments.invoke({"session_id": session_id, "doc_ids": doc_ids})

    return {
        "success": True,
        "session_id": session_id,
        "message": result,
    }


# ============================================================================
# Recruitment Deep Agent Endpoints
# ============================================================================


@app.post("/api/recruitment-agent/start", response_model=DeepAgentStartResponse, tags=["Recruitment Agent"])
async def recruitment_agent_start(request: DeepAgentStartRequest) -> DeepAgentStartResponse:
    """Start a new Recruitment Deep Agent session.

    The Recruitment Deep Agent assists with SharePoint document management,
    resume screening (L1/L2/L3), interview question generation, candidate
    evaluation, and scoring reports.

    Returns:
        Session ID and welcome message.
    """
    if not deep_agent_loaded or recruitment_deep_agent is None:
        return DeepAgentStartResponse(
            success=False,
            error="Recruitment Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.",
        )

    import uuid

    session_id = str(uuid.uuid4())

    return DeepAgentStartResponse(
        success=True,
        session_id=session_id,
        message="Recruitment Deep Agent session started. I can help with SharePoint document management, resume screening at L1/L2/L3 levels, interview question generation, candidate evaluation, and scoring reports.",
    )


@app.post("/api/recruitment-agent/chat", response_model=DeepAgentChatResponse, tags=["Recruitment Agent"])
async def recruitment_agent_chat(request: DeepAgentChatRequest) -> DeepAgentChatResponse:
    """Chat with the Recruitment Deep Agent.

    The Deep Agent will:
    1. Manage SharePoint documents (JDs, resumes, questions)
    2. Screen resumes at L1, L2, L3 levels
    3. Generate interview questions based on skillsets
    4. Evaluate candidate answers
    5. Generate scoring reports and shortlists

    Args:
        request: Chat message and optional session ID.

    Returns:
        Agent response with todos, files, and tool calls.
    """
    if not deep_agent_loaded or recruitment_deep_agent is None:
        return DeepAgentChatResponse(
            success=False,
            error="Recruitment Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.",
        )

    try:
        result = await recruitment_deep_agent.achat(
            message=request.message,
            session_id=request.session_id,
            user_id=request.user_id,
        )

        return DeepAgentChatResponse(
            success=True,
            response=result.get("response"),
            session_id=result.get("session_id"),
            todos=result.get("todos"),
            files=result.get("files"),
            tool_calls=result.get("tool_calls"),
            iteration_count=result.get("iteration_count"),
        )
    except Exception as e:
        return DeepAgentChatResponse(
            success=False,
            session_id=request.session_id,
            error=str(e),
        )


@app.post("/api/recruitment-agent/chat/stream", tags=["Recruitment Agent"])
async def recruitment_agent_chat_stream(request: DeepAgentChatRequest):
    """Stream chat with the Recruitment Deep Agent.

    Uses Server-Sent Events (SSE) to stream:
    - thinking: Agent's reasoning steps
    - tool_start: When a tool is being called
    - tool_result: Tool execution results
    - todo_update: Todo list changes
    - token: Streaming response tokens
    - complete: Final response with all context

    Args:
        request: Chat message and session ID.

    Returns:
        SSE stream of events.
    """
    import json

    def serialize_event(event: dict) -> str:
        """Serialize event to JSON, handling datetime and other types."""

        def default_serializer(obj):
            if hasattr(obj, "isoformat"):
                return obj.isoformat()
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "__dict__"):
                return str(obj)
            return str(obj)

        return json.dumps(event, default=default_serializer)

    async def event_generator():
        if not deep_agent_loaded or recruitment_deep_agent is None:
            yield f"data: {serialize_event({'type': 'error', 'data': {'error': 'Recruitment Agent not available'}})}\n\n"
            return

        try:
            print(f"[DEBUG] Starting Recruitment Agent stream for session: {request.session_id}")
            async for event in recruitment_deep_agent.astream_chat(
                message=request.message,
                session_id=request.session_id,
                user_id=request.user_id,
            ):
                try:
                    yield f"data: {serialize_event(event)}\n\n"
                except Exception as serialize_err:
                    print(f"[ERROR] Failed to serialize event: {serialize_err}")
                    yield f"data: {serialize_event({'type': 'error', 'data': {'error': f'Serialization error: {serialize_err}'}})}\n\n"
            print(f"[DEBUG] Recruitment Agent stream completed for session: {request.session_id}")
        except GeneratorExit:
            print(f"[DEBUG] Client disconnected from Recruitment Agent stream: {request.session_id}")
        except Exception as e:
            print(f"[ERROR] Recruitment Agent stream error for session {request.session_id}: {e}")
            import traceback

            traceback.print_exc()
            yield f"data: {serialize_event({'type': 'error', 'data': {'error': str(e)}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get(
    "/api/recruitment-agent/context/{session_id}", response_model=DeepAgentContextResponse, tags=["Recruitment Agent"]
)
async def recruitment_agent_context(session_id: str) -> DeepAgentContextResponse:
    """Get context for a Recruitment Agent session.

    Args:
        session_id: Session identifier.

    Returns:
        Session context including todos and files.
    """
    if not deep_agent_loaded or recruitment_deep_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Recruitment Agent not available.",
        )

    try:
        context = recruitment_deep_agent.get_session_context(session_id)

        return DeepAgentContextResponse(
            success=True,
            session_id=session_id,
            todos=context.get("todos", []),
            files=context.get("files", []),
            metadata=context.get("metadata"),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get context: {e}",
        )


@app.get("/api/recruitment-agent/todos/{session_id}", tags=["Recruitment Agent"])
async def recruitment_agent_todos(session_id: str) -> dict:
    """Get todos for a Recruitment Agent session.

    Args:
        session_id: Session identifier.

    Returns:
        List of todos with status.
    """
    if not deep_agent_loaded or recruitment_deep_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Recruitment Agent not available.",
        )

    context = recruitment_deep_agent.get_session_context(session_id)

    return {
        "success": True,
        "session_id": session_id,
        "todos": context.get("todos", []),
    }


@app.get("/api/recruitment-agent/files/{session_id}", tags=["Recruitment Agent"])
async def recruitment_agent_files(session_id: str) -> dict:
    """Get files for a Recruitment Agent session.

    Args:
        session_id: Session identifier.

    Returns:
        List of files created during session.
    """
    if not deep_agent_loaded or recruitment_deep_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Recruitment Agent not available.",
        )

    context = recruitment_deep_agent.get_session_context(session_id)

    return {
        "success": True,
        "session_id": session_id,
        "files": context.get("files", []),
    }


@app.get("/api/recruitment-agent/subagents", tags=["Recruitment Agent"])
async def recruitment_agent_subagents() -> dict:
    """Get available subagents for the Recruitment Deep Agent.

    Returns:
        List of subagent definitions with their descriptions and tools.
    """
    subagents = [
        {
            "name": "document-manager",
            "description": "Specialized in SharePoint document management - listing, downloading, uploading, and organizing recruitment documents.",
            "tools": [
                "list_sharepoint_folder",
                "download_sharepoint_document",
                "upload_to_sharepoint",
                "search_sharepoint_documents",
                "get_cached_document",
                "create_sharepoint_folder",
            ],
        },
        {
            "name": "resume-screener",
            "description": "Specialized in resume parsing and candidate screening - extracting skills, experience, and matching candidates to job requirements.",
            "tools": [
                "parse_resume",
                "parse_job_description",
                "screen_candidate",
                "batch_screen_resumes",
                "get_candidate_profile",
                "list_candidates",
                "list_job_descriptions",
                "get_shortlisted_candidates",
            ],
        },
        {
            "name": "question-generator",
            "description": "Specialized in creating technical interview questions based on candidate skills and level.",
            "tools": [
                "generate_interview_questions",
                "export_question_set",
                "list_question_sets",
                "get_candidate_profile",
                "list_candidates",
            ],
        },
        {
            "name": "answer-evaluator",
            "description": "Specialized in evaluating candidate answers and generating scores.",
            "tools": [
                "submit_candidate_answers",
                "evaluate_candidate_answers",
                "get_candidate_score",
                "list_question_sets",
                "get_candidate_profile",
            ],
        },
        {
            "name": "report-generator",
            "description": "Specialized in generating recruitment reports, Excel exports, and shortlists.",
            "tools": [
                "generate_scoring_report",
                "export_scoring_excel",
                "get_ranking_summary",
                "generate_shortlist_report",
                "get_passing_score_thresholds",
                "get_shortlisted_candidates",
            ],
        },
    ]

    return {
        "success": True,
        "subagents": subagents,
        "count": len(subagents),
    }


@app.post("/api/recruitment-agent/upload", response_model=DeepAgentUploadResponse, tags=["Recruitment Agent"])
async def recruitment_agent_upload(
    file: UploadFile = File(...),
    session_id: str = Form(...),
) -> DeepAgentUploadResponse:
    """Upload a document to the Recruitment Deep Agent session.

    Supported file types: PDF, TXT, DOCX, DOC, PPTX, PPT, PNG, JPG, JPEG
    Documents are processed, chunked, and indexed for RAG-based search.
    The agent can then use search_attachments tool to query document content.
    Useful for JDs, resumes, question sets, and answer files.

    Args:
        file: Document file to upload
        session_id: Session ID to associate the document with

    Returns:
        Upload response with document ID and metadata
    """
    if not deep_agent_loaded:
        return DeepAgentUploadResponse(
            success=False,
            error="Recruitment Agent not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.",
        )

    from app.deepagents.tools.document_tools import process_and_store_document

    # Validate file extension
    allowed_extensions = {".pdf", ".txt", ".docx", ".doc", ".pptx", ".ppt", ".png", ".jpg", ".jpeg"}
    file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""

    if file_ext not in allowed_extensions:
        return DeepAgentUploadResponse(
            success=False,
            filename=file.filename,
            error=f"Unsupported file type: {file_ext}. Supported: {', '.join(allowed_extensions)}",
        )

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    if file.size and file.size > MAX_FILE_SIZE:
        return DeepAgentUploadResponse(
            success=False,
            filename=file.filename,
            error="File too large. Maximum size: 10MB",
        )
    try:
        # Read file content
        content = await file.read()

        # Process and store document
        result = process_and_store_document(
            content=content,
            filename=file.filename,
            session_id=session_id,
        )

        return DeepAgentUploadResponse(
            success=True,
            document_id=result["doc_id"],
            filename=result["filename"],
            file_type=result["file_type"],
            chunks_created=result["chunk_count"],
            detected_language=result["language"],
            session_id=session_id,
            message=f"Document '{file.filename}' uploaded successfully with {result['chunk_count']} chunks. Use search_attachments tool to query content.",
        )

    except ValueError as ve:
        return DeepAgentUploadResponse(
            success=False,
            filename=file.filename,
            session_id=session_id,
            error=str(ve),
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return DeepAgentUploadResponse(
            success=False,
            filename=file.filename,
            session_id=session_id,
            error=f"Failed to process document: {e}",
        )


@app.get("/api/recruitment-agent/attachments/{session_id}", tags=["Recruitment Agent"])
async def recruitment_agent_list_attachments(session_id: str) -> dict:
    """List all uploaded documents for a Recruitment Agent session.

    Args:
        session_id: Session identifier

    Returns:
        List of uploaded documents with metadata
    """
    if not deep_agent_loaded:
        raise HTTPException(
            status_code=503,
            detail="Recruitment Agent not available.",
        )

    from app.deepagents.tools.document_tools import _current_document, _document_metadata

    docs = _document_metadata.get(session_id, {})
    current_doc_id = _current_document.get(session_id)

    documents = []
    for doc_id, meta in docs.items():
        documents.append(
            {
                **meta,
                "is_current": doc_id == current_doc_id,
            }
        )

    return {
        "success": True,
        "session_id": session_id,
        "documents": documents,
        "count": len(documents),
        "current_document_id": current_doc_id,
    }


@app.delete("/api/recruitment-agent/attachments/{session_id}", tags=["Recruitment Agent"])
async def recruitment_agent_clear_attachments(
    session_id: str,
    document_ids: str | None = None,
) -> dict:
    """Clear uploaded documents from a Recruitment Agent session.

    Args:
        session_id: Session identifier
        document_ids: Comma-separated document IDs to clear, or None for all

    Returns:
        Confirmation of cleared documents
    """
    if not deep_agent_loaded:
        raise HTTPException(
            status_code=503,
            detail="Recruitment Agent not available.",
        )

    from app.deepagents.tools.document_tools import clear_attachments

    doc_ids = document_ids.split(",") if document_ids else None
    result = clear_attachments.invoke({"session_id": session_id, "doc_ids": doc_ids})

    return {
        "success": True,
        "session_id": session_id,
        "message": result,
    }


@app.get("/api/recruitment-agent/config", tags=["Recruitment Agent"])
async def recruitment_agent_config() -> dict:
    """Get recruitment agent configuration including passing scores and thresholds.

    Returns:
        Configuration including passing scores for L1, L2, L3 levels and other parameters.
    """
    from app.deepagents.config.recruitment_config import get_recruitment_config

    config = get_recruitment_config()

    return {
        "success": True,
        "config": {
            "scoring": {
                "l1_passing_score": config.scoring.l1_passing_score,
                "l2_passing_score": config.scoring.l2_passing_score,
                "l3_passing_score": config.scoring.l3_passing_score,
                "technical_weight": config.scoring.technical_weight,
                "experience_weight": config.scoring.experience_weight,
                "education_weight": config.scoring.education_weight,
                "soft_skills_weight": config.scoring.soft_skills_weight,
                "certification_weight": config.scoring.certification_weight,
            },
            "interview": {
                "l1_question_count": config.interview.l1_question_count,
                "l2_question_count": config.interview.l2_question_count,
                "l3_question_count": config.interview.l3_question_count,
                "mcq_percentage": config.interview.mcq_percentage,
                "coding_percentage": config.interview.coding_percentage,
            },
            "resume_parsing": {
                "l2_min_experience": config.resume_parsing.l2_min_experience,
                "l3_min_experience": config.resume_parsing.l3_min_experience,
                "supported_formats": config.resume_parsing.supported_formats,
            },
            "sharepoint": {
                "folder_structure": {
                    "jd": config.sharepoint.jd_folder,
                    "resumes": config.sharepoint.resumes_folder,
                    "questions": config.sharepoint.interview_questions_folder,
                    "scoring": config.sharepoint.scoring_folder,
                    "shortlist": config.sharepoint.shortlist_folder,
                },
            },
        },
    }


@app.get("/api/recruitment-agent/dashboard/{session_id}", tags=["Recruitment Agent"])
async def recruitment_agent_dashboard(session_id: str) -> dict:
    """Get comprehensive session dashboard with progress and next steps.

    Args:
        session_id: Session identifier.

    Returns:
        Dashboard data including phase, progress, and recommendations.
    """
    if not deep_agent_loaded or recruitment_deep_agent is None:
        raise HTTPException(status_code=503, detail="Recruitment agent not loaded")

    from app.deepagents.tools.recruitment_tools import get_session_dashboard

    result = get_session_dashboard.invoke({"session_id": session_id})

    return {
        "success": True,
        "session_id": session_id,
        "dashboard": result,
    }


@app.delete("/api/recruitment-agent/session/{session_id}", tags=["Recruitment Agent"])
async def recruitment_agent_clear_session(session_id: str) -> dict:
    """Clear all session data for PII compliance.

    Removes all candidate profiles, JDs, screening results,
    interview data, scores, and document caches.

    Args:
        session_id: Session identifier.

    Returns:
        Confirmation of cleared data.
    """
    if not deep_agent_loaded or recruitment_deep_agent is None:
        raise HTTPException(status_code=503, detail="Recruitment agent not loaded")

    from app.deepagents.tools.recruitment_tools import clear_session_data

    result = clear_session_data.invoke({"session_id": session_id})

    return {
        "success": True,
        "session_id": session_id,
        "result": result,
    }


# ============================================================================
# Chat UI (Static Files)
# ============================================================================

# Get the static directory path
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/chat", response_class=HTMLResponse)
async def chat_ui() -> HTMLResponse:
    """Serve the chat UI at /chat.

    Returns the HTML with no-cache headers and dynamic timestamp to ensure
    the latest version is always served and verifiable.
    """
    chat_file = STATIC_DIR / "chat.html"
    if chat_file.exists():
        # Read file fresh every time (no caching)
        html_content = chat_file.read_text(encoding="utf-8")

        # Inject server timestamp for cache verification (invisible to user)
        server_timestamp = f"<!-- Server-rendered: {time.time()} -->\n"
        html_content = server_timestamp + html_content

        return HTMLResponse(
            content=html_content,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
                "X-Content-Type-Options": "nosniff",
                "ETag": f'"{int(time.time())}"',
            },
        )

    return HTMLResponse(
        content="<h1>Chat UI not found</h1><p>Please ensure app/static/chat.html exists.</p>",
        status_code=404,
    )


@app.get("/chatui")
async def chatui_redirect() -> RedirectResponse:
    """Redirect legacy /chatui to /chat for backwards compatibility."""
    return RedirectResponse(url="/chat", status_code=301)


@app.get("/analytics", response_class=HTMLResponse)
async def analytics_ui() -> HTMLResponse:
    """Serve the real-time analytics dashboard at /analytics."""
    analytics_file = STATIC_DIR / "analytics.html"
    if analytics_file.exists():
        return HTMLResponse(
            content=analytics_file.read_text(encoding="utf-8"),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return HTMLResponse(
        content="<h1>Analytics UI not found</h1><p>Please ensure app/static/analytics.html exists.</p>",
        status_code=404,
    )


@app.get("/software-dev-chat", response_class=HTMLResponse)
async def software_dev_chat_ui() -> HTMLResponse:
    """Serve the Software Development Deep Agent chat UI.

    Returns the HTML with no-cache headers for the SDLC automation interface.
    """
    chat_file = STATIC_DIR / "software_dev_chat.html"
    if chat_file.exists():
        html_content = chat_file.read_text(encoding="utf-8")

        # Inject server timestamp for cache verification
        server_timestamp = f"<!-- Server-rendered: {time.time()} -->\n"
        html_content = server_timestamp + html_content

        return HTMLResponse(
            content=html_content,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
                "X-Content-Type-Options": "nosniff",
                "ETag": f'"{int(time.time())}"',
            },
        )

    return HTMLResponse(
        content="<h1>Software Development Chat UI not found</h1><p>Please ensure app/static/software_dev_chat.html exists.</p>",
        status_code=404,
    )


# Mount static files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============================================================================
# LangServe Routes Setup
# ============================================================================


def setup_langchain_routes() -> None:
    """Set up LangServe routes for LangChain chains.

    API routes are prefixed with /api/langserve/ for clean separation:
    - /api/langserve/chat/invoke, /stream, /batch
    - /api/langserve/rag/invoke, /stream, /batch
    - /api/langserve/agent/invoke, /stream
    """
    if not chains_loaded:
        return

    # Chat chain endpoint
    add_routes(
        app,
        chat_chain,
        path="/api/langserve/chat",
        enabled_endpoints=["invoke", "stream", "batch"],
    )

    # RAG chain endpoint
    add_routes(
        app,
        rag_chain,
        path="/api/langserve/rag",
        enabled_endpoints=["invoke", "stream", "batch"],
    )

    # Agent endpoint
    add_routes(
        app,
        agent_executor,
        path="/api/langserve/agent",
        enabled_endpoints=["invoke", "stream"],
    )


# NOTE: Agent loading is handled in the lifespan handler, not at module import time.
# This prevents blocking during module import when using --reload mode.
# LangServe routes are added dynamically after chains are loaded in lifespan.


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.server:app",
        host="0.0.0.0",  # noqa: S104
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
