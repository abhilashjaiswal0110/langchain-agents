"""LangChain Platform API Server.

This FastAPI application serves multiple LangChain chains and LangGraph agents
as REST API endpoints using LangServe with LangSmith tracing enabled.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from langserve import add_routes
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# LangSmith Tracing Configuration
# ============================================================================

def setup_langsmith_tracing() -> bool:
    """Configure LangSmith tracing if enabled.

    Returns:
        True if tracing is enabled, False otherwise.
    """
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")

    if tracing_enabled and langsmith_api_key:
        # Ensure all required env vars are set
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
        os.environ.setdefault("LANGCHAIN_PROJECT", "langchain-platform")
        os.environ.setdefault(
            "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
        )
        print(f"LangSmith tracing enabled for project: {os.getenv('LANGCHAIN_PROJECT')}")
        return True

    if tracing_enabled and not langsmith_api_key:
        print("Warning: LANGCHAIN_TRACING_V2=true but LANGCHAIN_API_KEY not set")
        print("Tracing will not work. Get your API key from https://smith.langchain.com")

    return False


# Initialize tracing early
tracing_enabled = setup_langsmith_tracing()

# ============================================================================
# Chain Loading
# ============================================================================

chains_loaded = False
langgraph_loaded = False
chat_chain = None
rag_chain = None
agent_executor = None
langgraph_agent = None


def load_chains() -> bool:
    """Load LangChain chains if API key is available.

    Returns:
        True if chains loaded successfully, False otherwise.
    """
    global chains_loaded, chat_chain, rag_chain, agent_executor

    if not os.getenv("OPENAI_API_KEY"):
        return False

    try:
        from app.chains.chat import chat_chain as _chat_chain
        from app.chains.rag import rag_chain as _rag_chain
        from app.chains.agent import agent_executor as _agent_executor

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

    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    if not (has_openai or has_anthropic):
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
        print(f"[OK] LangSmith tracing enabled")
        print(f"     Project: {os.getenv('LANGCHAIN_PROJECT')}")
        print(f"     Endpoint: {os.getenv('LANGCHAIN_ENDPOINT')}")
    else:
        print("[--] LangSmith tracing disabled")

    # Load chains
    if load_chains():
        print("[OK] LangChain chains loaded (OpenAI)")
    else:
        print("[--] LangChain chains not loaded (OPENAI_API_KEY not set)")

    # Load LangGraph agent
    if load_langgraph_agent():
        provider = "Anthropic" if os.getenv("ANTHROPIC_API_KEY") else "OpenAI"
        print(f"[OK] LangGraph agent loaded ({provider})")
    else:
        print("[--] LangGraph agent not loaded (no API keys set)")

    print("=" * 60)
    print("Platform ready!")
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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Response Models
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    version: str
    chains_loaded: bool
    langgraph_loaded: bool
    tracing_enabled: bool
    langsmith_project: str | None


class LangGraphRequest(BaseModel):
    """LangGraph agent request model."""

    input: str


class LangGraphResponse(BaseModel):
    """LangGraph agent response model."""

    output: str


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
# LangServe Routes Setup
# ============================================================================

def setup_langchain_routes() -> None:
    """Set up LangServe routes for LangChain chains."""
    if not chains_loaded:
        return

    # Chat chain endpoint
    add_routes(
        app,
        chat_chain,
        path="/chat",
        enabled_endpoints=["invoke", "stream", "batch"],
    )

    # RAG chain endpoint
    add_routes(
        app,
        rag_chain,
        path="/rag",
        enabled_endpoints=["invoke", "stream", "batch"],
    )

    # Agent endpoint
    add_routes(
        app,
        agent_executor,
        path="/agent",
        enabled_endpoints=["invoke", "stream"],
    )


# Initialize routes at import time
if os.getenv("OPENAI_API_KEY"):
    load_chains()
    setup_langchain_routes()

if os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
    load_langgraph_agent()


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
