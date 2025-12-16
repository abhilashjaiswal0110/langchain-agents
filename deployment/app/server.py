"""LangChain Platform API Server.

This FastAPI application serves multiple LangChain chains and LangGraph agents
as REST API endpoints using LangServe with LangSmith tracing enabled.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
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
doc_rag_loaded = False
it_support_loaded = False
chat_chain = None
rag_chain = None
agent_executor = None
langgraph_agent = None
doc_rag_chain = None
conversation_manager = None


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


def load_doc_rag() -> bool:
    """Load Document RAG chain if OpenAI API key is available.

    Returns:
        True if Document RAG chain loaded successfully, False otherwise.
    """
    global doc_rag_loaded, doc_rag_chain

    if not os.getenv("OPENAI_API_KEY"):
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

    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    if not (has_openai or has_anthropic):
        return False

    try:
        from app.agents.conversation_manager import ConversationManager
        conversation_manager = ConversationManager()
        it_support_loaded = True
        return True
    except Exception as e:
        print(f"Failed to load IT Support agents: {e}")
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

    print("=" * 60)
    print("Platform ready!")
    print("  - API Docs: http://localhost:8000/docs")
    print("  - Chat UI:  http://localhost:8000/chat")
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
    doc_rag_loaded: bool
    it_support_loaded: bool
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

    agent_type: Literal["it_helpdesk", "servicenow"]
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

@app.post("/api/conversation/start", response_model=ConversationStartResponse)
async def conversation_start(request: ConversationStartRequest) -> ConversationStartResponse:
    """Start a new conversation with an IT Support agent.

    Args:
        request: The conversation start request with agent type.

    Returns:
        Session ID, welcome message, and available commands.
    """
    if not it_support_loaded or conversation_manager is None:
        raise HTTPException(
            status_code=503,
            detail="IT Support agents not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.",
        )

    result = conversation_manager.start_conversation(
        agent_type=request.agent_type,
        user_id=request.user_id,
        metadata=request.metadata,
    )

    return ConversationStartResponse(**result)


@app.post("/api/conversation/chat", response_model=ConversationChatResponse)
async def conversation_chat(request: ConversationChatRequest) -> ConversationChatResponse:
    """Send a message in an existing conversation.

    Args:
        request: The chat request with session ID and message.

    Returns:
        Agent's response and metadata.
    """
    if not it_support_loaded or conversation_manager is None:
        raise HTTPException(
            status_code=503,
            detail="IT Support agents not available.",
        )

    result = await conversation_manager.achat(
        session_id=request.session_id,
        message=request.message,
    )

    return ConversationChatResponse(**result)


@app.get("/api/conversation/{session_id}")
async def conversation_info(session_id: str) -> dict:
    """Get information about a conversation session.

    Args:
        session_id: The session ID to query.

    Returns:
        Session information.
    """
    if not it_support_loaded or conversation_manager is None:
        raise HTTPException(status_code=503, detail="IT Support agents not available.")

    info = conversation_manager.get_session_info(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found.")

    return info


@app.delete("/api/conversation/{session_id}")
async def conversation_end(session_id: str) -> dict:
    """End a conversation session.

    Args:
        session_id: The session ID to end.

    Returns:
        Session summary.
    """
    if not it_support_loaded or conversation_manager is None:
        raise HTTPException(status_code=503, detail="IT Support agents not available.")

    return conversation_manager.end_conversation(session_id)


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
# Chat UI (Static Files)
# ============================================================================

# Get the static directory path
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/chat", response_class=HTMLResponse)
async def chat_ui() -> HTMLResponse:
    """Serve the chat UI for demos."""
    chat_file = STATIC_DIR / "chat.html"
    if chat_file.exists():
        return HTMLResponse(content=chat_file.read_text(encoding="utf-8"))

    return HTMLResponse(
        content="<h1>Chat UI not found</h1><p>Please ensure app/static/chat.html exists.</p>",
        status_code=404,
    )


# Mount static files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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
    load_doc_rag()

if os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
    load_langgraph_agent()
    load_it_support_agents()


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
