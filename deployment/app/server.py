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
enterprise_agents_loaded = False
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


def load_enterprise_agents() -> dict[str, bool]:
    """Load enterprise IT agents.

    Returns:
        Dictionary with load status for each agent.
    """
    global enterprise_agents_loaded
    global research_agent, content_agent, data_analyst_agent
    global document_agent, multilingual_rag_agent, hitl_support_agent
    global code_assistant_agent

    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    if not (has_openai or has_anthropic):
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

    # Content Agent
    try:
        from app.agents.content import ContentAgent
        content_agent = ContentAgent()
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

    enterprise_agents_loaded = any(status.values())
    status["loaded"] = enterprise_agents_loaded
    return status


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

    # Load Enterprise Agents
    enterprise_status = load_enterprise_agents()
    if enterprise_status.get("loaded"):
        loaded_agents = [k for k, v in enterprise_status.items() if v is True and k != "loaded"]
        print(f"[OK] Enterprise agents loaded: {', '.join(loaded_agents)}")
    else:
        print("[--] Enterprise agents not loaded (no API keys set)")

    print("=" * 60)
    print("Platform ready!")
    print("  - API Docs: http://localhost:8000/docs")
    print("  - Chat UI:  http://localhost:8000/chat")
    print("  - Enterprise Agents: http://localhost:8000/docs#/Enterprise%20Agents")
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
    enterprise_agents_loaded: bool
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
# 3rd Party Platform Webhook Endpoints
# ============================================================================

async def _invoke_enterprise_agent(agent_type: str, query: str) -> tuple[bool, str | None]:
    """Helper to invoke an enterprise agent by type.

    Returns:
        Tuple of (success, response_or_error)
    """
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
        result = await agent.ainvoke({"query": query})
        response = result.get("response", result.get("output", str(result)))
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
    success, response = await _invoke_enterprise_agent(request.agent_type, request.query)

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
    success, response = await _invoke_enterprise_agent(request.agent_type, request.query)

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
    success, response = await _invoke_enterprise_agent(request.agent_type, request.query)

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
        },
    }


@app.post("/api/enterprise/research/invoke", response_model=EnterpriseAgentResponse, tags=["Enterprise Agents"])
async def research_agent_invoke(request: ResearchAgentRequest) -> EnterpriseAgentResponse:
    """Invoke the Research Agent for web search and information synthesis.

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

    try:
        result = research_agent.research(
            query=request.query,
            session_id=request.session_id,
        )
        return EnterpriseAgentResponse(
            success=True,
            response=result.get("output", ""),
            session_id=result.get("session_id"),
            agent_type="research",
            tool_calls=result.get("tool_calls"),
        )
    except Exception as e:
        return EnterpriseAgentResponse(
            success=False,
            error=str(e),
            agent_type="research",
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
        result = content_agent.generate(
            topic=request.topic,
            platform=request.platform,
            tone=request.tone,
            audience=request.audience,
            session_id=request.session_id,
        )
        return EnterpriseAgentResponse(
            success=True,
            response=result.get("output", ""),
            session_id=result.get("session_id"),
            agent_type="content",
            tool_calls=result.get("tool_calls"),
        )
    except Exception as e:
        return EnterpriseAgentResponse(
            success=False,
            error=str(e),
            agent_type="content",
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
        result = data_analyst_agent.invoke(
            message=request.message,
            session_id=request.session_id,
        )
        return EnterpriseAgentResponse(
            success=True,
            response=result.get("output", ""),
            session_id=result.get("session_id"),
            agent_type="data_analyst",
            tool_calls=result.get("tool_calls"),
        )
    except Exception as e:
        return EnterpriseAgentResponse(
            success=False,
            error=str(e),
            agent_type="data_analyst",
        )


@app.post("/api/enterprise/data-analyst/upload", tags=["Enterprise Agents"])
async def data_analyst_upload(file: UploadFile = File(...)) -> dict:
    """Upload a file for data analysis.

    Supports Excel (.xlsx, .xls) and CSV files.

    Args:
        file: The data file to upload.

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

    return {
        "status": "success",
        "filename": filename,
        "path": tmp_path,
        "message": f"File uploaded. Use path '{tmp_path}' in your analysis requests.",
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
        result = document_agent.generate(
            doc_type=request.doc_type,
            title=request.title,
            description=request.description,
            sections=request.sections,
            session_id=request.session_id,
        )
        return EnterpriseAgentResponse(
            success=True,
            response=result.get("output", ""),
            session_id=result.get("session_id"),
            agent_type="document",
            tool_calls=result.get("tool_calls"),
        )
    except Exception as e:
        return EnterpriseAgentResponse(
            success=False,
            error=str(e),
            agent_type="document",
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
            language=request.language,
            session_id=request.session_id,
        )
        return EnterpriseAgentResponse(
            success=True,
            response=result.get("output", ""),
            session_id=result.get("session_id"),
            agent_type="multilingual_rag",
            tool_calls=result.get("tool_calls"),
        )
    except Exception as e:
        return EnterpriseAgentResponse(
            success=False,
            error=str(e),
            agent_type="multilingual_rag",
        )


@app.post("/api/enterprise/rag/upload", tags=["Enterprise Agents"])
async def rag_upload_document(file: UploadFile = File(...)) -> dict:
    """Upload a document for RAG processing.

    Supports PDF, Word, and text files in multiple languages.

    Args:
        file: The document to upload.

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

    content = await file.read()

    try:
        result = multilingual_rag_agent.upload_document(content, filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        return EnterpriseAgentResponse(
            success=True,
            response=result.get("output", ""),
            session_id=result.get("session_id"),
            agent_type="hitl_support",
            tool_calls=result.get("tool_calls"),
        )
    except Exception as e:
        return EnterpriseAgentResponse(
            success=False,
            error=str(e),
            agent_type="hitl_support",
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

        return EnterpriseAgentResponse(
            success=True,
            response=result.get("output", ""),
            session_id=result.get("session_id"),
            agent_type="code_assistant",
            tool_calls=result.get("tool_calls"),
        )
    except Exception as e:
        return EnterpriseAgentResponse(
            success=False,
            error=str(e),
            agent_type="code_assistant",
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
    load_enterprise_agents()


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
