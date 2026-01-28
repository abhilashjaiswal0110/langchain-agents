"""REST API Routes for Software Development DeepAgent.

This module provides FastAPI routes for:
- Starting software development sessions
- Chat interactions with the agent
- Streaming responses with thinking steps
- Session management
- Phase transitions
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json

from app.deepagents.software_dev.software_dev_agent import (
    SoftwareDevDeepAgent,
    create_software_dev_agent,
)
from app.deepagents.config.software_dev_config import SDLCPhase

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/software-dev-agent", tags=["Software Development Agent"])

# Global agent instance (singleton)
_agent_instance: SoftwareDevDeepAgent | None = None


def get_agent() -> SoftwareDevDeepAgent:
    """Get or create the agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = create_software_dev_agent()
        logger.info("Software Development DeepAgent initialized")
    return _agent_instance


# =============================================================================
# Request/Response Models
# =============================================================================


class StartSessionRequest(BaseModel):
    """Request to start a new development session."""

    user_id: str | None = Field(default=None, description="User identifier")
    project_name: str | None = Field(default=None, description="Project name")
    initial_context: str | None = Field(default=None, description="Initial context or requirements")


class StartSessionResponse(BaseModel):
    """Response from starting a session."""

    session_id: str
    status: str = "active"
    message: str
    project_name: str | None = None
    available_phases: list[str]
    available_subagents: list[str]


class ChatRequest(BaseModel):
    """Request to chat with the agent."""

    session_id: str = Field(..., description="Session identifier")
    message: str = Field(..., description="User message")


class ChatResponse(BaseModel):
    """Response from chat."""

    session_id: str
    response: str
    phase: str
    todos: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    thinking_steps: list[str] = Field(default_factory=list)


class SessionStateResponse(BaseModel):
    """Response with session state."""

    session_id: str
    phase: str
    todos: list[dict[str, Any]] = Field(default_factory=list)
    requirements_count: int = 0
    code_files_count: int = 0
    test_cases_count: int = 0
    security_issues_count: int = 0


class PhaseTransitionRequest(BaseModel):
    """Request to transition to a new phase."""

    session_id: str
    new_phase: str


class SubagentInfo(BaseModel):
    """Information about a subagent."""

    name: str
    description: str
    tools_count: int


class ConfigResponse(BaseModel):
    """Configuration response."""

    model: str
    max_iterations: int
    recursion_limit: int
    supported_languages: list[str]
    quality_gates: dict[str, Any]


# =============================================================================
# Routes
# =============================================================================


@router.post("/start", response_model=StartSessionResponse)
async def start_session(request: StartSessionRequest) -> StartSessionResponse:
    """Start a new software development session.

    Creates a new session with the Software Development DeepAgent
    and returns session information.
    """
    agent = get_agent()

    # Generate initial message if context provided
    initial_message = request.initial_context or "Hello! I'm ready to help with software development."

    try:
        result = agent.chat(
            message=initial_message,
            user_id=request.user_id,
            project_name=request.project_name,
        )

        return StartSessionResponse(
            session_id=result["session_id"],
            status="active",
            message=result["response"],
            project_name=request.project_name,
            available_phases=[phase.value for phase in SDLCPhase],
            available_subagents=[
                "requirements-intelligence",
                "architecture-design",
                "code-generator",
                "code-reviewer",
                "testing-automation",
                "debugging-optimization",
                "security-compliance",
                "devops-integration",
                "documentation",
            ],
        )
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Send a message to the agent.

    Processes the message through the Software Development DeepAgent
    and returns the response with metadata.
    """
    agent = get_agent()

    try:
        result = agent.chat(
            message=request.message,
            session_id=request.session_id,
        )

        return ChatResponse(
            session_id=result["session_id"],
            response=result["response"],
            phase=result["phase"],
            todos=result.get("todos", []),
            metrics={
                "requirements_count": result.get("requirements_count", 0),
                "code_files_count": result.get("code_files_count", 0),
                "test_coverage": result.get("test_coverage", 0.0),
                "security_issues_count": result.get("security_issues_count", 0),
                "iteration_count": result.get("iteration_count", 0),
            },
            tool_calls=result.get("tool_calls", []),
            thinking_steps=result.get("thinking_steps", []),
        )
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat responses with thinking steps.

    Returns Server-Sent Events (SSE) with real-time updates
    including thinking steps, tool calls, and final response.
    """
    agent = get_agent()

    async def event_generator():
        try:
            async for event in agent.astream(
                message=request.message,
                session_id=request.session_id,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/session/{session_id}", response_model=SessionStateResponse)
async def get_session_state(session_id: str) -> SessionStateResponse:
    """Get current state of a session.

    Returns the current phase, todos, and metrics for the session.
    """
    agent = get_agent()

    try:
        state = agent.get_session_state(session_id)

        if state.get("state") == "not_found":
            raise HTTPException(status_code=404, detail="Session not found")

        return SessionStateResponse(
            session_id=session_id,
            phase=state.get("phase", "requirements"),
            todos=state.get("todos", []),
            requirements_count=state.get("requirements", 0),
            code_files_count=state.get("code_files", 0),
            test_cases_count=state.get("test_cases", 0),
            security_issues_count=state.get("security_issues", 0),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/phase")
async def transition_phase(session_id: str, request: PhaseTransitionRequest):
    """Transition to a new SDLC phase.

    Manually transitions the session to a specified SDLC phase.
    """
    agent = get_agent()

    try:
        # Validate phase
        valid_phases = [phase.value for phase in SDLCPhase]
        if request.new_phase not in valid_phases:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid phase. Valid phases: {valid_phases}",
            )

        new_phase = SDLCPhase(request.new_phase)
        result = agent.transition_phase(session_id, new_phase)

        if result.get("error"):
            raise HTTPException(status_code=404, detail=result["error"])

        return {
            "status": "success",
            "previous_phase": result.get("previous_phase"),
            "new_phase": result.get("new_phase"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error transitioning phase: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def end_session(session_id: str):
    """End a session.

    Cleans up session resources and marks the session as ended.
    """
    # Note: In production, implement proper cleanup
    return {
        "status": "ended",
        "session_id": session_id,
        "message": "Session ended successfully",
    }


@router.get("/subagents", response_model=list[SubagentInfo])
async def list_subagents() -> list[SubagentInfo]:
    """List available subagents.

    Returns information about all specialized subagents
    available in the Software Development DeepAgent.
    """
    from app.deepagents.software_dev.subagents import get_all_subagents

    subagents = get_all_subagents()

    return [
        SubagentInfo(
            name=sa.name,
            description=sa.description,
            tools_count=len(sa.tools),
        )
        for sa in subagents
    ]


@router.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """Get agent configuration.

    Returns the current configuration settings for the agent.
    """
    agent = get_agent()

    return ConfigResponse(
        model=agent.config.model,
        max_iterations=agent.config.max_iterations,
        recursion_limit=agent.config.recursion_limit,
        supported_languages=[lang.value for lang in agent.config.supported_languages],
        quality_gates=agent.config.quality_gates.model_dump(),
    )


@router.get("/phases")
async def list_phases():
    """List all SDLC phases.

    Returns information about each phase in the software development lifecycle.
    """
    phases = [
        {
            "name": "requirements",
            "description": "Analyze and refine software requirements",
            "subagent": "requirements-intelligence",
        },
        {
            "name": "design",
            "description": "Design system architecture and APIs",
            "subagent": "architecture-design",
        },
        {
            "name": "implementation",
            "description": "Generate and refactor code",
            "subagent": "code-generator",
        },
        {
            "name": "review",
            "description": "Perform code reviews and quality checks",
            "subagent": "code-reviewer",
        },
        {
            "name": "testing",
            "description": "Create and run tests",
            "subagent": "testing-automation",
        },
        {
            "name": "security",
            "description": "Scan for security vulnerabilities",
            "subagent": "security-compliance",
        },
        {
            "name": "devops",
            "description": "Create CI/CD pipelines and deployment configs",
            "subagent": "devops-integration",
        },
        {
            "name": "debugging",
            "description": "Debug issues and optimize performance",
            "subagent": "debugging-optimization",
        },
        {
            "name": "documentation",
            "description": "Generate technical documentation",
            "subagent": "documentation",
        },
    ]

    return {"phases": phases, "count": len(phases)}


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent": "Software Development DeepAgent",
        "timestamp": datetime.now().isoformat(),
    }
