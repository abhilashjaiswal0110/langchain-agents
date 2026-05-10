"""REST routes for domain-specific agents.

Exposes all 8 domain agents under /api/domain/* with:
- GET  /api/domain/agents              — list available agents
- POST /api/domain/{domain}/invoke     — invoke a specific domain agent
- POST /api/domain/chat                — auto-route via DomainRouter

Following Enterprise Development Standards:
- Software Architect: Clean REST API design with lazy instantiation
- Software Engineer: Type-safe, well-documented endpoints
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.agents.domains import (
    CloudAgent,
    CybersecurityAgent,
    DataAIAgent,
    DatacenterAgent,
    DomainAgent,
    HRAgent,
    LnDAgent,
    MarComAgent,
    PresalesAgent,
)
from app.agents.supervisors.domain_router import DomainRouter

router = APIRouter(prefix="/api/domain", tags=["Domain Agents"])

# Registry of domain type -> agent class
DOMAIN_AGENT_REGISTRY: dict[str, type[DomainAgent]] = {
    "marcom": MarComAgent,
    "hr": HRAgent,
    "lnd": LnDAgent,
    "presales": PresalesAgent,
    "datacenter": DatacenterAgent,
    "cloud": CloudAgent,
    "cybersecurity": CybersecurityAgent,
    "data_ai": DataAIAgent,
}

# Lazy-loaded agent instances (singleton per domain)
_agent_instances: dict[str, DomainAgent] = {}
_domain_router: DomainRouter | None = None


def _get_agent(domain: str) -> DomainAgent:
    """Get or create a domain agent instance.

    Args:
        domain: Domain type key (e.g. 'cloud', 'hr').

    Returns:
        DomainAgent instance for the requested domain.

    Raises:
        HTTPException: If the domain is not in the registry.
    """
    if domain not in DOMAIN_AGENT_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Domain '{domain}' not found. Available: {list(DOMAIN_AGENT_REGISTRY.keys())}",
        )
    if domain not in _agent_instances:
        _agent_instances[domain] = DOMAIN_AGENT_REGISTRY[domain]()
    return _agent_instances[domain]


def _get_domain_router() -> DomainRouter:
    """Get or create the singleton DomainRouter."""
    global _domain_router
    if _domain_router is None:
        _domain_router = DomainRouter()
    return _domain_router


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class DomainInvokeRequest(BaseModel):
    """Request body for domain agent invocation.

    Attributes:
        message: User message text.
        session_id: Optional thread ID for conversation continuity.
        user_context: Optional user metadata (display_name, email, role, etc.).
    """

    message: str
    session_id: str | None = None
    user_context: dict[str, Any] | None = None


class DomainInvokeResponse(BaseModel):
    """Response from a domain agent invocation.

    Attributes:
        response: Agent response text.
        agent_type: The domain that handled the request.
        domain: Alias for agent_type (for compatibility).
        escalation_requested: True if the agent escalated the request.
    """

    response: str
    agent_type: str
    domain: str
    escalation_requested: bool = False


class AgentEntry(BaseModel):
    """Single entry in the agent list response.

    Attributes:
        type: Domain key.
        name: Human-readable name.
        description: Agent description.
    """

    type: str
    name: str
    description: str


class AgentListResponse(BaseModel):
    """Response for the agent listing endpoint."""

    agents: list[AgentEntry]


class DomainChatRequest(BaseModel):
    """Request body for the auto-routing chat endpoint.

    Attributes:
        message: User message text.
        user_context: Optional user metadata.
        session_id: Optional thread ID.
    """

    message: str
    user_context: dict[str, Any] | None = None
    session_id: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/agents", response_model=AgentListResponse)
def list_domain_agents() -> AgentListResponse:
    """List all available domain agents.

    Returns:
        JSON with a list of agent descriptors (type, name, description).
    """
    entries = []
    for key in DOMAIN_AGENT_REGISTRY:
        agent = _get_agent(key)
        entries.append(AgentEntry(
            type=key,
            name=agent.name,
            description=agent.description or "",
        ))
    return AgentListResponse(agents=entries)


@router.post("/{domain}/invoke", response_model=DomainInvokeResponse)
async def invoke_domain_agent(
    domain: str,
    body: DomainInvokeRequest,
) -> DomainInvokeResponse:
    """Invoke a specific domain agent with a user message.

    Args:
        domain: Domain type (e.g. ``cloud``, ``hr``, ``cybersecurity``).
        body: Request containing message and optional session/user context.

    Returns:
        Agent response with domain metadata.

    Raises:
        HTTPException: 404 if the domain is not found.
    """
    agent = _get_agent(domain)

    messages = [HumanMessage(content=body.message)]
    result = await agent.ainvoke(
        messages=messages,
        user_context=body.user_context or {},
        thread_id=body.session_id,
    )

    return DomainInvokeResponse(
        response=result.get("response", ""),
        agent_type=domain,
        domain=result.get("domain", domain),
        escalation_requested=result.get("escalation_requested", False),
    )


@router.post("/chat")
async def chat_via_router(body: DomainChatRequest) -> dict[str, Any]:
    """Route a message to the best-matching domain agent automatically.

    Uses keyword-based classification first, then LLM fallback, to determine
    the target domain and invokes the appropriate agent.

    Args:
        body: Request containing message and optional user context.

    Returns:
        Agent response with routing metadata.
    """
    domain_router = _get_domain_router()

    try:
        # Classify the intent
        routing = await domain_router.aclassify(body.message)
        target_domain = routing.intent.value

        # If classified to a known domain, invoke it
        if target_domain in DOMAIN_AGENT_REGISTRY:
            agent = _get_agent(target_domain)
            messages = [HumanMessage(content=body.message)]
            result = await agent.ainvoke(
                messages=messages,
                user_context=body.user_context or {},
                thread_id=body.session_id,
            )
            return {
                "response": result.get("response", ""),
                "domain": target_domain,
                "routing": routing.to_dict(),
            }

        # Fallback: unknown or general intent without a dedicated domain agent
        return {
            "response": "I'm not sure which specialist can best help with this request. Please provide more details or contact IT support.",
            "domain": target_domain,
            "routing": routing.to_dict(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Router error: {str(e)}")
