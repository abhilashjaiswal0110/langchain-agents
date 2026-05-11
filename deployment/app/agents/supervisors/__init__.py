"""Supervisor Agents for Multi-Agent Orchestration.

Provides hierarchical agent management using LangGraph patterns:
- IT Supervisor: Main orchestrator for all domain agents
- Domain Router: Fast routing based on intent classification
- Escalation Handler: Cross-domain escalation management

Following Enterprise Development Standards:
- Software Architect: Supervisor pattern for complex workflows
- Security Architect: User context propagation, audit trails
- Data Architect: State management across agents
- Software Engineer: Type-safe, well-documented

Example:
    >>> from app.agents.supervisors import get_it_supervisor
    >>> supervisor = get_it_supervisor()
    >>> result = await supervisor.ainvoke({
    ...     "messages": [HumanMessage(content="I need help with cloud VMs")],
    ...     "user_context": user,
    ... })
"""

from app.agents.supervisors.domain_router import (
    DomainIntent,
    DomainRouter,
    get_domain_router,
)
from app.agents.supervisors.escalation_handler import (
    EscalationHandler,
    EscalationLevel,
    EscalationRequest,
)
from app.agents.supervisors.it_supervisor import (
    ITSupervisor,
    SupervisorState,
    create_it_supervisor,
    get_it_supervisor,
)

__all__ = [
    # IT Supervisor
    "ITSupervisor",
    "SupervisorState",
    "get_it_supervisor",
    "create_it_supervisor",
    # Domain Router
    "DomainRouter",
    "DomainIntent",
    "get_domain_router",
    # Escalation Handler
    "EscalationHandler",
    "EscalationLevel",
    "EscalationRequest",
]
