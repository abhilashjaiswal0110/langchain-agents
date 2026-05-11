"""Master Orchestrator — unified single entry point for all agent clusters.

Routes any user message to the most appropriate agent automatically:
- IT Support / Helpdesk → ConversationManager (it_helpdesk)
- Domain-specific (HR, Finance, Cloud, …) → DomainRouter → domain agent
- Deep analysis (IT Ops, Sales Intel, Recruitment) → appropriate deep agent
- General knowledge / research → Research enterprise agent

Usage:
    from app.agents.supervisors.master_orchestrator import MasterOrchestrator

    orchestrator = MasterOrchestrator()
    result = await orchestrator.route("Reset my password", session_id="s1")
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# Cluster constants — determines which pipeline handles the request.
_CLUSTER_IT_SUPPORT = "IT_SUPPORT"
_CLUSTER_DOMAIN = "DOMAIN"
_CLUSTER_DEEP = "DEEP_AGENT"
_CLUSTER_RESEARCH = "RESEARCH"

# Domain intents that map to domain agents (not general IT)
_DOMAIN_INTENTS = {
    "marcom", "hr", "lnd", "presales", "datacenter",
    "cloud", "cybersecurity", "data_ai", "finance",
}

# Deep-agent trigger keywords
_DEEP_AGENT_KEYWORDS = {
    "root cause analysis", "rca", "post-mortem", "incident report",
    "capacity planning", "performance analysis", "architecture review",
    "recruitment", "candidate", "shortlist", "job description",
    "sales intelligence", "competitor analysis", "deal strategy",
}


class MasterOrchestrator:
    """Routes any user message to the most appropriate agent cluster.

    Classification order:
    1. Deep-agent keyword trigger → deep agent
    2. DomainRouter intent (non-general) → domain agent
    3. General IT keywords → IT Helpdesk
    4. Default → Research agent

    Args:
        conversation_manager: Optional pre-created ConversationManager.
        domain_router: Optional pre-created DomainRouter.
    """

    def __init__(
        self,
        conversation_manager: Any = None,
        domain_router: Any = None,
    ) -> None:
        self._cm = conversation_manager
        self._domain_router = domain_router

    def _get_domain_router(self) -> Any:
        if self._domain_router is None:
            from app.agents.supervisors.domain_router import DomainRouter

            self._domain_router = DomainRouter()
        return self._domain_router

    def _classify_cluster(self, message: str) -> str:
        """Classify which cluster should handle *message*.

        Args:
            message: User message to classify.

        Returns:
            Cluster constant string.
        """
        msg_lower = message.lower()

        # 1. Deep-agent keyword trigger
        for kw in _DEEP_AGENT_KEYWORDS:
            if kw in msg_lower:
                logger.debug("Deep-agent keyword '%s' matched", kw)
                return _CLUSTER_DEEP

        # 2. Domain router classification (keyword-based, no LLM call)
        router = self._get_domain_router()
        result = router.classify(message)
        if result.intent in _DOMAIN_INTENTS:
            logger.debug("Domain intent '%s' matched (confidence %.2f)", result.intent, result.confidence)
            return _CLUSTER_DOMAIN

        # 3. General IT (maps to IT Helpdesk)
        if result.intent == "general":
            return _CLUSTER_IT_SUPPORT

        # 4. Default: research
        return _CLUSTER_RESEARCH

    async def route(
        self,
        message: str,
        session_id: str | None = None,
        user_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Route a message to the appropriate agent and return its response.

        Args:
            message: User message.
            session_id: Optional session/thread ID for conversation continuity.
            user_context: Optional metadata (user_id, tenant_id, …).

        Returns:
            Dict with ``cluster``, ``response``, and ``agent_type``.
        """
        user_context = user_context or {}
        cluster = self._classify_cluster(message)
        logger.info("Orchestrator routing '%s...' → cluster=%s", message[:40], cluster)

        try:
            if cluster == _CLUSTER_IT_SUPPORT:
                return await self._route_it_support(message, session_id, user_context)
            elif cluster == _CLUSTER_DOMAIN:
                return await self._route_domain(message, session_id, user_context)
            elif cluster == _CLUSTER_DEEP:
                return await self._route_deep(message, session_id, user_context)
            else:
                return await self._route_research(message, session_id)
        except Exception as exc:
            logger.error("Orchestrator routing failed for cluster %s: %s", cluster, exc)
            return {
                "cluster": cluster,
                "agent_type": "error",
                "response": "An error occurred routing your request. Please try again.",
                "error": str(exc),
            }

    async def _route_it_support(
        self, message: str, session_id: str | None, user_context: dict
    ) -> dict[str, Any]:
        if self._cm is None:
            from app.agents.conversation_manager import ConversationManager

            self._cm = ConversationManager()

        tenant_id = user_context.get("tenant_id", "default")
        if session_id is None:
            response = self._cm.start_conversation(
                agent_type="it_helpdesk",
                user_id=user_context.get("user_id"),
                tenant_id=tenant_id,
            )
            session_id = response.get("session_id")

        result = await self._cm.achat(session_id, message, tenant_id=tenant_id)
        return {
            "cluster": _CLUSTER_IT_SUPPORT,
            "agent_type": "it_helpdesk",
            "session_id": session_id,
            "response": result.get("response", ""),
        }

    async def _route_domain(
        self, message: str, session_id: str | None, user_context: dict
    ) -> dict[str, Any]:
        router = self._get_domain_router()
        result = router.classify(message)
        domain = result.intent

        from app.agents.domains.routes import DOMAIN_AGENT_REGISTRY, _get_agent

        agent_key = domain if domain in DOMAIN_AGENT_REGISTRY else "data_ai"
        agent = _get_agent(agent_key)
        invoke_result = await agent.ainvoke(
            [HumanMessage(content=message)],
            thread_id=session_id,
        )
        return {
            "cluster": _CLUSTER_DOMAIN,
            "agent_type": agent_key,
            "session_id": session_id,
            "response": invoke_result.get("response", ""),
            "routing_confidence": result.confidence,
        }

    async def _route_deep(
        self, message: str, session_id: str | None, user_context: dict
    ) -> dict[str, Any]:
        # Default deep agent: IT Operations
        try:
            from app.deepagents.it_operations_agent import ITOperationsDeepAgent  # type: ignore[import]

            agent = ITOperationsDeepAgent()
            response = await agent.achat(message, session_id=session_id or "default")
        except Exception:
            # Graceful fallback to research if deep agent unavailable
            return await self._route_research(message, session_id)

        return {
            "cluster": _CLUSTER_DEEP,
            "agent_type": "it_operations_deep",
            "session_id": session_id,
            "response": response,
        }

    async def _route_research(
        self, message: str, session_id: str | None
    ) -> dict[str, Any]:
        try:
            from app.agents.research.research_agent import ResearchAgent  # type: ignore[import]

            agent = ResearchAgent()
            result = agent.research(query=message, session_id=session_id)
            if isinstance(result, dict):
                last = result.get("messages", [None])[-1]
                response = last.content if hasattr(last, "content") else str(last) if last is not None else ""
            else:
                response = str(result)
        except Exception as exc:
            logger.warning("Research agent unavailable: %s", exc)
            response = f"I couldn't process your request automatically. Please try a more specific agent endpoint."

        return {
            "cluster": _CLUSTER_RESEARCH,
            "agent_type": "research",
            "session_id": session_id,
            "response": response,
        }
