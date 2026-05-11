"""Base Domain Agent for specialized IT support domains.

Provides a common interface and functionality for all domain agents:
- Domain-specific system prompts
- Tool registration
- User context handling
- Response formatting

Following Enterprise Development Standards:
- Software Architect: Template method pattern
- Software Engineer: Clean abstractions, type safety
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langsmith import traceable

from app.agents.base.llm_factory import get_llm
from app.auth.user_context import UserContext


class DomainType(str, Enum):
    """Available domain types."""

    MARCOM = "marcom"
    HR = "hr"
    LND = "lnd"
    PRESALES = "presales"
    DATACENTER = "datacenter"
    CLOUD = "cloud"
    CYBERSECURITY = "cybersecurity"
    DATA_AI = "data_ai"
    FINANCE = "finance"
    GENERAL = "general"


@dataclass
class DomainConfig:
    """Configuration for a domain agent.

    Attributes:
        domain: Domain type
        name: Human-readable name
        description: Domain description
        expertise: List of expertise areas
        escalation_keywords: Keywords that trigger escalation
        requires_approval: Actions requiring approval
    """

    domain: DomainType
    name: str
    description: str
    expertise: list[str] = field(default_factory=list)
    escalation_keywords: list[str] = field(default_factory=list)
    requires_approval: list[str] = field(default_factory=list)


class DomainAgent(ABC):
    """Base class for domain-specific agents.

    Provides common functionality for all domain agents:
    - LLM integration via factory
    - Tool management
    - User context handling
    - Response generation

    Subclasses must implement:
    - get_config(): Return domain configuration
    - get_tools(): Return domain-specific tools
    - get_system_prompt(): Return domain system prompt

    Example:
        >>> class CloudAgent(DomainAgent):
        ...     def get_config(self) -> DomainConfig:
        ...         return DomainConfig(
        ...             domain=DomainType.CLOUD,
        ...             name="Cloud Infrastructure",
        ...             description="Azure, AWS, GCP support",
        ...         )
        ...
        ...     def get_tools(self) -> list[BaseTool]:
        ...         return [list_vms_tool, check_status_tool]
        ...
        ...     def get_system_prompt(self) -> str:
        ...         return "You are a cloud infrastructure specialist..."
    """

    def __init__(
        self,
        llm: Any = None,
        checkpointer: Any = None,
    ) -> None:
        """Initialize domain agent.

        Args:
            llm: LangChain LLM instance.
            checkpointer: LangGraph checkpointer.
        """
        self._llm = llm
        self._checkpointer = checkpointer or MemorySaver()
        self._agent = None
        self._config = self.get_config()

    def _get_llm(self) -> Any:
        """Get or create LLM instance."""
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    @abstractmethod
    def get_config(self) -> DomainConfig:
        """Get domain configuration.

        Returns:
            DomainConfig for this agent.
        """

    @abstractmethod
    def get_tools(self) -> list[BaseTool]:
        """Get domain-specific tools.

        Returns:
            List of tools for this domain.
        """

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get domain-specific system prompt.

        Returns:
            System prompt string.
        """

    def _build_full_prompt(self, user_context: dict[str, Any]) -> str:
        """Build full prompt with user context.

        Args:
            user_context: User context dictionary.

        Returns:
            Complete system prompt.
        """
        base_prompt = self.get_system_prompt()

        user_info = f"""
User Information:
- Name: {user_context.get("display_name", "User")}
- Email: {user_context.get("email", "unknown")}
- Role: {user_context.get("primary_role", "user")}
- Department: {user_context.get("department", "unknown")}

Instructions:
1. Be helpful and professional
2. Stay within your domain expertise
3. If a request is outside your domain, politely indicate that
4. For sensitive actions, explain what you'll do before doing it
5. Ask clarifying questions if the request is ambiguous
"""
        return base_prompt + "\n\n" + user_info

    def _create_agent(self, user_context: dict[str, Any]) -> Any:
        """Create the LangGraph agent.

        Args:
            user_context: User context for prompt.

        Returns:
            Compiled agent graph.
        """
        llm = self._get_llm()
        tools = self.get_tools()
        system_prompt = self._build_full_prompt(user_context)

        return create_react_agent(
            model=llm,
            tools=tools,
            prompt=system_prompt,
            checkpointer=self._checkpointer,
        )

    def _should_escalate(self, message: str) -> bool:
        """Check if message requires escalation.

        Args:
            message: User message.

        Returns:
            True if escalation is needed.
        """
        message_lower = message.lower()
        for keyword in self._config.escalation_keywords:
            if keyword.lower() in message_lower:
                return True
        return False

    @traceable(name="domain_agent_invoke")
    async def ainvoke(
        self,
        messages: list[BaseMessage] | dict[str, Any],
        user_context: UserContext | dict[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Invoke the domain agent asynchronously.

        Args:
            messages: Conversation messages or state dict.
            user_context: User context.
            thread_id: Thread ID for persistence.

        Returns:
            Result with response and metadata.
        """
        # Handle dict input
        if isinstance(messages, dict):
            msg_list = messages.get("messages", [])
            user_ctx = messages.get("user_context", user_context)
        else:
            msg_list = messages
            user_ctx = user_context

        # Convert UserContext to dict
        if isinstance(user_ctx, UserContext):
            user_ctx = user_ctx.to_dict()
        user_ctx = user_ctx or {}

        # Create agent with user context
        agent = self._create_agent(user_ctx)

        config = {"configurable": {"thread_id": thread_id or "default"}}

        # Check for escalation
        if msg_list:
            last_msg = msg_list[-1]
            content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            if self._should_escalate(content):
                return {
                    "response": self._escalation_response(),
                    "messages": [AIMessage(content=self._escalation_response())],
                    "escalation_requested": True,
                    "domain": self._config.domain.value,
                }

        # Invoke agent
        try:
            result = await agent.ainvoke(
                {"messages": msg_list},
                config,
            )

            response = ""
            if result.get("messages"):
                last = result["messages"][-1]
                response = last.content if hasattr(last, "content") else str(last)

            return {
                "response": response,
                "messages": result.get("messages", []),
                "domain": self._config.domain.value,
            }

        except Exception as e:
            error_msg = (
                f"I encountered an issue processing your request. Please try again or contact support. Error: {e}"
            )
            return {
                "response": error_msg,
                "messages": [AIMessage(content=error_msg)],
                "error": str(e),
                "domain": self._config.domain.value,
            }

    def _escalation_response(self) -> str:
        """Get escalation response message.

        Returns:
            Escalation message.
        """
        return (
            f"This request involves sensitive matters that require human review. "
            f"I'm escalating this to the {self._config.name} team for assistance."
        )

    @property
    def domain(self) -> str:
        """Get domain identifier."""
        return self._config.domain.value

    @property
    def name(self) -> str:
        """Get domain name."""
        return self._config.name

    @property
    def description(self) -> str:
        """Get domain description."""
        return self._config.description


def create_domain_agent(
    domain: DomainType,
    llm: Any = None,
    checkpointer: Any = None,
) -> DomainAgent:
    """Factory function to create a domain agent.

    Args:
        domain: Domain type to create.
        llm: Optional LLM instance.
        checkpointer: Optional checkpointer.

    Returns:
        DomainAgent instance.

    Raises:
        ValueError: If domain is not supported.
    """
    # Import here to avoid circular imports
    from app.agents.domains.cloud_agent import CloudAgent
    from app.agents.domains.cybersecurity_agent import CybersecurityAgent
    from app.agents.domains.data_ai_agent import DataAIAgent
    from app.agents.domains.datacenter_agent import DatacenterAgent
    from app.agents.domains.finance_agent import FinanceAgent
    from app.agents.domains.hr_agent import HRAgent
    from app.agents.domains.lnd_agent import LnDAgent
    from app.agents.domains.marcom_agent import MarComAgent
    from app.agents.domains.presales_agent import PresalesAgent

    agents = {
        DomainType.MARCOM: MarComAgent,
        DomainType.HR: HRAgent,
        DomainType.LND: LnDAgent,
        DomainType.PRESALES: PresalesAgent,
        DomainType.DATACENTER: DatacenterAgent,
        DomainType.CLOUD: CloudAgent,
        DomainType.CYBERSECURITY: CybersecurityAgent,
        DomainType.DATA_AI: DataAIAgent,
        DomainType.FINANCE: FinanceAgent,
    }

    agent_class = agents.get(domain)
    if agent_class is None:
        msg = f"Unsupported domain: {domain}"
        raise ValueError(msg)

    return agent_class(llm=llm, checkpointer=checkpointer)
