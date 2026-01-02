"""IT Supervisor Agent for Multi-Agent Orchestration.

Implements the Supervisor pattern using LangGraph:
- Routes requests to appropriate domain agents
- Manages cross-domain conversations
- Handles escalations and human-in-the-loop
- Maintains conversation context across agents

Following Enterprise Development Standards:
- Software Architect: Supervisor pattern, state machine design
- Security Architect: User context propagation, audit logging
- Data Architect: Conversation state management
- Software Engineer: Type-safe, async-first
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Literal, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langsmith import traceable
from pydantic import BaseModel, Field

from app.agents.base.llm_factory import get_llm
from app.auth.user_context import UserContext


class DomainType(str, Enum):
    """Available domain agents."""

    MARCOM = "marcom"
    HR = "hr"
    LND = "lnd"  # Learning & Development
    PRESALES = "presales"
    DATACENTER = "datacenter"
    CLOUD = "cloud"
    CYBERSECURITY = "cybersecurity"
    DATA_AI = "data_ai"
    GENERAL = "general"  # General IT support


class SupervisorAction(str, Enum):
    """Actions the supervisor can take."""

    ROUTE = "route"  # Route to a domain agent
    RESPOND = "respond"  # Respond directly
    ESCALATE = "escalate"  # Escalate to human
    CLARIFY = "clarify"  # Ask for clarification


@dataclass
class SupervisorState:
    """State for supervisor workflow.

    Attributes:
        messages: Conversation history
        user_context: Authenticated user information
        current_domain: Currently active domain agent
        domains_consulted: List of domains already consulted
        escalation_requested: Whether escalation was requested
        final_response: Final response to user
        metadata: Additional metadata for tracking
    """

    messages: Annotated[list[BaseMessage], add_messages] = field(default_factory=list)
    user_context: dict[str, Any] = field(default_factory=dict)
    current_domain: str | None = None
    domains_consulted: list[str] = field(default_factory=list)
    escalation_requested: bool = False
    final_response: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RoutingDecision(BaseModel):
    """Structured output for routing decisions."""

    action: SupervisorAction = Field(description="Action to take")
    domain: DomainType | None = Field(default=None, description="Domain to route to")
    reasoning: str = Field(description="Brief reasoning for the decision")
    response: str | None = Field(default=None, description="Direct response if action is RESPOND")


SUPERVISOR_PROMPT = """You are the IT Support Supervisor for a large enterprise organization.
Your role is to understand user requests and route them to the appropriate specialized domain agent.

Available domain agents:
- marcom: Marketing & Communications support (campaigns, branding, content)
- hr: Human Resources (benefits, policies, onboarding, payroll)
- lnd: Learning & Development (training, certifications, courses)
- presales: Presales/Sales support (demos, proposals, customer inquiries)
- datacenter: Datacenter operations (servers, storage, networking, physical infrastructure)
- cloud: Cloud infrastructure (Azure, AWS, GCP, VMs, containers, IaaS/PaaS)
- cybersecurity: Security operations (incidents, vulnerabilities, compliance, access)
- data_ai: Data & AI support (analytics, ML, data pipelines, AI tools)
- general: General IT support (basic tech issues, account access, software)

User Information:
- Name: {user_name}
- Email: {user_email}
- Role: {user_role}
- Department: {user_department}

Instructions:
1. Analyze the user's request to determine the most appropriate domain
2. If the request spans multiple domains, start with the primary one
3. If the request is simple and you can answer directly, respond with action=RESPOND
4. If the request is unclear, use action=CLARIFY
5. For sensitive issues or escalations, use action=ESCALATE

Previous domains consulted: {domains_consulted}

Respond with your routing decision."""


class ITSupervisor:
    """IT Supervisor for orchestrating domain agents.

    Uses LangGraph to implement a supervisor pattern that:
    1. Receives user requests
    2. Classifies intent and routes to appropriate domain
    3. Manages multi-domain conversations
    4. Handles escalations to human operators

    Example:
        >>> supervisor = ITSupervisor()
        >>> graph = supervisor.build()
        >>> result = await graph.ainvoke({
        ...     "messages": [HumanMessage(content="Help with Azure VMs")],
        ...     "user_context": {"name": "John", "role": "user"},
        ... })
    """

    def __init__(
        self,
        llm: Any = None,
        checkpointer: Any = None,
        domain_agents: dict[str, Any] | None = None,
    ) -> None:
        """Initialize IT Supervisor.

        Args:
            llm: LangChain LLM instance (defaults to factory).
            checkpointer: LangGraph checkpointer for persistence.
            domain_agents: Dict of domain name to agent instances.
        """
        self._llm = llm
        self._checkpointer = checkpointer or MemorySaver()
        self._domain_agents = domain_agents or {}
        self._graph = None

    def _get_llm(self) -> Any:
        """Get or create LLM instance."""
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def _create_routing_chain(self) -> Any:
        """Create the routing decision chain."""
        llm = self._get_llm()

        prompt = ChatPromptTemplate.from_messages([
            ("system", SUPERVISOR_PROMPT),
            ("placeholder", "{messages}"),
        ])

        # Use structured output for reliable parsing
        return prompt | llm.with_structured_output(RoutingDecision)

    @traceable(name="supervisor_route")
    async def _route_request(self, state: dict[str, Any]) -> dict[str, Any]:
        """Route the request to appropriate domain.

        Args:
            state: Current supervisor state.

        Returns:
            Updated state with routing decision.
        """
        messages = state.get("messages", [])
        user_context = state.get("user_context", {})
        domains_consulted = state.get("domains_consulted", [])

        chain = self._create_routing_chain()

        decision = await chain.ainvoke({
            "messages": messages,
            "user_name": user_context.get("display_name", "User"),
            "user_email": user_context.get("email", "unknown"),
            "user_role": user_context.get("primary_role", "user"),
            "user_department": user_context.get("department", "unknown"),
            "domains_consulted": ", ".join(domains_consulted) if domains_consulted else "none",
        })

        # Update state based on decision
        updates: dict[str, Any] = {
            "metadata": {
                **state.get("metadata", {}),
                "last_decision": decision.model_dump(),
            },
        }

        if decision.action == SupervisorAction.ROUTE and decision.domain:
            updates["current_domain"] = decision.domain.value
        elif decision.action == SupervisorAction.RESPOND and decision.response:
            updates["final_response"] = decision.response
            updates["messages"] = [AIMessage(content=decision.response)]
        elif decision.action == SupervisorAction.ESCALATE:
            updates["escalation_requested"] = True
            updates["final_response"] = (
                "I'm escalating this to a human operator who can better assist you. "
                "Someone will reach out shortly."
            )
            updates["messages"] = [AIMessage(content=updates["final_response"])]
        elif decision.action == SupervisorAction.CLARIFY:
            clarify_msg = decision.response or "Could you please provide more details about your request?"
            updates["messages"] = [AIMessage(content=clarify_msg)]

        return updates

    async def _invoke_domain_agent(self, state: dict[str, Any]) -> dict[str, Any]:
        """Invoke the selected domain agent.

        Args:
            state: Current supervisor state.

        Returns:
            Updated state with domain agent response.
        """
        domain = state.get("current_domain")
        messages = state.get("messages", [])
        user_context = state.get("user_context", {})
        domains_consulted = state.get("domains_consulted", [])

        if not domain:
            return {
                "messages": [AIMessage(content="I apologize, but I couldn't determine the right department. Please try again.")],
            }

        # Get domain agent
        agent = self._domain_agents.get(domain)

        if agent:
            # Real domain agent available
            try:
                result = await agent.ainvoke({
                    "messages": messages,
                    "user_context": user_context,
                })
                response = result.get("response", result.get("messages", [])[-1].content if result.get("messages") else "")
            except Exception as e:
                response = f"The {domain} team is currently experiencing issues. Please try again later."
        else:
            # Simulate domain agent response for now
            response = await self._simulate_domain_response(domain, messages, user_context)

        # Track consulted domains
        new_domains = domains_consulted + [domain] if domain not in domains_consulted else domains_consulted

        return {
            "messages": [AIMessage(content=response)],
            "domains_consulted": new_domains,
            "final_response": response,
            "current_domain": None,  # Reset for next routing
        }

    async def _simulate_domain_response(
        self,
        domain: str,
        messages: list[BaseMessage],
        user_context: dict[str, Any],
    ) -> str:
        """Simulate domain agent response when not available.

        Args:
            domain: Domain name.
            messages: Conversation messages.
            user_context: User context.

        Returns:
            Simulated response.
        """
        llm = self._get_llm()

        domain_prompts = {
            "marcom": "You are the Marketing & Communications specialist. Help with campaigns, branding, content, and communications.",
            "hr": "You are the HR specialist. Help with benefits, policies, onboarding, payroll, and employee relations.",
            "lnd": "You are the Learning & Development specialist. Help with training, certifications, courses, and skill development.",
            "presales": "You are the Presales/Sales support specialist. Help with demos, proposals, RFPs, and customer inquiries.",
            "datacenter": "You are the Datacenter operations specialist. Help with servers, storage, networking, and physical infrastructure.",
            "cloud": "You are the Cloud infrastructure specialist. Help with Azure, AWS, GCP, VMs, containers, and cloud services.",
            "cybersecurity": "You are the Security operations specialist. Help with security incidents, vulnerabilities, compliance, and access control.",
            "data_ai": "You are the Data & AI specialist. Help with analytics, machine learning, data pipelines, and AI tools.",
            "general": "You are the General IT support specialist. Help with basic tech issues, account access, and software.",
        }

        prompt = ChatPromptTemplate.from_messages([
            ("system", domain_prompts.get(domain, domain_prompts["general"])),
            ("placeholder", "{messages}"),
        ])

        chain = prompt | llm
        result = await chain.ainvoke({"messages": messages})

        return result.content if hasattr(result, "content") else str(result)

    def _should_continue(self, state: dict[str, Any]) -> str:
        """Determine next step in workflow.

        Args:
            state: Current state.

        Returns:
            Next node name or END.
        """
        if state.get("escalation_requested"):
            return END
        if state.get("final_response"):
            return END
        if state.get("current_domain"):
            return "domain_agent"
        return END

    def build(self) -> Any:
        """Build the supervisor graph.

        Returns:
            Compiled LangGraph workflow.
        """
        if self._graph is not None:
            return self._graph

        # Create state graph
        workflow = StateGraph(dict)

        # Add nodes
        workflow.add_node("route", self._route_request)
        workflow.add_node("domain_agent", self._invoke_domain_agent)

        # Set entry point
        workflow.set_entry_point("route")

        # Add conditional edges
        workflow.add_conditional_edges(
            "route",
            self._should_continue,
            {
                "domain_agent": "domain_agent",
                END: END,
            },
        )

        # Domain agent always ends or routes back
        workflow.add_edge("domain_agent", END)

        # Compile with checkpointer
        self._graph = workflow.compile(checkpointer=self._checkpointer)

        return self._graph

    @traceable(name="it_supervisor_invoke")
    async def invoke(
        self,
        messages: list[BaseMessage],
        user_context: UserContext | dict[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Invoke the supervisor with a request.

        Args:
            messages: Conversation messages.
            user_context: User context for personalization.
            thread_id: Thread ID for conversation persistence.

        Returns:
            Result with response and metadata.
        """
        graph = self.build()

        # Convert UserContext to dict if needed
        if isinstance(user_context, UserContext):
            user_ctx = user_context.to_dict()
        else:
            user_ctx = user_context or {}

        config = {"configurable": {"thread_id": thread_id or "default"}}

        result = await graph.ainvoke(
            {
                "messages": messages,
                "user_context": user_ctx,
                "domains_consulted": [],
                "escalation_requested": False,
            },
            config,
        )

        return {
            "response": result.get("final_response", ""),
            "messages": result.get("messages", []),
            "domains_consulted": result.get("domains_consulted", []),
            "escalation_requested": result.get("escalation_requested", False),
            "metadata": result.get("metadata", {}),
        }

    def register_domain_agent(self, domain: str, agent: Any) -> None:
        """Register a domain agent.

        Args:
            domain: Domain name (must be valid DomainType).
            agent: Agent instance with ainvoke method.
        """
        if domain not in [d.value for d in DomainType]:
            msg = f"Invalid domain: {domain}"
            raise ValueError(msg)
        self._domain_agents[domain] = agent


# Singleton instance
_supervisor: ITSupervisor | None = None


def get_it_supervisor(
    domain_agents: dict[str, Any] | None = None,
) -> ITSupervisor:
    """Get or create IT Supervisor singleton.

    Args:
        domain_agents: Optional domain agents to register.

    Returns:
        ITSupervisor instance.
    """
    global _supervisor
    if _supervisor is None:
        _supervisor = ITSupervisor()

    if domain_agents:
        for domain, agent in domain_agents.items():
            _supervisor.register_domain_agent(domain, agent)

    return _supervisor


def create_it_supervisor(
    llm: Any = None,
    checkpointer: Any = None,
    domain_agents: dict[str, Any] | None = None,
) -> ITSupervisor:
    """Create a new IT Supervisor instance.

    Args:
        llm: LangChain LLM instance.
        checkpointer: LangGraph checkpointer.
        domain_agents: Domain agents to register.

    Returns:
        New ITSupervisor instance.
    """
    return ITSupervisor(
        llm=llm,
        checkpointer=checkpointer,
        domain_agents=domain_agents or {},
    )
