"""Base agent class for all enterprise IT agents.

This module provides the foundational architecture for building
stateful, multi-conversational agents with LangGraph.

Following Enterprise Development Standards:
- Software Architect: Modular, extensible design
- Security Architect: No hardcoded secrets, input validation
- Data Architect: Type-safe state management
- Software Engineer: Full type hints, error handling
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langsmith import traceable
from pydantic import BaseModel, Field

# Import memory module components
from app.agents.memory.checkpointers import (
    CheckpointerBackend,
    get_checkpointer,
)
from app.agents.memory.semantic_memory import get_semantic_memory
from app.agents.memory.summarizer import get_summarizer


# Type variable for state
StateT = TypeVar("StateT", bound=BaseModel)


@dataclass
class AgentConfig:
    """Configuration for agent initialization.

    Attributes:
        model_provider: LLM provider to use ("openai", "anthropic", "auto")
        model_name: Specific model name (e.g., "gpt-4o-mini", "claude-3-sonnet")
        temperature: Model temperature for response generation
        max_tokens: Maximum tokens in response
        checkpointer: Optional checkpointer for state persistence
        memory_backend: Memory backend type ("postgres", "sqlite", "memory", "auto")
        semantic_memory_enabled: Whether to enable semantic long-term memory
        conversation_summarization: Whether to enable conversation summarization
        tracing_enabled: Whether to enable LangSmith tracing
        project_name: LangSmith project name for tracing
    """

    model_provider: Literal["openai", "anthropic", "auto"] = "auto"
    model_name: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    checkpointer: BaseCheckpointSaver | None = None
    memory_backend: Literal["postgres", "sqlite", "memory", "auto"] = "auto"
    semantic_memory_enabled: bool = False
    conversation_summarization: bool = False
    tracing_enabled: bool = True
    project_name: str = "enterprise-it-agents"


class BaseAgentState(BaseModel):
    """Base state schema for all agents.

    All agent states should extend this class and add
    domain-specific fields.
    """

    messages: Annotated[list[BaseMessage], add_messages] = Field(
        default_factory=list,
        description="Conversation message history"
    )
    session_id: str | None = Field(
        default=None,
        description="Unique session identifier"
    )
    user_id: str | None = Field(
        default=None,
        description="User identifier for personalization"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the session"
    )


class BaseAgent(ABC):
    """Abstract base class for all enterprise IT agents.

    This class provides:
    - LLM provider abstraction (OpenAI/Anthropic)
    - State management with checkpointing
    - LangSmith tracing integration
    - Common utility methods

    Subclasses must implement:
    - _build_graph(): Define the agent's workflow graph
    - _get_system_prompt(): Return the agent's system prompt
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize the base agent.

        Args:
            config: Agent configuration. Uses defaults if not provided.
        """
        self.config = config or AgentConfig()
        self._llm: BaseChatModel | None = None
        self._graph: StateGraph | None = None
        self._compiled_graph = None
        self._tools: list = []

        # Initialize LLM
        self._init_llm()

        # Setup tracing
        if self.config.tracing_enabled:
            self._setup_tracing()

    def _init_llm(self) -> None:
        """Initialize the language model based on configuration."""
        provider = self.config.model_provider

        # Auto-detect available provider (prefer OpenAI for broader compatibility)
        if provider == "auto":
            if os.getenv("OPENAI_API_KEY"):
                provider = "openai"
            elif os.getenv("ANTHROPIC_API_KEY"):
                provider = "anthropic"
            else:
                raise ValueError(
                    "No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY"
                )

        if provider == "openai":
            model_name = self.config.model_name or "gpt-4o-mini"
            self._llm = ChatOpenAI(
                model=model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        elif provider == "anthropic":
            model_name = self.config.model_name or "claude-3-5-sonnet-latest"
            self._llm = ChatAnthropic(
                model=model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _setup_tracing(self) -> None:
        """Configure LangSmith tracing."""
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", self.config.project_name)

    @property
    def llm(self) -> BaseChatModel:
        """Get the language model instance."""
        if self._llm is None:
            raise RuntimeError("LLM not initialized")
        return self._llm

    @property
    def llm_with_tools(self) -> BaseChatModel:
        """Get the language model with tools bound."""
        if not self._tools:
            return self.llm
        return self.llm.bind_tools(self._tools)

    def register_tools(self, tools: list) -> None:
        """Register tools for the agent to use.

        Args:
            tools: List of tool functions decorated with @tool
        """
        self._tools = tools

    @abstractmethod
    def _build_graph(self) -> StateGraph:
        """Build the agent's workflow graph.

        Subclasses must implement this method to define
        the agent's specific workflow.

        Returns:
            Configured StateGraph instance
        """
        pass

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the agent's system prompt.

        Returns:
            System prompt string
        """
        pass

    def compile(self) -> None:
        """Compile the agent's graph for execution."""
        self._graph = self._build_graph()

        # Use provided checkpointer, or get from factory based on config
        if self.config.checkpointer is not None:
            checkpointer = self.config.checkpointer
        elif self.config.memory_backend == "auto":
            # Auto mode: use factory which reads from environment
            checkpointer = get_checkpointer()
        elif self.config.memory_backend == "postgres":
            from app.agents.memory.checkpointers import (
                CheckpointerConfig,
                CheckpointerBackend,
                create_checkpointer,
            )
            config = CheckpointerConfig(backend=CheckpointerBackend.POSTGRES)
            checkpointer = create_checkpointer(config)
        elif self.config.memory_backend == "sqlite":
            from app.agents.memory.checkpointers import (
                CheckpointerConfig,
                CheckpointerBackend,
                create_checkpointer,
            )
            config = CheckpointerConfig(backend=CheckpointerBackend.SQLITE)
            checkpointer = create_checkpointer(config)
        else:
            checkpointer = MemorySaver()

        self._compiled_graph = self._graph.compile(checkpointer=checkpointer)

    @traceable(name="agent_invoke")
    def invoke(
        self,
        message: str,
        session_id: str | None = None,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Invoke the agent with a message.

        Args:
            message: User message to process
            session_id: Optional session ID for continuity
            user_id: Optional user ID for personalization
            **kwargs: Additional state fields

        Returns:
            Agent response and updated state
        """
        if self._compiled_graph is None:
            self.compile()

        # Build input state
        input_state = {
            "messages": [HumanMessage(content=message)],
            "session_id": session_id,
            "user_id": user_id,
            **kwargs,
        }

        # Configure thread for checkpointing
        config = {"configurable": {"thread_id": session_id or "default"}}

        # Invoke graph
        result = self._compiled_graph.invoke(input_state, config=config)

        return result

    @traceable(name="agent_stream")
    async def astream(
        self,
        message: str,
        session_id: str | None = None,
        user_id: str | None = None,
        **kwargs: Any,
    ):
        """Stream agent responses asynchronously.

        Args:
            message: User message to process
            session_id: Optional session ID for continuity
            user_id: Optional user ID for personalization
            **kwargs: Additional state fields

        Yields:
            Streaming response chunks
        """
        if self._compiled_graph is None:
            self.compile()

        input_state = {
            "messages": [HumanMessage(content=message)],
            "session_id": session_id,
            "user_id": user_id,
            **kwargs,
        }

        config = {"configurable": {"thread_id": session_id or "default"}}

        async for chunk in self._compiled_graph.astream(
            input_state, config=config, stream_mode="values"
        ):
            yield chunk

    def get_last_response(self, result: dict[str, Any]) -> str:
        """Extract the last AI response from result.

        Args:
            result: Result dict from invoke()

        Returns:
            Last AI message content
        """
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return msg.content
        return ""

    def get_state(self, session_id: str) -> dict[str, Any] | None:
        """Get the current state for a session.

        Args:
            session_id: Session identifier

        Returns:
            Current state dict or None if not found
        """
        if self._compiled_graph is None:
            return None

        config = {"configurable": {"thread_id": session_id}}
        snapshot = self._compiled_graph.get_state(config)
        return snapshot.values if snapshot else None

    def clear_session(self, session_id: str) -> None:
        """Clear the state for a session.

        Args:
            session_id: Session identifier to clear
        """
        # For MemorySaver, we can't truly delete, but we can
        # update to empty state. Production should use proper DB.
        pass

    def inject_semantic_context(
        self,
        user_id: str | None = None,
        query: str | None = None,
        top_k: int = 3,
    ) -> str | None:
        """Retrieve relevant context from semantic memory.

        This method retrieves past conversation summaries and context
        that may be relevant to the current conversation.

        Args:
            user_id: User identifier for personalized context.
            query: Optional query to find relevant past context.
            top_k: Number of relevant memories to retrieve.

        Returns:
            Formatted context string or None if semantic memory disabled.
        """
        if not self.config.semantic_memory_enabled:
            return None

        try:
            semantic_memory = get_semantic_memory()
            context_parts = []

            # Get user-specific context
            if user_id:
                user_context = semantic_memory.get_user_context(
                    user_id=user_id,
                    top_k=top_k,
                )
                if user_context:
                    context_parts.append(f"## User History\n{user_context}")

            # Get query-relevant context
            if query:
                results = semantic_memory.search(
                    query=query,
                    top_k=top_k,
                    user_id=user_id,
                )
                if results:
                    relevant = "\n".join([
                        f"- {r.content[:200]}..." if len(r.content) > 200 else f"- {r.content}"
                        for r in results
                    ])
                    context_parts.append(f"## Relevant Past Context\n{relevant}")

            if context_parts:
                return "\n\n".join(context_parts)
            return None

        except Exception as e:
            print(f"Warning: Failed to inject semantic context: {e}")
            return None

    def store_conversation_summary(
        self,
        session_id: str,
        user_id: str | None = None,
        messages: list[BaseMessage] | None = None,
    ) -> bool:
        """Store a conversation summary in semantic memory.

        This method creates a summary of the conversation and stores it
        for future retrieval and context injection.

        Args:
            session_id: Session identifier.
            user_id: Optional user identifier.
            messages: Optional messages to summarize. If None, retrieves from state.

        Returns:
            True if summary was stored successfully.
        """
        if not self.config.semantic_memory_enabled:
            return False

        try:
            # Get messages from state if not provided
            if messages is None:
                state = self.get_state(session_id)
                if state:
                    messages = state.get("messages", [])

            if not messages:
                return False

            # Get summarizer
            summarizer = get_summarizer()

            # Check if summarization is needed
            if not summarizer.should_summarize(messages):
                return False

            # Create summary
            summary = summarizer.summarize(messages)

            if not summary:
                return False

            # Store in semantic memory
            semantic_memory = get_semantic_memory()
            semantic_memory.add_memory(
                content=summary,
                memory_type="conversation_summary",
                session_id=session_id,
                user_id=user_id,
                agent_type=self.__class__.__name__,
                metadata={
                    "message_count": len(messages),
                    "session_id": session_id,
                },
            )
            return True

        except Exception as e:
            print(f"Warning: Failed to store conversation summary: {e}")
            return False

    def apply_conversation_compression(
        self,
        messages: list[BaseMessage],
        keep_recent: int = 3,
    ) -> tuple[list[BaseMessage], str | None]:
        """Apply conversation compression if enabled.

        This method uses the summarizer to compress older messages
        while keeping recent ones intact.

        Args:
            messages: Full list of conversation messages.
            keep_recent: Number of recent messages to keep intact.

        Returns:
            Tuple of (compressed messages, summary if created).
        """
        if not self.config.conversation_summarization:
            return messages, None

        try:
            summarizer = get_summarizer()
            return summarizer.summarize_and_compress(messages, keep_recent)
        except Exception as e:
            print(f"Warning: Conversation compression failed: {e}")
            return messages, None


def create_react_agent_graph(
    agent: BaseAgent,
    state_class: type[BaseModel],
) -> StateGraph:
    """Create a standard ReAct agent graph.

    This helper creates the common ReAct pattern:
    agent -> tools -> agent (loop until done)

    Args:
        agent: BaseAgent instance with LLM and tools
        state_class: State schema class

    Returns:
        Configured StateGraph
    """

    def call_model(state) -> dict:
        """Call the LLM with current messages."""
        system_prompt = SystemMessage(content=agent._get_system_prompt())
        # Handle both Pydantic model and dict state
        msgs = list(state.messages) if hasattr(state, "messages") else state.get("messages", [])
        messages = [system_prompt] + msgs
        response = agent.llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state) -> str:
        """Determine if we should continue to tools or end."""
        # Handle both Pydantic model and dict state
        messages = list(state.messages) if hasattr(state, "messages") else state.get("messages", [])
        if not messages:
            return "end"
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    # Build graph
    graph = StateGraph(state_class)

    # Add nodes
    graph.add_node("agent", call_model)
    if agent._tools:
        graph.add_node("tools", ToolNode(agent._tools))

    # Add edges
    graph.add_edge(START, "agent")

    if agent._tools:
        graph.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", "end": END}
        )
        graph.add_edge("tools", "agent")
    else:
        graph.add_edge("agent", END)

    return graph
