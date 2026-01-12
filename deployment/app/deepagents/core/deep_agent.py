"""Deep Agent Core Implementation.

This module provides the main DeepAgent class that combines LangGraph
with middleware for planning, file system, and subagent capabilities.
"""

import os
import uuid
from datetime import datetime
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langsmith import traceable

from app.deepagents.core.types import (
    DeepAgentConfig,
    SubAgentDefinition,
    Todo,
    TodoStatus,
    FileEntry,
    SubAgentResult,
)
from app.deepagents.core.state import DeepAgentState
from app.deepagents.core.middleware import (
    TodoListMiddleware,
    FilesystemMiddleware,
    SubAgentMiddleware,
)


class DeepAgent:
    """Deep Agent with planning, file system, and subagent capabilities.

    Inspired by LangChain's deepagents library, this agent can:
    - Break down complex tasks into discrete steps (TodoList)
    - Manage large context through file system tools
    - Delegate work to specialized subagents
    - Persist memory across conversations
    """

    def __init__(
        self,
        config: DeepAgentConfig,
        tools: list | None = None,
        subagents: list[SubAgentDefinition] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize Deep Agent.

        Args:
            config: Agent configuration.
            tools: Additional tools for the agent.
            subagents: Specialized subagent definitions.
            system_prompt: Custom system prompt.
        """
        self.config = config
        self.custom_tools = tools or []
        self.system_prompt = system_prompt

        # Initialize LLM
        self.llm = self._create_llm()

        # Initialize middleware
        self.todo_middleware = TodoListMiddleware(max_todos=config.max_todos)
        self.filesystem_middleware = FilesystemMiddleware(
            workspace_path=config.workspace_path,
            max_file_size=config.max_file_size,
            persistent=config.persistent_storage,
        )
        self.subagent_middleware = SubAgentMiddleware(
            subagents=subagents,
            default_model=config.model,
            max_concurrent=config.max_subagents,
        )

        # Collect all tools
        self.tools = self._collect_tools()

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Initialize checkpointer for memory
        self.checkpointer = MemorySaver()

        # Build the graph
        self.graph = self._build_graph()

        # Internal state for file system
        self._files: dict[str, FileEntry] = {}
        self._todos: list[Todo] = []

    def _create_llm(self) -> ChatOpenAI | ChatAnthropic:
        """Create LLM instance based on configuration."""
        provider = self.config.model_provider
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

        if provider == "auto":
            if has_openai:
                provider = "openai"
            elif has_anthropic:
                provider = "anthropic"
            else:
                raise ValueError("No LLM API key found.")

        if provider == "anthropic":
            return ChatAnthropic(
                model=self.config.model if "claude" in self.config.model else "claude-sonnet-4-20250514",
                temperature=self.config.temperature,
            )
        else:
            return ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
            )

    def _collect_tools(self) -> list:
        """Collect all tools from middleware and custom tools."""
        tools = []

        # Add middleware tools
        tools.extend(self.todo_middleware.get_tools())
        tools.extend(self.filesystem_middleware.get_tools())
        tools.extend(self.subagent_middleware.get_tools())

        # Add custom tools
        tools.extend(self.custom_tools)

        return tools

    def _get_system_prompt(self) -> str:
        """Generate the system prompt."""
        if self.system_prompt:
            return self.system_prompt

        return f"""You are a Deep Agent - an AI assistant capable of handling complex, multi-step tasks.

## Your Capabilities

### 1. Planning (TodoList)
Use the `write_todos` tool to break down complex tasks into discrete steps.
Track progress with `update_todo` and view tasks with `get_todos`.

### 2. Context Management (File System)
Use file system tools to manage large context:
- `ls`: List files in workspace
- `read_file`: Read saved context
- `write_file`: Save notes, reports, or intermediate results
- `edit_file`: Update existing files

### 3. Subagent Delegation
Use the `task` tool to delegate specialized work to subagents.
Available subagents provide focused expertise for specific domains.

## Guidelines

1. For complex tasks, ALWAYS start by creating a task plan with `write_todos`
2. Save important findings to files for later reference
3. Delegate specialized subtasks to appropriate subagents
4. Update todo status as you complete each step
5. Provide clear, actionable responses

Agent Name: {self.config.name}
Model: {self.config.model}
"""

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        graph = StateGraph(DeepAgentState)

        # Add nodes
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._create_tool_node())
        graph.add_node("process_tools", self._process_tool_results)

        # Add edges
        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": END},
        )
        graph.add_edge("tools", "process_tools")
        graph.add_edge("process_tools", "agent")

        return graph.compile(checkpointer=self.checkpointer)

    def _create_tool_node(self) -> ToolNode:
        """Create the tool execution node."""
        return ToolNode(self.tools)

    def _agent_node(self, state: DeepAgentState) -> dict:
        """Process messages and decide on actions."""
        messages = list(state.messages)

        # Ensure system message is first
        if not messages or not isinstance(messages[0], SystemMessage):
            system_msg = SystemMessage(content=self._get_system_prompt())
            messages = [system_msg] + messages

        # Add context about current state
        context_parts = []
        if state.todos:
            context_parts.append(f"\n[Todo Status: {state.get_todo_summary()}]")
        if state.files:
            context_parts.append(f"\n[Files: {', '.join(state.get_file_list())}]")
        if state.current_incident:
            context_parts.append(f"\n[Current Incident: {state.current_incident}]")

        if context_parts and messages:
            # Append context to the last human message
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], HumanMessage):
                    context_info = "\n".join(context_parts)
                    # Don't modify original, create new message
                    break

        # Call LLM
        response = self.llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "iteration_count": state.iteration_count + 1,
            "last_activity": datetime.now(),
        }

    def _process_tool_results(self, state: DeepAgentState) -> dict:
        """Process tool results and update state."""
        updates: dict[str, Any] = {
            "total_tool_calls": state.total_tool_calls + 1,
        }

        # Check for file operations in recent messages
        # This would be enhanced to actually process tool results

        return updates

    def _should_continue(self, state: DeepAgentState) -> Literal["continue", "end"]:
        """Determine if we should continue to tools or end."""
        if state.iteration_count > 50:  # Safety limit
            return "end"

        last_message = state.messages[-1] if state.messages else None

        if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"

        return "end"

    @traceable(name="deep_agent_chat", tags=["deep_agent"])
    def chat(
        self,
        message: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Process a chat message.

        Args:
            message: User message.
            session_id: Session identifier for memory.
            user_id: User identifier.

        Returns:
            Response dictionary.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        config = {"configurable": {"thread_id": session_id}}

        # Invoke graph
        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=message)],
                "session_id": session_id,
                "user_id": user_id,
            },
            config=config,
        )

        # Extract response
        last_message = result["messages"][-1] if result.get("messages") else None
        response_text = last_message.content if last_message else ""

        return {
            "response": response_text,
            "session_id": session_id,
            "todos": [t.model_dump() for t in result.get("todos", [])],
            "files": list(result.get("files", {}).keys()),
            "tool_calls": getattr(last_message, "tool_calls", []),
            "iteration_count": result.get("iteration_count", 0),
        }

    async def achat(
        self,
        message: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Async version of chat."""
        if session_id is None:
            session_id = str(uuid.uuid4())

        config = {"configurable": {"thread_id": session_id}}

        result = await self.graph.ainvoke(
            {
                "messages": [HumanMessage(content=message)],
                "session_id": session_id,
                "user_id": user_id,
            },
            config=config,
        )

        last_message = result["messages"][-1] if result.get("messages") else None
        response_text = last_message.content if last_message else ""

        return {
            "response": response_text,
            "session_id": session_id,
            "todos": [t.model_dump() for t in result.get("todos", [])],
            "files": list(result.get("files", {}).keys()),
            "tool_calls": getattr(last_message, "tool_calls", []),
        }

    def get_todos(self, session_id: str) -> list[dict]:
        """Get todos for a session."""
        config = {"configurable": {"thread_id": session_id}}
        state = self.graph.get_state(config)
        if state and state.values:
            return [t.model_dump() for t in state.values.get("todos", [])]
        return []

    def get_files(self, session_id: str) -> list[str]:
        """Get file list for a session."""
        config = {"configurable": {"thread_id": session_id}}
        state = self.graph.get_state(config)
        if state and state.values:
            return list(state.values.get("files", {}).keys())
        return []


def create_deep_agent(
    name: str = "deep_agent",
    model: str = "gpt-4o-mini",
    model_provider: Literal["openai", "anthropic", "auto"] = "auto",
    tools: list | None = None,
    subagents: list[SubAgentDefinition] | None = None,
    system_prompt: str | None = None,
    **kwargs,
) -> DeepAgent:
    """Factory function to create a Deep Agent.

    Args:
        name: Agent name.
        model: LLM model to use.
        model_provider: LLM provider.
        tools: Additional tools.
        subagents: Specialized subagent definitions.
        system_prompt: Custom system prompt.
        **kwargs: Additional config options.

    Returns:
        Configured DeepAgent instance.
    """
    config = DeepAgentConfig(
        name=name,
        model=model,
        model_provider=model_provider,
        **kwargs,
    )

    return DeepAgent(
        config=config,
        tools=tools,
        subagents=subagents,
        system_prompt=system_prompt,
    )
