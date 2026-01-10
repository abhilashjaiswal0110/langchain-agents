"""IT Operations Deep Agent - Main Coordinator.

This is the main Deep Agent for IT Managed Services (Atos-style).
It coordinates specialized subagents to handle complex IT operations workflows.
"""

import os
import uuid
from datetime import datetime
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langsmith import traceable

from app.deepagents.core.types import (
    DeepAgentConfig,
    Todo,
    TodoStatus,
    FileEntry,
    SubAgentResult,
)
from app.deepagents.core.state import DeepAgentState
from app.deepagents.subagents.definitions import (
    get_all_subagents,
    get_subagent_tools,
    INCIDENT_AGENT,
    CHANGE_AGENT,
    PROBLEM_AGENT,
    ASSET_AGENT,
    SLA_AGENT,
    KNOWLEDGE_AGENT,
)
from app.deepagents.tools import (
    # All IT Operations tools
    search_incidents,
    get_incident_details,
    create_incident,
    update_incident,
    escalate_incident,
    search_changes,
    get_change_details,
    validate_change,
    assess_change_risk,
    search_problems,
    get_problem_details,
    create_problem,
    link_incidents_to_problem,
    create_known_error,
    search_cmdb,
    get_ci_details,
    get_ci_relationships,
    get_affected_services,
    get_sla_status,
    calculate_sla_breach_time,
    get_sla_report,
    predict_sla_breach,
    search_knowledge_base,
    get_kb_article,
    create_kb_article,
    suggest_kb_articles,
)
from app.deepagents.storage.persistent_backend import PersistentStorage


IT_OPERATIONS_SYSTEM_PROMPT = """You are an IT Operations Deep Agent - an advanced AI coordinator for IT Managed Services.

## Your Role
You coordinate complex IT operations by:
1. Breaking down tasks into actionable steps (planning)
2. Delegating specialized work to expert subagents
3. Managing context through file system tools
4. Tracking progress and ensuring completion

## Available Subagents
Use the `task` tool to delegate to specialized subagents:

- **incident-manager**: Incident lifecycle management
- **change-manager**: Change request validation and risk assessment
- **problem-manager**: Root cause analysis and known error management
- **asset-manager**: CMDB queries and impact analysis
- **sla-monitor**: SLA tracking and breach prediction
- **knowledge-manager**: Knowledge base operations

## Planning Guidelines
For complex requests, ALWAYS start with `write_todos` to create a task plan:
1. Identify all components of the request
2. Determine which subagents/tools are needed
3. Create ordered todo items
4. Execute and track progress

## Context Management
Use file system tools to:
- `write_file`: Save investigation notes, findings, reports
- `read_file`: Retrieve saved context
- `ls`: List available context files

## Integration
You are connected to ServiceNow for real ITSM data (when in live mode).
All actions are traceable via LangSmith.

## Response Format
- Provide clear, actionable responses
- Reference ticket numbers and CI names
- Summarize subagent findings
- Recommend next steps when appropriate

## Example Workflows

### Major Incident
1. Create incident record
2. Query CMDB for affected systems
3. Check for related problems/known errors
4. Search knowledge base for solutions
5. Track SLA and escalate if needed
6. Document resolution steps

### Change Validation
1. Review change request details
2. Assess risk factors
3. Check CI relationships for impact
4. Verify SLA implications
5. Provide recommendation

### Problem Investigation
1. Search for related incidents
2. Query affected CIs
3. Perform root cause analysis
4. Create problem record
5. Document known error if applicable
"""


class ITOperationsDeepAgent:
    """IT Operations Deep Agent for managed services.

    Coordinates specialized subagents to handle complex IT workflows
    including incident management, change management, problem management,
    asset management, SLA monitoring, and knowledge management.
    """

    def __init__(
        self,
        model_provider: Literal["openai", "anthropic", "auto"] = "auto",
        model_name: str | None = None,
        temperature: float = 0,
        storage_path: str = "./data/deepagent_context",
    ) -> None:
        """Initialize IT Operations Deep Agent.

        Args:
            model_provider: LLM provider to use.
            model_name: Specific model name.
            temperature: LLM temperature.
            storage_path: Path for persistent context storage.
        """
        self.model_provider = model_provider
        self.temperature = temperature
        self.storage = PersistentStorage(base_path=storage_path)

        # Initialize LLM
        self.llm = self._create_llm(model_provider, model_name, temperature)

        # Collect all tools
        self.tools = self._collect_tools()

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Initialize checkpointer
        self.checkpointer = MemorySaver()

        # Build the graph
        self.graph = self._build_graph()

    def _create_llm(
        self,
        provider: str,
        model_name: str | None,
        temperature: float,
    ) -> ChatOpenAI | ChatAnthropic:
        """Create LLM instance."""
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
                model=model_name or "claude-sonnet-4-20250514",
                temperature=temperature,
            )
        else:
            return ChatOpenAI(
                model=model_name or "gpt-4o-mini",
                temperature=temperature,
            )

    def _collect_tools(self) -> list:
        """Collect all tools for the agent."""
        # Planning tools
        planning_tools = [
            self._create_write_todos_tool(),
            self._create_update_todo_tool(),
        ]

        # File system tools
        fs_tools = [
            self._create_write_file_tool(),
            self._create_read_file_tool(),
            self._create_list_files_tool(),
        ]

        # Subagent task tool
        task_tools = [self._create_task_tool()]

        # IT Operations tools (direct access)
        it_tools = [
            search_incidents,
            get_incident_details,
            create_incident,
            update_incident,
            escalate_incident,
            search_changes,
            get_change_details,
            validate_change,
            assess_change_risk,
            search_problems,
            get_problem_details,
            create_problem,
            link_incidents_to_problem,
            create_known_error,
            search_cmdb,
            get_ci_details,
            get_ci_relationships,
            get_affected_services,
            get_sla_status,
            calculate_sla_breach_time,
            get_sla_report,
            predict_sla_breach,
            search_knowledge_base,
            get_kb_article,
            create_kb_article,
            suggest_kb_articles,
        ]

        return planning_tools + fs_tools + task_tools + it_tools

    def _create_write_todos_tool(self):
        """Create the write_todos planning tool."""
        storage = self.storage

        @tool
        def write_todos(todos: list[dict[str, Any]], session_id: str = "default") -> str:
            """Create or update the task plan for current work.

            Break down complex tasks into discrete steps for tracking.

            Args:
                todos: List of todo items with 'content' and optional 'priority' (0-2).
                session_id: Session identifier.

            Returns:
                Formatted task plan.
            """
            created_todos = []
            for item in todos:
                todo = Todo(
                    id=str(uuid.uuid4())[:8],
                    content=item.get("content", ""),
                    priority=int(item.get("priority", 0)),
                    status=TodoStatus.PENDING,
                )
                created_todos.append(todo)

            # Save to storage
            storage.save_todos(session_id, created_todos)

            lines = ["**Task Plan Created:**\n"]
            for i, todo in enumerate(created_todos, 1):
                priority_marker = "!" * todo.priority if todo.priority > 0 else ""
                lines.append(f"{i}. [ ] {priority_marker}{todo.content}")

            lines.append(f"\nTotal: {len(created_todos)} tasks")
            return "\n".join(lines)

        return write_todos

    def _create_update_todo_tool(self):
        """Create the update_todo tool."""

        @tool
        def update_todo(
            todo_index: int,
            status: Literal["pending", "in_progress", "completed", "blocked"],
            notes: str | None = None,
        ) -> str:
            """Update the status of a todo item.

            Args:
                todo_index: The index (1-based) of the todo to update.
                status: New status.
                notes: Optional notes about the update.

            Returns:
                Confirmation of the update.
            """
            status_symbols = {
                "pending": "[ ]",
                "in_progress": "[~]",
                "completed": "[x]",
                "blocked": "[!]",
            }

            result = f"Task {todo_index} marked as {status} {status_symbols[status]}"
            if notes:
                result += f"\nNotes: {notes}"

            return result

        return update_todo

    def _create_write_file_tool(self):
        """Create the write_file tool."""
        storage = self.storage

        @tool
        def write_file(
            path: str,
            content: str,
            session_id: str = "default",
        ) -> str:
            """Write content to a context file.

            Use this to save investigation notes, reports, or findings.

            Args:
                path: File path (e.g., /notes/incident_investigation.md).
                content: Content to write.
                session_id: Session identifier.

            Returns:
                Confirmation message.
            """
            entry = storage.save_file(session_id, path, content)
            return f"File saved: {path} ({len(content)} characters)"

        return write_file

    def _create_read_file_tool(self):
        """Create the read_file tool."""
        storage = self.storage

        @tool
        def read_file(path: str, session_id: str = "default") -> str:
            """Read a context file.

            Args:
                path: File path to read.
                session_id: Session identifier.

            Returns:
                File contents or error message.
            """
            entry = storage.read_file(session_id, path)
            if entry:
                return f"**File: {path}**\n\n{entry.content}"
            return f"File not found: {path}"

        return read_file

    def _create_list_files_tool(self):
        """Create the ls (list files) tool."""
        storage = self.storage

        @tool
        def ls(directory: str = "/", session_id: str = "default") -> str:
            """List context files.

            Args:
                directory: Directory to list.
                session_id: Session identifier.

            Returns:
                List of files.
            """
            files = storage.list_files(session_id, directory)
            if files:
                return f"**Files in {directory}:**\n" + "\n".join(f"- {f}" for f in files)
            return f"No files in {directory}"

        return ls

    def _create_task_tool(self):
        """Create the task tool for subagent delegation."""
        llm = self.llm
        subagents = {s.name: s for s in get_all_subagents()}

        @tool
        def task(
            subagent_type: str,
            task_description: str,
            context: str | None = None,
        ) -> str:
            """Delegate a task to a specialized subagent.

            Available subagents:
            - incident-manager: Incident management
            - change-manager: Change request handling
            - problem-manager: Root cause analysis
            - asset-manager: CMDB/CI operations
            - sla-monitor: SLA tracking
            - knowledge-manager: KB operations

            Args:
                subagent_type: Type of subagent to use.
                task_description: What the subagent should do.
                context: Optional context to provide.

            Returns:
                Result from the subagent.
            """
            if subagent_type not in subagents:
                available = ", ".join(subagents.keys())
                return f"Unknown subagent: {subagent_type}. Available: {available}"

            subagent_def = subagents[subagent_type]
            tools = get_subagent_tools(subagent_type)

            if not tools:
                return f"No tools available for subagent: {subagent_type}"

            # Create subagent LLM with tools
            subagent_llm = llm.bind_tools(tools)

            # Build prompt
            messages = [
                SystemMessage(content=subagent_def.system_prompt),
            ]

            if context:
                messages.append(HumanMessage(content=f"Context:\n{context}\n\nTask: {task_description}"))
            else:
                messages.append(HumanMessage(content=task_description))

            # Execute subagent (simplified - single turn)
            try:
                response = subagent_llm.invoke(messages)

                # If tool calls, execute them
                if hasattr(response, "tool_calls") and response.tool_calls:
                    tool_results = []
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]

                        # Find and execute the tool
                        for t in tools:
                            if t.name == tool_name:
                                result = t.invoke(tool_args)
                                tool_results.append(f"[{tool_name}]: {result}")
                                break

                    if tool_results:
                        # Get final response with tool results
                        messages.append(response)
                        for i, tc in enumerate(response.tool_calls):
                            from langchain_core.messages import ToolMessage
                            messages.append(ToolMessage(
                                content=tool_results[i].split("]: ", 1)[1],
                                tool_call_id=tc["id"],
                            ))

                        final_response = llm.invoke(messages)
                        return f"**[Subagent: {subagent_type}]**\n\n{final_response.content}"

                return f"**[Subagent: {subagent_type}]**\n\n{response.content}"

            except Exception as e:
                return f"Subagent error: {e}"

        return task

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        graph = StateGraph(DeepAgentState)

        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", ToolNode(self.tools))

        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": END},
        )
        graph.add_edge("tools", "agent")

        return graph.compile(checkpointer=self.checkpointer)

    def _agent_node(self, state: DeepAgentState) -> dict:
        """Process messages and decide on actions."""
        messages = list(state.messages)

        # Ensure system message is first
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=IT_OPERATIONS_SYSTEM_PROMPT)] + messages

        response = self.llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "iteration_count": state.iteration_count + 1,
            "last_activity": datetime.now(),
        }

    def _should_continue(self, state: DeepAgentState) -> Literal["continue", "end"]:
        """Determine if we should continue to tools or end."""
        if state.iteration_count > 30:  # Safety limit
            return "end"

        last_message = state.messages[-1] if state.messages else None

        if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"

        return "end"

    @traceable(name="it_operations_chat", tags=["deep_agent", "it_operations"])
    def chat(
        self,
        message: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Process a chat message.

        Args:
            message: User message.
            session_id: Session identifier.
            user_id: User identifier.

        Returns:
            Response dictionary.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        config = {"configurable": {"thread_id": session_id}}

        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=message)],
                "session_id": session_id,
                "user_id": user_id,
            },
            config=config,
        )

        last_message = result["messages"][-1] if result.get("messages") else None
        response_text = last_message.content if last_message else ""

        # Get saved context
        files = self.storage.list_files(session_id)
        todos = self.storage.get_todos(session_id)

        return {
            "response": response_text,
            "session_id": session_id,
            "todos": [t.model_dump() for t in todos],
            "files": files,
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
            "tool_calls": getattr(last_message, "tool_calls", []),
        }

    def get_session_context(self, session_id: str) -> dict[str, Any]:
        """Get context for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Dictionary with todos, files, and metadata.
        """
        return {
            "todos": [t.model_dump() for t in self.storage.get_todos(session_id)],
            "files": self.storage.list_files(session_id),
            "metadata": self.storage.get_session_metadata(session_id),
        }


def create_it_operations_agent(
    model_provider: Literal["openai", "anthropic", "auto"] = "auto",
    model_name: str | None = None,
    **kwargs,
) -> ITOperationsDeepAgent:
    """Factory function to create IT Operations Deep Agent.

    Args:
        model_provider: LLM provider.
        model_name: Specific model name.
        **kwargs: Additional configuration.

    Returns:
        Configured ITOperationsDeepAgent instance.
    """
    return ITOperationsDeepAgent(
        model_provider=model_provider,
        model_name=model_name,
        **kwargs,
    )


# LangGraph Studio entry point
def get_graph():
    """Entry point for LangGraph Studio."""
    agent = ITOperationsDeepAgent(model_provider="auto")
    return agent.graph
