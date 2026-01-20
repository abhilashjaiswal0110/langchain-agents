"""IT Operations Deep Agent - Main Coordinator.

This is the main Deep Agent for IT Managed Services (Atos-style).
It coordinates specialized subagents to handle complex IT operations workflows.
"""

import os
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langsmith import traceable

from app.deepagents.core.types import (
    Todo,
    TodoStatus,
)
from app.deepagents.core.state import DeepAgentState
from app.deepagents.subagents.definitions import (
    get_all_subagents,
    get_subagent_tools,
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
    # Document tools for attachment handling
    search_attachments,
    list_attachments,
    get_attachment_summary,
    clear_attachments,
    get_document_context,
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

## Document Attachments
Users can upload documents (PDF, Word, TXT, PPT, images) for context.
When documents are uploaded, use these tools:
- `search_attachments`: Search document content using semantic similarity
- `list_attachments`: See all uploaded documents
- `get_attachment_summary`: Get overview of a document

**IMPORTANT**: When users ask about uploaded documents, ALWAYS use `search_attachments`
to find relevant content. Never answer from memory - search the documents first.

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
        """Create LLM instance.

        Note: OpenAI reasoning models (o1, o3-mini, o4-mini) do not support
        the temperature parameter - they only accept the default value of 1.
        """
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
            # Determine actual model name
            actual_model = model_name or "gpt-4o-mini"

            # OpenAI reasoning models (o1, o3, o4 series) don't support temperature
            # They only accept the default value of 1
            reasoning_models = ("o1", "o3", "o4", "o1-mini", "o3-mini", "o4-mini")
            is_reasoning_model = any(
                actual_model.startswith(prefix) for prefix in reasoning_models
            )

            if is_reasoning_model:
                print(f"[DEBUG] Using reasoning model {actual_model} (temperature not supported)")
                return ChatOpenAI(model=actual_model)
            else:
                return ChatOpenAI(
                    model=actual_model,
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

        # Document attachment tools
        doc_tools = [
            search_attachments,
            list_attachments,
            get_attachment_summary,
            clear_attachments,
        ]

        return planning_tools + fs_tools + task_tools + it_tools + doc_tools

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
            storage.save_file(session_id, path, content)
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
                    tool_results = {}
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        tool_call_id = tool_call["id"]

                        # Find and execute the tool
                        for t in tools:
                            if t.name == tool_name:
                                result = t.invoke(tool_args)
                                tool_results[tool_call_id] = result
                                break

                    if tool_results:
                        # Get final response with tool results
                        messages.append(response)
                        from langchain_core.messages import ToolMessage
                        for tc in response.tool_calls:
                            if tc["id"] in tool_results:
                                messages.append(ToolMessage(
                                    content=str(tool_results[tc["id"]]),
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

        # Build dynamic system prompt with document context
        system_prompt = IT_OPERATIONS_SYSTEM_PROMPT

        # Inject document context if documents are uploaded
        session_id = state.session_id or "default"
        doc_context = get_document_context(session_id)
        if doc_context:
            system_prompt += "\n" + doc_context

        # Ensure system message is first
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages
        else:
            # Update existing system message with document context
            messages[0] = SystemMessage(content=system_prompt)

        response = self.llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "iteration_count": state.iteration_count + 1,
            "last_activity": datetime.now(),
        }

    def _should_continue(self, state: DeepAgentState) -> Literal["continue", "end"]:
        """Determine if we should continue to tools or end."""
        if state.iteration_count > 40:  # Safety limit (below recursion_limit of 50)
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

        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": 50,  # Allow complex multi-tool workflows
        }

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

        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": 50,  # Allow complex multi-tool workflows
        }

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

        # Get updated context
        files = self.storage.list_files(session_id)
        todos = self.storage.get_todos(session_id)

        return {
            "response": response_text,
            "session_id": session_id,
            "todos": [t.model_dump() for t in todos],
            "files": files,
            "tool_calls": getattr(last_message, "tool_calls", []),
        }

    async def astream_chat(
        self,
        message: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream chat responses with thinking and progress updates.

        Yields events for:
        - thinking: Agent's reasoning process (collapsible in UI)
        - tool_start: When a tool is being called
        - tool_result: Tool execution result
        - todo_update: Todo list changes
        - token: Streaming tokens
        - complete: Final response

        The thinking event includes:
        - iteration: Step number in the agent loop
        - phase: Current processing phase (planning/analyzing/executing/summarizing)
        - content: Full reasoning content (for expanded view)
        - summary: Brief summary (for collapsed view)
        - is_reasoning_token: True if from extended thinking models (Claude/o1)

        Args:
            message: User message.
            session_id: Session identifier.
            user_id: User identifier.

        Yields:
            Event dictionaries with type and data.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": 50,  # Allow complex multi-tool workflows
        }

        # Initial event
        yield {
            "type": "start",
            "data": {
                "session_id": session_id,
                "message": "Processing your request...",
            },
        }

        iteration = 0
        final_response = ""
        accumulated_tokens = ""
        current_phase = "planning"
        thinking_buffer = ""  # Buffer for accumulating thinking content

        # Phase detection based on iteration and tool usage
        def get_phase(iter_num: int, has_tool_calls: bool, tool_name: str | None = None) -> str:
            if iter_num == 1:
                return "planning"
            if tool_name:
                if tool_name in ("write_todos", "update_todo"):
                    return "planning"
                elif tool_name in ("search_incidents", "search_changes", "search_cmdb", "search_knowledge_base", "search_attachments"):
                    return "analyzing"
                else:
                    return "executing"
            if has_tool_calls:
                return "executing"
            return "summarizing"

        try:
            async for event in self.graph.astream_events(
                {
                    "messages": [HumanMessage(content=message)],
                    "session_id": session_id,
                    "user_id": user_id,
                },
                config=config,
                version="v2",
            ):
                event_type = event.get("event")
                event_name = event.get("name", "")
                event_data = event.get("data", {})

                # Agent thinking/reasoning - iteration start
                if event_type == "on_chat_model_start":
                    iteration += 1
                    current_phase = get_phase(iteration, False)
                    thinking_buffer = ""  # Reset thinking buffer for new iteration

                    phase_messages = {
                        "planning": "Planning approach...",
                        "analyzing": "Analyzing data...",
                        "executing": "Executing actions...",
                        "summarizing": "Preparing response...",
                    }

                    yield {
                        "type": "thinking",
                        "data": {
                            "iteration": iteration,
                            "phase": current_phase,
                            "content": "",  # Content will be streamed
                            "summary": phase_messages.get(current_phase, f"Step {iteration}..."),
                            "is_reasoning_token": False,
                        },
                    }

                # Streaming tokens from LLM
                elif event_type == "on_chat_model_stream":
                    chunk = event_data.get("chunk")
                    if chunk:
                        # Check for extended thinking content (Claude with thinking enabled)
                        # The content may be a list of blocks for Claude models
                        if hasattr(chunk, "content"):
                            content = chunk.content

                            # Handle Claude extended thinking blocks
                            if isinstance(content, list):
                                for block in content:
                                    if isinstance(block, dict):
                                        block_type = block.get("type", "")
                                        if block_type == "thinking":
                                            # Extended thinking content from Claude
                                            thinking_text = block.get("thinking", "")
                                            if thinking_text:
                                                thinking_buffer += thinking_text
                                                yield {
                                                    "type": "thinking",
                                                    "data": {
                                                        "iteration": iteration,
                                                        "phase": current_phase,
                                                        "content": thinking_text,
                                                        "summary": self._summarize_thinking(thinking_text),
                                                        "is_reasoning_token": True,
                                                    },
                                                }
                                        elif block_type == "text":
                                            # Regular text content
                                            text = block.get("text", "")
                                            if text:
                                                accumulated_tokens += text
                                                yield {
                                                    "type": "token",
                                                    "data": {"content": text},
                                                }
                                    elif hasattr(block, "type"):
                                        # Handle Pydantic-style content blocks
                                        if block.type == "thinking":
                                            thinking_text = getattr(block, "thinking", "")
                                            if thinking_text:
                                                thinking_buffer += thinking_text
                                                yield {
                                                    "type": "thinking",
                                                    "data": {
                                                        "iteration": iteration,
                                                        "phase": current_phase,
                                                        "content": thinking_text,
                                                        "summary": self._summarize_thinking(thinking_text),
                                                        "is_reasoning_token": True,
                                                    },
                                                }
                                        elif block.type == "text":
                                            text = getattr(block, "text", "")
                                            if text:
                                                accumulated_tokens += text
                                                yield {
                                                    "type": "token",
                                                    "data": {"content": text},
                                                }
                            elif isinstance(content, str) and content:
                                # Regular string content
                                accumulated_tokens += content
                                yield {
                                    "type": "token",
                                    "data": {"content": content},
                                }

                        # Check for OpenAI reasoning tokens (o1/o3/o4 models)
                        # These may appear in additional_kwargs or as special fields
                        if hasattr(chunk, "additional_kwargs"):
                            additional = chunk.additional_kwargs
                            if additional:
                                # OpenAI o1/o3/o4 reasoning tokens
                                reasoning = additional.get("reasoning_content") or additional.get("reasoning")
                                if reasoning:
                                    thinking_buffer += reasoning
                                    yield {
                                        "type": "thinking",
                                        "data": {
                                            "iteration": iteration,
                                            "phase": current_phase,
                                            "content": reasoning,
                                            "summary": self._summarize_thinking(reasoning),
                                            "is_reasoning_token": True,
                                        },
                                    }

                # Tool call initiated
                elif event_type == "on_tool_start":
                    tool_name = event_name
                    tool_input = event_data.get("input", {})

                    # Update phase based on tool type
                    current_phase = get_phase(iteration, True, tool_name)

                    # Format tool description for user
                    tool_desc = self._get_tool_description(tool_name, tool_input)

                    # Safely serialize tool_input (may contain non-serializable objects)
                    safe_input = {}
                    if isinstance(tool_input, dict):
                        for k, v in tool_input.items():
                            try:
                                if hasattr(v, "model_dump"):
                                    safe_input[k] = v.model_dump()
                                elif hasattr(v, "isoformat"):
                                    safe_input[k] = v.isoformat()
                                else:
                                    safe_input[k] = str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v
                            except Exception:
                                safe_input[k] = str(v)
                    else:
                        safe_input = str(tool_input)

                    yield {
                        "type": "tool_start",
                        "data": {
                            "tool": tool_name,
                            "input": safe_input,
                            "description": tool_desc,
                            "phase": current_phase,
                        },
                    }

                # Tool execution complete
                elif event_type == "on_tool_end":
                    tool_name = event_name
                    output = event_data.get("output", "")

                    yield {
                        "type": "tool_result",
                        "data": {
                            "tool": tool_name,
                            "result": str(output)[:500],  # Truncate long results
                        },
                    }

                    # Update todos after any todo-related tool
                    if tool_name in ("write_todos", "update_todo"):
                        try:
                            todos = self.storage.get_todos(session_id)
                            yield {
                                "type": "todo_update",
                                "data": {
                                    "todos": [t.model_dump(mode="json") for t in todos],
                                    "action": "updated",
                                },
                            }
                        except Exception as todo_err:
                            print(f"[WARN] Failed to serialize todos: {todo_err}")

                # Chain/Graph completion
                elif event_type == "on_chain_end" and event_name == "LangGraph":
                    output = event_data.get("output", {})
                    messages = output.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        if hasattr(last_msg, "content"):
                            final_response = last_msg.content

            # Get final context
            try:
                files = self.storage.list_files(session_id)
            except Exception as file_err:
                print(f"[WARN] Failed to list files: {file_err}")
                files = []

            try:
                todos = self.storage.get_todos(session_id)
                todo_list = [t.model_dump(mode="json") for t in todos]
            except Exception as todo_err:
                print(f"[WARN] Failed to get todos: {todo_err}")
                todo_list = []

            # Final complete event
            yield {
                "type": "complete",
                "data": {
                    "response": final_response,
                    "session_id": session_id,
                    "todos": todo_list,
                    "files": files,
                    "iterations": iteration,
                },
            }

        except Exception as e:
            import traceback
            print(f"[ERROR] Stream error: {e}")
            traceback.print_exc()
            yield {
                "type": "error",
                "data": {
                    "error": str(e),
                    "session_id": session_id,
                },
            }

    def _get_tool_description(self, tool_name: str, tool_input: dict) -> str:
        """Generate human-readable description for tool execution."""
        descriptions = {
            "write_todos": "Creating task plan...",
            "update_todo": f"Updating task status...",
            "write_file": f"Saving to {tool_input.get('path', 'file')}...",
            "read_file": f"Reading {tool_input.get('path', 'file')}...",
            "ls": "Listing workspace files...",
            "task": f"Delegating to {tool_input.get('subagent_type', 'subagent')}...",
            "search_incidents": "Searching incidents...",
            "get_incident_details": f"Getting incident {tool_input.get('incident_id', '')}...",
            "create_incident": "Creating new incident...",
            "search_changes": "Searching change requests...",
            "search_problems": "Searching problems...",
            "search_cmdb": "Querying CMDB...",
            "get_sla_status": "Checking SLA status...",
            "search_knowledge_base": "Searching knowledge base...",
            # Document tools
            "search_attachments": f"Searching documents for '{tool_input.get('query', '')[:30]}...'",
            "list_attachments": "Listing uploaded documents...",
            "get_attachment_summary": "Getting document summary...",
            "clear_attachments": "Clearing documents...",
        }
        return descriptions.get(tool_name, f"Executing {tool_name}...")

    def _summarize_thinking(self, thinking_content: str, max_length: int = 60) -> str:
        """Summarize thinking content for collapsed display.

        Args:
            thinking_content: Full thinking/reasoning content.
            max_length: Maximum length for summary.

        Returns:
            Brief summary suitable for collapsed view.
        """
        if not thinking_content:
            return "Thinking..."

        # Clean and truncate
        content = thinking_content.strip()

        # Try to get first meaningful sentence or phrase
        # Look for sentence boundaries
        for sep in [". ", ".\n", "! ", "? ", "\n\n"]:
            if sep in content:
                first_part = content.split(sep)[0]
                if len(first_part) >= 10:  # Meaningful length
                    content = first_part
                    break

        # Truncate if still too long
        if len(content) > max_length:
            content = content[:max_length - 3] + "..."

        return content

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
