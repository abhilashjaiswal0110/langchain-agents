"""Sales & Pre-Sales Intelligence Deep Agent - Main Coordinator.

This is the main Deep Agent for Sales and Pre-Sales operations.
It coordinates specialized subagents to handle complex sales workflows
including deal qualification, RFP responses, pricing, and competitive strategy.
"""

import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langsmith import traceable

from app.agents.base.llm_factory import get_llm
from app.deepagents.core.state import DeepAgentState
from app.deepagents.core.types import (
    Todo,
    TodoStatus,
)
from app.deepagents.storage.persistent_backend import PersistentStorage
from app.deepagents.subagents.sales_subagents import (
    get_all_sales_subagents,
)
from app.deepagents.tools.analytics_tools import (
    assess_deal_risk,
    calculate_win_probability,
    get_sales_performance_summary,
    get_similar_deals,
)
from app.deepagents.tools.competitor_tools import (
    compare_solutions,
    get_competitive_analysis,
    get_objection_handler,
    suggest_differentiators,
)
from app.deepagents.tools.crm_tools import (
    get_customer_history,
    get_deal_details,
    get_pipeline_summary,
    search_opportunities,
    update_opportunity_stage,
)
from app.deepagents.tools.document_tools import (
    clear_attachments,
    get_attachment_summary,
    get_document_context,
    list_attachments,
    search_attachments,
    set_current_session,
)
from app.deepagents.tools.pricing_tools import (
    analyze_margin,
    calculate_pricing,
    generate_pricing_options,
    get_pricing_model_recommendation,
)
from app.deepagents.tools.proposal_tools import (
    draft_proposal_section,
    extract_requirements,
    generate_executive_summary,
    get_template_details,
    search_past_proposals,
    search_rfp_templates,
)

SALES_INTELLIGENCE_SYSTEM_PROMPT = """You are a Sales & Pre-Sales Intelligence Deep Agent — an advanced AI coordinator for sales operations.

## Your Role
You assist with deal qualification, solution shaping, proposal drafting, and win-probability optimization by:
1. Breaking down tasks into actionable steps (planning)
2. Delegating specialized work to expert subagents
3. Managing context through file system tools
4. Tracking progress and ensuring completion

## Available Subagents
Use the `task` tool to delegate to specialized subagents:

- **deal-qualifier**: Lead qualification using BANT/MEDDIC, opportunity assessment
- **solution-architect**: Requirement mapping, solution design by business line
- **proposal-writer**: RFP/RFI response drafting, executive summaries
- **pricing-analyst**: Pricing strategy, margin analysis, commercial modeling
- **competitive-strategist**: Competitive positioning, objection handling, win strategies

## Core Capabilities

### Deal Qualification
- Assess opportunities using BANT/MEDDIC frameworks
- Calculate win probability based on key factors
- Identify risks and recommend mitigations

### RFP/RFI Response
- Extract requirements from RFP documents
- Draft proposal sections using proven templates
- Generate compelling executive summaries

### Solution Mapping
- Map customer requirements to solutions
- Identify the right business line and approach
- Design solution architectures

### Competitive Positioning
- Analyze competitor strengths and weaknesses
- Develop differentiation strategies
- Prepare objection responses

### Pricing Optimization
- Calculate pricing with margin analysis
- Generate pricing options (economy/standard/premium)
- Recommend pricing models

## Planning Guidelines
For complex requests, ALWAYS start with `write_todos` to create a task plan:
1. Identify all components of the request
2. Determine which subagents/tools are needed
3. Create ordered todo items
4. Execute and track progress

## Context Management
Use file system tools to:
- `write_file`: Save draft proposals, analysis, notes
- `read_file`: Retrieve saved context
- `ls`: List available context files

## Document Attachments
Users can upload documents (PDF, Word, TXT, PPT, images) for context.
This is especially useful for RFP documents, customer requirements, competitive intel.
When documents are uploaded, use these tools:
- `search_attachments(query)`: Search document content using semantic similarity
- `list_attachments()`: See all uploaded documents
- `get_attachment_summary()`: Get overview of a document
- `clear_attachments()`: Clear documents from session

**NOTE**: You do NOT need to provide session_id parameter - it is handled automatically.

**IMPORTANT**: When users ask about uploaded documents (RFPs, requirements, etc.),
ALWAYS use `search_attachments` to find relevant content first.
Never answer from memory - search the documents for accurate information.
The document context is automatically available to you through these tools.

## Response Format
- Provide clear, actionable responses
- Reference opportunity IDs and customer names
- Include win probability and risk assessments
- Summarize key insights and recommendations
- Recommend specific next steps

## Example Workflows

### RFP Response
1. Extract requirements from RFP
2. Search for relevant templates
3. Draft key proposal sections
4. Generate executive summary
5. Calculate pricing with options
6. Competitive positioning analysis

### Deal Qualification
1. Get deal and customer details
2. Apply BANT/MEDDIC framework
3. Calculate win probability
4. Assess deal risks
5. Recommend qualification actions

### Competitive Deal Strategy
1. Identify competitors
2. Analyze competitive landscape
3. Develop positioning strategy
4. Prepare objection handlers
5. Document win themes
"""


class SalesIntelligenceDeepAgent:
    """Sales & Pre-Sales Intelligence Deep Agent.

    Coordinates specialized subagents to handle complex sales workflows
    including deal qualification, RFP responses, solution mapping,
    competitive strategy, and pricing optimization.
    """

    def __init__(
        self,
        model_provider: Literal["openai", "anthropic", "auto"] = "auto",
        model_name: str | None = None,
        temperature: float = 0,
        storage_path: str = "./data/deepagent_context",
    ) -> None:
        """Initialize Sales Intelligence Deep Agent.

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
    ):
        """Create LLM instance.

        Uses the centralized LLM factory which supports:
        - Azure OpenAI (primary for production)
        - OpenAI (disabled by default)
        - Anthropic (fallback)
        """
        provider_arg = provider if provider != "auto" else None
        return get_llm(
            provider=provider_arg,
            model=model_name,
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

        # Sales tools (direct access)
        sales_tools = [
            # CRM tools
            search_opportunities,
            get_deal_details,
            update_opportunity_stage,
            get_customer_history,
            get_pipeline_summary,
            # Proposal tools
            search_rfp_templates,
            get_template_details,
            extract_requirements,
            draft_proposal_section,
            generate_executive_summary,
            search_past_proposals,
            # Competitor tools
            get_competitive_analysis,
            compare_solutions,
            suggest_differentiators,
            get_objection_handler,
            # Pricing tools
            calculate_pricing,
            analyze_margin,
            generate_pricing_options,
            get_pricing_model_recommendation,
            # Analytics tools
            calculate_win_probability,
            assess_deal_risk,
            get_similar_deals,
            get_sales_performance_summary,
        ]

        # Document attachment tools
        doc_tools = [
            search_attachments,
            list_attachments,
            get_attachment_summary,
            clear_attachments,
        ]

        return planning_tools + fs_tools + task_tools + sales_tools + doc_tools

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
                todo_item = Todo(
                    id=str(uuid.uuid4())[:8],
                    content=item.get("content", ""),
                    priority=int(item.get("priority", 0)),
                    status=TodoStatus.PENDING,
                )
                created_todos.append(todo_item)

            # Save to storage
            storage.save_todos(session_id, created_todos)

            lines = ["**Task Plan Created:**\n"]
            for i, todo_item in enumerate(created_todos, 1):
                priority_marker = "!" * todo_item.priority if todo_item.priority > 0 else ""
                lines.append(f"{i}. [ ] {priority_marker}{todo_item.content}")

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

            Use this to save draft proposals, analysis, or findings.

            Args:
                path: File path (e.g., /proposals/draft_executive_summary.md).
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
        subagents = {s.name: s for s in get_all_sales_subagents()}

        # Map subagent names to their tools
        subagent_tools_map = {
            "deal-qualifier": [
                search_opportunities,
                get_deal_details,
                get_customer_history,
                calculate_win_probability,
                assess_deal_risk,
            ],
            "solution-architect": [
                get_deal_details,
                extract_requirements,
                search_rfp_templates,
                get_template_details,
                get_customer_history,
                get_competitive_analysis,
            ],
            "proposal-writer": [
                search_rfp_templates,
                get_template_details,
                extract_requirements,
                draft_proposal_section,
                generate_executive_summary,
                search_past_proposals,
                suggest_differentiators,
            ],
            "pricing-analyst": [
                calculate_pricing,
                analyze_margin,
                generate_pricing_options,
                get_pricing_model_recommendation,
                get_deal_details,
                get_competitive_analysis,
            ],
            "competitive-strategist": [
                get_competitive_analysis,
                compare_solutions,
                suggest_differentiators,
                get_objection_handler,
                get_similar_deals,
            ],
        }

        @tool
        def task(
            subagent_type: str,
            task_description: str,
            context: str | None = None,
        ) -> str:
            """Delegate a task to a specialized subagent.

            Available subagents:
            - deal-qualifier: Lead qualification using BANT/MEDDIC
            - solution-architect: Requirement mapping, solution design
            - proposal-writer: RFP/RFI response drafting
            - pricing-analyst: Pricing strategy and margin analysis
            - competitive-strategist: Competitive positioning and objections

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
            tools = subagent_tools_map.get(subagent_type, [])

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
                        for tc in response.tool_calls:
                            if tc["id"] in tool_results:
                                messages.append(
                                    ToolMessage(
                                        content=str(tool_results[tc["id"]]),
                                        tool_call_id=tc["id"],
                                    )
                                )

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
        system_prompt = SALES_INTELLIGENCE_SYSTEM_PROMPT

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
        if state.iteration_count > 40:  # Safety limit
            return "end"

        last_message = state.messages[-1] if state.messages else None

        if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"

        return "end"

    @traceable(name="sales_intelligence_chat", tags=["deep_agent", "sales"])
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

        # Set session context for document tools
        set_current_session(session_id)

        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": 50,
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

        # Set session context for document tools
        set_current_session(session_id)

        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": 50,
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

        # Set session context for document tools
        set_current_session(session_id)

        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": 50,
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
        pending_tool_calls: list[dict] = []  # Track pending tool calls
        last_tool_name: str | None = None  # Track the last tool being executed

        # Phase detection based on iteration and tool usage
        def get_phase(iter_num: int, has_tool_calls: bool, tool_name: str | None = None) -> str:
            if tool_name:
                if tool_name in ("write_todos", "update_todo"):
                    return "planning"
                elif tool_name in (
                    "search_opportunities",
                    "get_deal_details",
                    "get_customer_history",
                    "search_rfp_templates",
                    "extract_requirements",
                    "search_attachments",
                    "list_attachments",
                    "get_attachment_summary",
                    "get_competitive_analysis",
                    "get_similar_deals",
                ):
                    return "analyzing"
                else:
                    return "executing"
            if has_tool_calls:
                return "executing"
            if iter_num == 1:
                return "planning"
            return "summarizing"

        # Get a descriptive message for what's happening
        def get_phase_message(phase: str, tool_name: str | None = None, tool_input: dict | None = None) -> str:
            if tool_name:
                return self._get_tool_description(tool_name, tool_input or {})
            phase_messages = {
                "planning": "Planning approach...",
                "analyzing": "Analyzing data...",
                "executing": "Executing actions...",
                "summarizing": "Preparing response...",
            }
            return phase_messages.get(phase, "Processing...")

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
                    pending_tool_calls = []  # Reset for new iteration
                    thinking_buffer = ""  # Reset thinking buffer for new iteration

                    # Determine phase based on context
                    if last_tool_name:
                        # Coming back after tool execution - analyzing results
                        current_phase = "analyzing"
                        phase_msg = f"Analyzing {last_tool_name} results..."
                    elif iteration == 1:
                        current_phase = "planning"
                        phase_msg = "Planning approach..."
                    else:
                        current_phase = "summarizing"
                        phase_msg = "Preparing response..."

                    yield {
                        "type": "thinking",
                        "data": {
                            "iteration": iteration,
                            "phase": current_phase,
                            "content": "",
                            "summary": phase_msg,
                            "is_reasoning_token": False,
                        },
                    }

                # LLM finished - check for tool calls
                elif event_type == "on_chat_model_end":
                    output = event_data.get("output")
                    if output and hasattr(output, "tool_calls") and output.tool_calls:
                        pending_tool_calls = output.tool_calls
                        # Emit thinking update about pending tool calls
                        tool_names = [
                            tc.get("name", tc.get("function", {}).get("name", "unknown")) for tc in pending_tool_calls
                        ]
                        current_phase = "executing"

                        yield {
                            "type": "thinking",
                            "data": {
                                "iteration": iteration,
                                "phase": current_phase,
                                "content": f"Calling tools: {', '.join(tool_names)}",
                                "summary": f"Executing {len(tool_names)} tool(s)...",
                                "is_reasoning_token": False,
                            },
                        }

                # Streaming tokens from LLM
                elif event_type == "on_chat_model_stream":
                    chunk = event_data.get("chunk")
                    if chunk:
                        # Check for extended thinking content (Claude with thinking enabled)
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
                    last_tool_name = tool_name  # Track for next iteration context

                    # Update phase based on tool type
                    current_phase = get_phase(iteration, True, tool_name)

                    # Format tool description
                    tool_desc = self._get_tool_description(tool_name, tool_input)

                    # Safely serialize tool_input
                    safe_input = {}
                    if isinstance(tool_input, dict):
                        for k, v in tool_input.items():
                            try:
                                if hasattr(v, "model_dump"):
                                    safe_input[k] = v.model_dump()
                                elif hasattr(v, "isoformat"):
                                    safe_input[k] = v.isoformat()
                                else:
                                    safe_input[k] = (
                                        str(v)
                                        if not isinstance(v, (str, int, float, bool, list, dict, type(None)))
                                        else v
                                    )
                            except Exception:
                                safe_input[k] = str(v)
                    else:
                        safe_input = str(tool_input)

                    # Emit thinking update with tool-specific message
                    yield {
                        "type": "thinking",
                        "data": {
                            "iteration": iteration,
                            "phase": current_phase,
                            "content": f"Executing: {tool_name}",
                            "summary": tool_desc,
                            "is_reasoning_token": False,
                        },
                    }

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

                    # Emit thinking update about tool completion
                    yield {
                        "type": "thinking",
                        "data": {
                            "iteration": iteration,
                            "phase": current_phase,
                            "content": f"Completed: {tool_name}",
                            "summary": f"{tool_name} completed",
                            "is_reasoning_token": False,
                        },
                    }

                    yield {
                        "type": "tool_result",
                        "data": {
                            "tool": tool_name,
                            "result": str(output)[:500],
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
            except Exception:
                files = []

            try:
                todos = self.storage.get_todos(session_id)
                todo_list = [t.model_dump(mode="json") for t in todos]
            except Exception:
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
            "update_todo": "Updating task status...",
            "write_file": f"Saving to {tool_input.get('path', 'file')}...",
            "read_file": f"Reading {tool_input.get('path', 'file')}...",
            "ls": "Listing workspace files...",
            "task": f"Delegating to {tool_input.get('subagent_type', 'subagent')}...",
            "search_opportunities": "Searching CRM opportunities...",
            "get_deal_details": f"Getting deal {tool_input.get('opportunity_id', '')}...",
            "get_customer_history": "Retrieving customer history...",
            "search_rfp_templates": "Searching proposal templates...",
            "extract_requirements": "Extracting requirements from RFP...",
            "draft_proposal_section": "Drafting proposal section...",
            "generate_executive_summary": "Generating executive summary...",
            "get_competitive_analysis": f"Analyzing competitor {tool_input.get('competitor_name', '')}...",
            "calculate_pricing": "Calculating pricing...",
            "analyze_margin": "Analyzing margins...",
            "calculate_win_probability": "Calculating win probability...",
            "assess_deal_risk": "Assessing deal risks...",
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
            content = content[: max_length - 3] + "..."

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


def create_sales_intelligence_agent(
    model_provider: Literal["openai", "anthropic", "auto"] = "auto",
    model_name: str | None = None,
    **kwargs,
) -> SalesIntelligenceDeepAgent:
    """Factory function to create Sales Intelligence Deep Agent.

    Args:
        model_provider: LLM provider.
        model_name: Specific model name.
        **kwargs: Additional configuration.

    Returns:
        Configured SalesIntelligenceDeepAgent instance.
    """
    return SalesIntelligenceDeepAgent(
        model_provider=model_provider,
        model_name=model_name,
        **kwargs,
    )


# LangGraph Studio entry point
def get_graph():
    """Entry point for LangGraph Studio."""
    agent = SalesIntelligenceDeepAgent(model_provider="auto")
    return agent.graph
