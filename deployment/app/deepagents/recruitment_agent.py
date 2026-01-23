"""Recruitment Deep Agent - Main Coordinator.

This is the main Deep Agent for AI-powered recruitment automation.
It coordinates specialized subagents to handle the complete recruitment workflow
from resume screening to candidate shortlisting.

Following Enterprise Development Standards:
- Software Architect: Modular agent architecture with subagent delegation
- Security Architect: PII handling, secure document processing
- Data Architect: Structured candidate data management
- Software Engineer: Type-safe with comprehensive error handling
"""

import os
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langsmith import traceable

from app.agents.base.llm_factory import get_llm
from app.deepagents.core.types import Todo, TodoStatus
from app.deepagents.core.state import DeepAgentState
from app.deepagents.storage.persistent_backend import PersistentStorage

# Import all recruitment tools
from app.deepagents.tools.sharepoint_tools import (
    list_sharepoint_folder,
    download_sharepoint_document,
    upload_to_sharepoint,
    search_sharepoint_documents,
    get_cached_document,
    create_sharepoint_folder,
)

from app.deepagents.tools.recruitment_tools import (
    parse_resume,
    parse_job_description,
    screen_candidate,
    batch_screen_resumes,
    get_candidate_profile,
    list_candidates,
    list_job_descriptions,
    get_shortlisted_candidates,
    get_session_dashboard,
    clear_session_data,
)

from app.deepagents.tools.interview_tools import (
    generate_interview_questions,
    export_question_set,
    submit_candidate_answers,
    evaluate_candidate_answers,
    get_candidate_score,
    list_question_sets,
)

from app.deepagents.tools.scoring_tools import (
    generate_scoring_report,
    export_scoring_excel,
    get_ranking_summary,
    get_passing_score_thresholds,
    generate_shortlist_report,
)

from app.deepagents.subagents.recruitment_subagents import (
    get_recruitment_subagents,
    get_recruitment_subagent_tools,
)

# Import document tools for attachment handling
from app.deepagents.tools.document_tools import (
    search_attachments,
    list_attachments,
    get_attachment_summary,
    clear_attachments,
    get_document_context,
    set_current_session,
)


RECRUITMENT_SYSTEM_PROMPT = """You are a Recruitment Deep Agent - an advanced AI coordinator for end-to-end recruitment automation.

## Your Role
You coordinate the complete recruitment process by:
1. Managing job descriptions and resumes from SharePoint
2. Screening candidates through L1/L2/L3 levels
3. Generating and administering technical assessments
4. Evaluating candidate responses
5. Producing scoring reports and shortlists

## First Response Priority
When a user starts a new session or says "hello"/"start", ALWAYS begin by calling
`get_session_dashboard` to show the current state. This gives immediate visibility
into progress and next steps.

## Available Subagents
Use the `task` tool to delegate to specialized subagents:

- **document-manager**: SharePoint document operations (list, download, upload)
- **resume-screener**: Resume parsing and candidate screening
- **question-generator**: Technical interview question creation
- **answer-evaluator**: Candidate answer evaluation and scoring
- **report-generator**: Scoring reports and Excel exports

## Workflow Phases

### 1. Setup Phase
- List available JDs from SharePoint (`list_sharepoint_folder` with folder_type="jd")
- Download and parse the target job description
- List and download candidate resumes from SharePoint

### 2. Screening Phase
- Parse resumes to extract candidate profiles
- Screen candidates against JD requirements
- Categorize by level (L1, L2, L3)
- Generate shortlist of qualified candidates

### 3. Assessment Phase
- Generate interview questions for shortlisted candidates
- Export question sets to SharePoint for candidates
- Collect candidate answer submissions
- Evaluate answers and assign scores

### 4. Reporting Phase
- Generate comprehensive scoring reports
- Export results to Excel format
- Create final shortlist report
- Upload reports to SharePoint

## Quick Actions (User Shortcuts)
When users say these phrases, respond with the corresponding actions:

- **"show status"** / **"dashboard"** -> Call `get_session_dashboard`
- **"list positions"** -> Call `list_job_descriptions`
- **"list candidates"** -> Call `list_candidates`
- **"screen all"** -> Call `batch_screen_resumes` with the active JD
- **"generate questions for [name]"** -> Look up candidate, call `generate_interview_questions`
- **"show rankings"** -> Call `get_ranking_summary`
- **"show config"** -> Call `get_passing_score_thresholds`
- **"export report"** -> Call `export_scoring_excel`
- **"full cycle"** -> Execute complete recruitment cycle (parse JD -> resumes -> screen -> questions -> report)
- **"cleanup"** / **"clear data"** -> Call `clear_session_data` for PII compliance

## Planning Guidelines
For complex requests, ALWAYS start with `write_todos` to create a task plan:
1. Identify the recruitment stage required
2. Determine which tools/subagents are needed
3. Create ordered todo items
4. Execute and track progress

## Document Attachments
Users can upload documents (PDF, Word, TXT) for processing.
When documents are uploaded, use these tools:
- `search_attachments(query)`: Search document content
- `list_attachments()`: See all uploaded documents
- `get_attachment_summary()`: Get document overview

**IMPORTANT**: When users upload resumes or JDs directly, use these attachment tools
in addition to SharePoint tools for locally uploaded files.

## Context Management
Use file system tools to:
- `write_file`: Save analysis notes, candidate summaries
- `read_file`: Retrieve saved context
- `ls`: List available context files

## Configuration
The recruitment process is governed by configurable parameters:
- Passing scores by level (L1: 60%, L2: 70%, L3: 80% by default)
- Score weights (Technical: 40%, Experience: 25%, Education: 15%, etc.)
- Question counts and difficulty distributions

Use `get_passing_score_thresholds` to view current configuration.

## Response Format
- Provide clear, actionable responses with progress indicators
- Reference candidate names and IDs in all outputs
- Summarize screening/evaluation findings with tables when possible
- Recommend specific next steps based on current workflow phase
- Be transparent about all decisions with scoring justification
- After batch operations, always show the session dashboard

## Error Recovery
When a tool fails or returns unexpected results:
1. Explain clearly what went wrong
2. Suggest alternative approaches
3. Never repeat the same failing action without changing parameters
4. If SharePoint is unavailable, offer demo mode alternatives
5. If a candidate/JD is not found, list available ones

## Data Privacy (CRITICAL)
- Never expose raw PII (email, phone) in summary reports shared externally
- Recommend `clear_session_data` after recruitment cycles complete
- Do not retain candidate data beyond the session unless explicitly requested
- Mask sensitive fields in exported reports

## Example Workflows

### Screen Candidates for a Position
1. `get_session_dashboard` - Check current state
2. `list_sharepoint_folder(folder_type="jd")` - List available JDs
3. `download_sharepoint_document` + `parse_job_description` - Parse JD
4. `list_sharepoint_folder(folder_type="resumes")` - List resumes
5. Parse each resume with `parse_resume`
6. `batch_screen_resumes(jd_id)` - Screen all at once
7. `get_session_dashboard` - Show updated progress

### Conduct Technical Assessment
1. `get_shortlisted_candidates(jd_id)` - Get shortlisted candidates
2. `generate_interview_questions` for each candidate
3. `export_question_set` - Create candidate-facing documents
4. `upload_to_sharepoint` - Save to SharePoint
5. `submit_candidate_answers` - Record responses when received
6. `evaluate_candidate_answers` - Score all submissions
7. `generate_scoring_report` - Create comprehensive report

### Complete Recruitment Cycle (Full Automation)
1. Parse JD requirements
2. Screen all resumes (L1/L2/L3 classification)
3. Generate interview questions for shortlisted
4. Evaluate submitted answers
5. Export scoring Excel to SharePoint
6. Generate final shortlist report
7. Show final dashboard with recommendations
8. Offer `clear_session_data` for PII cleanup
"""


class RecruitmentDeepAgent:
    """Recruitment Deep Agent for end-to-end hiring automation.

    Coordinates specialized subagents to handle the complete recruitment workflow
    including resume screening, technical assessments, and candidate shortlisting.
    """

    def __init__(
        self,
        model_provider: Literal["openai", "anthropic", "auto"] = "auto",
        model_name: str | None = None,
        temperature: float = 0,
        storage_path: str = "./data/recruitment_context",
    ) -> None:
        """Initialize Recruitment Deep Agent.

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

        # SharePoint tools
        sharepoint_tools = [
            list_sharepoint_folder,
            download_sharepoint_document,
            upload_to_sharepoint,
            search_sharepoint_documents,
            get_cached_document,
            create_sharepoint_folder,
        ]

        # Recruitment screening tools
        recruitment_tools = [
            parse_resume,
            parse_job_description,
            screen_candidate,
            batch_screen_resumes,
            get_candidate_profile,
            list_candidates,
            list_job_descriptions,
            get_shortlisted_candidates,
            get_session_dashboard,
            clear_session_data,
        ]

        # Interview tools
        interview_tools = [
            generate_interview_questions,
            export_question_set,
            submit_candidate_answers,
            evaluate_candidate_answers,
            get_candidate_score,
            list_question_sets,
        ]

        # Scoring/reporting tools
        scoring_tools = [
            generate_scoring_report,
            export_scoring_excel,
            get_ranking_summary,
            get_passing_score_thresholds,
            generate_shortlist_report,
        ]

        # Document attachment tools
        doc_tools = [
            search_attachments,
            list_attachments,
            get_attachment_summary,
            clear_attachments,
        ]

        return (
            planning_tools +
            fs_tools +
            task_tools +
            sharepoint_tools +
            recruitment_tools +
            interview_tools +
            scoring_tools +
            doc_tools
        )

    def _create_write_todos_tool(self):
        """Create the write_todos planning tool."""
        storage = self.storage

        @tool
        def write_todos(todos: list[dict[str, Any]], session_id: str = "default") -> str:
            """Create or update the task plan for current work.

            Break down complex recruitment tasks into discrete steps.

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

            storage.save_todos(session_id, created_todos)

            lines = ["**Recruitment Task Plan Created:**\n"]
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

            Use this to save candidate summaries, analysis notes, or reports.

            Args:
                path: File path (e.g., /notes/candidate_summary.md).
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
        subagents = {s.name: s for s in get_recruitment_subagents()}

        @tool
        def task(
            subagent_type: str,
            task_description: str,
            context: str | None = None,
        ) -> str:
            """Delegate a task to a specialized recruitment subagent.

            Available subagents:
            - document-manager: SharePoint document operations
            - resume-screener: Resume parsing and screening
            - question-generator: Interview question creation
            - answer-evaluator: Candidate answer evaluation
            - report-generator: Scoring reports and exports

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
            tools = get_recruitment_subagent_tools(subagent_type)

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
        system_prompt = RECRUITMENT_SYSTEM_PROMPT

        # Inject document context if documents are uploaded
        session_id = state.session_id or "default"
        doc_context = get_document_context(session_id)
        if doc_context:
            system_prompt += "\n" + doc_context

        # Ensure system message is first
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages
        else:
            messages[0] = SystemMessage(content=system_prompt)

        response = self.llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "iteration_count": state.iteration_count + 1,
            "last_activity": datetime.now(),
        }

    def _should_continue(self, state: DeepAgentState) -> Literal["continue", "end"]:
        """Determine if we should continue to tools or end."""
        if state.iteration_count > 40:
            return "end"

        last_message = state.messages[-1] if state.messages else None

        if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"

        return "end"

    @traceable(name="recruitment_chat", tags=["deep_agent", "recruitment"])
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
        - thinking: Agent's reasoning process
        - tool_start: When a tool is being called
        - tool_result: Tool execution result
        - todo_update: Todo list changes
        - token: Streaming tokens
        - complete: Final response

        Args:
            message: User message.
            session_id: Session identifier.
            user_id: User identifier.

        Yields:
            Event dictionaries with type and data.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

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
                "message": "Processing recruitment request...",
            },
        }

        iteration = 0
        final_response = ""
        current_phase = "planning"
        last_tool_name: str | None = None

        def get_phase(iter_num: int, tool_name: str | None = None) -> str:
            if tool_name:
                if tool_name in ("write_todos", "update_todo"):
                    return "planning"
                elif tool_name in ("list_sharepoint_folder", "download_sharepoint_document",
                                   "search_sharepoint_documents", "parse_resume",
                                   "parse_job_description", "list_candidates",
                                   "list_job_descriptions"):
                    return "analyzing"
                elif tool_name in ("screen_candidate", "batch_screen_resumes",
                                   "generate_interview_questions", "evaluate_candidate_answers"):
                    return "executing"
                else:
                    return "executing"
            return "summarizing" if iter_num > 1 else "planning"

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

                if event_type == "on_chat_model_start":
                    iteration += 1
                    current_phase = get_phase(iteration, last_tool_name)

                    yield {
                        "type": "thinking",
                        "data": {
                            "iteration": iteration,
                            "phase": current_phase,
                            "content": "",
                            "summary": f"Processing step {iteration}...",
                            "is_reasoning_token": False,
                        },
                    }

                elif event_type == "on_chat_model_stream":
                    chunk = event_data.get("chunk")
                    if chunk and hasattr(chunk, "content"):
                        content = chunk.content
                        if isinstance(content, str) and content:
                            yield {
                                "type": "token",
                                "data": {"content": content},
                            }

                elif event_type == "on_tool_start":
                    tool_name = event_name
                    tool_input = event_data.get("input", {})
                    last_tool_name = tool_name
                    current_phase = get_phase(iteration, tool_name)

                    yield {
                        "type": "tool_start",
                        "data": {
                            "tool": tool_name,
                            "input": str(tool_input)[:200],
                            "description": self._get_tool_description(tool_name, tool_input),
                            "phase": current_phase,
                        },
                    }

                elif event_type == "on_tool_end":
                    tool_name = event_name
                    output = event_data.get("output", "")

                    yield {
                        "type": "tool_result",
                        "data": {
                            "tool": tool_name,
                            "result": str(output)[:500],
                        },
                    }

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
                        except Exception:
                            pass

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
            # SharePoint tools
            "list_sharepoint_folder": f"Listing SharePoint folder {tool_input.get('folder_type', '')}...",
            "download_sharepoint_document": f"Downloading {tool_input.get('filename', 'document')}...",
            "upload_to_sharepoint": f"Uploading {tool_input.get('filename', 'document')}...",
            "search_sharepoint_documents": f"Searching for '{tool_input.get('query', '')}'...",
            # Recruitment tools
            "parse_resume": f"Parsing resume: {tool_input.get('filename', 'resume')}...",
            "parse_job_description": f"Parsing JD: {tool_input.get('title', 'job')}...",
            "screen_candidate": "Screening candidate...",
            "batch_screen_resumes": "Batch screening all candidates...",
            "get_candidate_profile": "Getting candidate profile...",
            "list_candidates": "Listing all candidates...",
            "list_job_descriptions": "Listing job descriptions...",
            "get_shortlisted_candidates": "Getting shortlisted candidates...",
            # Interview tools
            "generate_interview_questions": f"Generating questions for {tool_input.get('candidate_name', 'candidate')}...",
            "export_question_set": "Exporting question set...",
            "submit_candidate_answers": "Submitting candidate answers...",
            "evaluate_candidate_answers": "Evaluating answers...",
            "get_candidate_score": "Getting candidate score...",
            "list_question_sets": "Listing question sets...",
            # Scoring tools
            "generate_scoring_report": "Generating scoring report...",
            "export_scoring_excel": "Exporting to Excel...",
            "get_ranking_summary": "Getting candidate rankings...",
            "generate_shortlist_report": "Generating shortlist report...",
            "get_passing_score_thresholds": "Getting score thresholds...",
            # Session management tools
            "get_session_dashboard": "Loading session dashboard...",
            "clear_session_data": "Clearing session data (PII cleanup)...",
            # Document tools
            "search_attachments": f"Searching documents for '{tool_input.get('query', '')[:30]}'...",
            "list_attachments": "Listing uploaded documents...",
            "get_attachment_summary": "Getting document summary...",
            "clear_attachments": "Clearing documents...",
        }
        return descriptions.get(tool_name, f"Executing {tool_name}...")

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


def create_recruitment_agent(
    model_provider: Literal["openai", "anthropic", "auto"] = "auto",
    model_name: str | None = None,
    **kwargs,
) -> RecruitmentDeepAgent:
    """Factory function to create Recruitment Deep Agent.

    Args:
        model_provider: LLM provider.
        model_name: Specific model name.
        **kwargs: Additional configuration.

    Returns:
        Configured RecruitmentDeepAgent instance.
    """
    return RecruitmentDeepAgent(
        model_provider=model_provider,
        model_name=model_name,
        **kwargs,
    )


# LangGraph Studio entry point
def get_graph():
    """Entry point for LangGraph Studio."""
    agent = RecruitmentDeepAgent(model_provider="auto")
    return agent.graph


__all__ = [
    "RecruitmentDeepAgent",
    "create_recruitment_agent",
    "get_graph",
]
