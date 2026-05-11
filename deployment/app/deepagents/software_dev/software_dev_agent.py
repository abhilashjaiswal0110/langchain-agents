"""Software Development DeepAgent - Main Coordinator.

This is the main Deep Agent for AI-powered software development lifecycle automation.
It coordinates specialized subagents to handle the complete SDLC from requirements
to deployment.

Following Enterprise Development Standards:
- Software Architect: Modular agent architecture with subagent delegation
- Security Architect: Secure coding practices, vulnerability scanning
- Data Architect: Structured code and artifact management
- Software Engineer: Type-safe with comprehensive error handling
"""

import logging
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langsmith import traceable

from app.agents.base.llm_factory import get_llm
from app.deepagents.config.software_dev_config import (
    SDLCPhase,
    SoftwareDevAgentConfig,
    get_default_config,
)
from app.deepagents.core.middleware import (
    FilesystemMiddleware,
    SubAgentMiddleware,
    TodoListMiddleware,
)
from app.deepagents.core.types import TodoStatus
from app.deepagents.software_dev.state import SoftwareDevState
from app.deepagents.software_dev.subagents import get_all_subagents

logger = logging.getLogger(__name__)


SOFTWARE_DEV_SYSTEM_PROMPT = """You are a Software Development DeepAgent - an advanced AI coordinator for end-to-end software development lifecycle (SDLC) automation.

## Your Role

You coordinate the complete software development process by:
1. Understanding and refining requirements
2. Designing system architecture
3. Generating production-ready code
4. Performing automated code reviews
5. Creating comprehensive tests
6. Scanning for security vulnerabilities
7. Setting up CI/CD pipelines
8. Debugging and optimizing performance
9. Generating technical documentation

## First Response Priority

When a user starts a new session or provides a task, ALWAYS:
1. Analyze the request to determine the SDLC phase needed
2. Create a task plan using `write_todos`
3. Delegate to appropriate subagents using the `task` tool

## Available Subagents

Use the `task` tool to delegate to specialized subagents:

- **requirements-intelligence**: Extract and validate software requirements
- **architecture-design**: Design system architecture and APIs
- **code-generator**: Generate production-ready code
- **code-reviewer**: Perform automated code reviews
- **testing-automation**: Create and run tests
- **debugging-optimization**: Debug issues and optimize performance
- **security-compliance**: Scan for vulnerabilities and compliance
- **devops-integration**: Create CI/CD pipelines and deployment configs
- **documentation**: Generate technical documentation

## SDLC Phases

### 1. Requirements Phase
- Analyze natural language requirements
- Extract user stories
- Generate acceptance criteria
- Detect ambiguities and risks

### 2. Design Phase
- Propose architecture patterns
- Create API specifications
- Design data models
- Suggest technology stack

### 3. Implementation Phase
- Generate code following best practices
- Apply design patterns
- Refactor existing code
- Format and organize imports

### 4. Review Phase
- Perform code reviews
- Check style compliance
- Analyze complexity
- Detect code smells

### 5. Testing Phase
- Generate unit tests
- Create integration tests
- Analyze test coverage
- Run test suites

### 6. Security Phase
- Scan for vulnerabilities
- Check OWASP compliance
- Detect secrets in code
- Analyze dependency security

### 7. DevOps Phase
- Create CI pipelines
- Create CD pipelines
- Generate Dockerfiles
- Create Kubernetes configs

### 8. Debugging Phase
- Analyze errors
- Trace execution
- Identify root causes
- Propose fixes

### 9. Documentation Phase
- Generate API docs
- Create README files
- Document architecture
- Write user guides

## Quick Actions

When users say these phrases, respond with corresponding actions:

- **"analyze requirements"** / **"understand this"** -> Use requirements-intelligence subagent
- **"design architecture"** / **"plan this"** -> Use architecture-design subagent
- **"generate code"** / **"implement this"** -> Use code-generator subagent
- **"review code"** / **"check this"** -> Use code-reviewer subagent
- **"create tests"** / **"test this"** -> Use testing-automation subagent
- **"debug this"** / **"fix this"** -> Use debugging-optimization subagent
- **"security scan"** / **"check security"** -> Use security-compliance subagent
- **"create pipeline"** / **"deploy this"** -> Use devops-integration subagent
- **"document this"** / **"create docs"** -> Use documentation subagent
- **"full cycle"** -> Execute complete SDLC workflow

## Planning Guidelines

For complex requests, ALWAYS start with `write_todos` to create a task plan:
1. Identify the SDLC phase(s) required
2. Determine which subagents are needed
3. Create ordered todo items
4. Execute and track progress

## Context Management

Use file system tools to:
- `write_file`: Save code, configs, documentation
- `read_file`: Retrieve saved context
- `ls`: List available context files

## Response Format

- Provide clear, actionable responses with progress indicators
- Show code snippets with syntax highlighting
- Summarize findings with bullet points
- Recommend specific next steps
- Be transparent about decisions and trade-offs

## Error Recovery

When a tool fails or returns unexpected results:
1. Explain clearly what went wrong
2. Suggest alternative approaches
3. Never repeat the same failing action without changing parameters
4. If unsure, ask for clarification

## Quality Standards

All generated code must:
- Include type hints (Python) or types (TypeScript)
- Have comprehensive error handling
- Follow language-specific style guides
- Include documentation
- Be security-conscious
"""


class SoftwareDevDeepAgent:
    """Software Development DeepAgent for SDLC automation.

    This agent coordinates specialized subagents to provide
    end-to-end software development assistance.
    """

    def __init__(
        self,
        config: SoftwareDevAgentConfig | None = None,
        model_provider: Literal["openai", "anthropic", "auto"] = "auto",
    ) -> None:
        """Initialize Software Development DeepAgent.

        Args:
            config: Agent configuration.
            model_provider: LLM provider to use.
        """
        self.config = config or get_default_config()
        self.model_provider = model_provider

        # Initialize LLM
        self.llm = self._create_llm()

        # Initialize middleware
        self.todo_middleware = TodoListMiddleware(max_todos=20)
        self.filesystem_middleware = FilesystemMiddleware(
            workspace_path="./workspace/software_dev",
            max_file_size=self.config.max_file_size,
            persistent=True,
        )
        self.subagent_middleware = SubAgentMiddleware(
            subagents=get_all_subagents(),
            default_model=self.config.model,
            max_concurrent=self.config.max_concurrent_subagents,
        )

        # Collect all tools
        self.tools = self._collect_tools()

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Initialize checkpointer for memory
        self.checkpointer = MemorySaver()

        # Build the graph
        self.graph = self._build_graph()

        logger.info(f"SoftwareDevDeepAgent initialized with {len(self.tools)} tools")

    def _create_llm(self):
        """Create LLM instance based on configuration."""
        provider = self.model_provider if self.model_provider != "auto" else None

        return get_llm(
            provider=provider,
            model=self.config.model,
            temperature=self.config.temperature,
        )

    def _collect_tools(self) -> list:
        """Collect all tools from middleware and subagents."""
        tools = []

        # Add middleware tools
        tools.extend(self.todo_middleware.get_tools())
        tools.extend(self.filesystem_middleware.get_tools())
        tools.extend(self.subagent_middleware.get_tools())

        # Add a subset of direct tools for common operations
        from app.deepagents.software_dev.tools.codegen_tools import generate_code
        from app.deepagents.software_dev.tools.requirements_tools import analyze_requirements
        from app.deepagents.software_dev.tools.review_tools import review_code
        from app.deepagents.software_dev.tools.security_tools import scan_security_issues
        from app.deepagents.software_dev.tools.testing_tools import generate_unit_tests

        tools.extend(
            [
                analyze_requirements,
                generate_code,
                review_code,
                generate_unit_tests,
                scan_security_issues,
            ]
        )

        return tools

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        graph = StateGraph(SoftwareDevState)

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

    def _agent_node(self, state: SoftwareDevState) -> dict:
        """Process messages and decide on actions."""
        messages = list(state.messages)

        # Ensure system message is first
        if not messages or not isinstance(messages[0], SystemMessage):
            system_msg = SystemMessage(content=SOFTWARE_DEV_SYSTEM_PROMPT)
            messages = [system_msg] + messages

        # Add context about current state
        context_parts = []
        if state.todos:
            pending = len([t for t in state.todos if t.status == TodoStatus.PENDING])
            completed = len([t for t in state.todos if t.status == TodoStatus.COMPLETED])
            context_parts.append(f"[Tasks: {completed}/{len(state.todos)} completed, {pending} pending]")

        if state.current_phase:
            context_parts.append(f"[Phase: {state.current_phase.value}]")

        if state.project_name:
            context_parts.append(f"[Project: {state.project_name}]")

        if state.requirements:
            context_parts.append(f"[Requirements: {len(state.requirements)}]")

        if state.code_files:
            context_parts.append(f"[Code files: {len(state.code_files)}]")

        if state.test_cases:
            coverage = state.test_coverage
            context_parts.append(f"[Tests: {len(state.test_cases)}, Coverage: {coverage:.1f}%]")

        if state.security_issues:
            critical = len([i for i in state.security_issues if i.severity.value == "critical"])
            context_parts.append(f"[Security issues: {len(state.security_issues)} ({critical} critical)]")

        # Log thinking step
        if context_parts:
            state.add_thinking_step(f"Context: {' | '.join(context_parts)}")

        # Call LLM
        response = self.llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "iteration_count": state.iteration_count + 1,
            "last_activity": datetime.now(),
        }

    def _process_tool_results(self, state: SoftwareDevState) -> dict:
        """Process tool results and update state."""
        updates: dict[str, Any] = {
            "total_tool_calls": state.total_tool_calls + 1,
        }

        # Log thinking step
        state.add_thinking_step(f"Processed tool call #{state.total_tool_calls + 1}")

        return updates

    def _should_continue(self, state: SoftwareDevState) -> Literal["continue", "end"]:
        """Determine if we should continue to tools or end."""
        # Check recursion limit
        if state.iteration_count > self.config.max_iterations:
            logger.warning(f"Reached max iterations ({self.config.max_iterations})")
            return "end"

        last_message = state.messages[-1] if state.messages else None

        if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"

        return "end"

    @traceable(name="software_dev_chat", tags=["software_dev", "deep_agent"])
    def chat(
        self,
        message: str,
        session_id: str | None = None,
        user_id: str | None = None,
        project_name: str | None = None,
    ) -> dict[str, Any]:
        """Process a chat message.

        Args:
            message: User message.
            session_id: Session identifier for memory.
            user_id: User identifier.
            project_name: Project name for context.

        Returns:
            Response dictionary with result and metadata.
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
                "project_name": project_name,
            },
            config=config,
        )

        # Extract response
        last_message = result["messages"][-1] if result.get("messages") else None
        response_text = last_message.content if last_message else ""

        return {
            "response": response_text,
            "session_id": session_id,
            "phase": result.get("current_phase", SDLCPhase.REQUIREMENTS).value,
            "todos": [t.model_dump() for t in result.get("todos", [])],
            "files": list(result.get("files", {}).keys()),
            "requirements_count": len(result.get("requirements", [])),
            "code_files_count": len(result.get("code_files", {})),
            "test_coverage": result.get("test_coverage", 0.0),
            "security_issues_count": len(result.get("security_issues", [])),
            "tool_calls": getattr(last_message, "tool_calls", []) if last_message else [],
            "iteration_count": result.get("iteration_count", 0),
            "thinking_steps": result.get("thinking_steps", []),
        }

    async def achat(
        self,
        message: str,
        session_id: str | None = None,
        user_id: str | None = None,
        project_name: str | None = None,
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
                "project_name": project_name,
            },
            config=config,
        )

        last_message = result["messages"][-1] if result.get("messages") else None
        response_text = last_message.content if last_message else ""

        return {
            "response": response_text,
            "session_id": session_id,
            "phase": result.get("current_phase", SDLCPhase.REQUIREMENTS).value,
            "todos": [t.model_dump() for t in result.get("todos", [])],
            "iteration_count": result.get("iteration_count", 0),
        }

    async def astream(
        self,
        message: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream responses with thinking steps.

        Yields events as the agent processes the request.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        config = {"configurable": {"thread_id": session_id}}

        async for event in self.graph.astream_events(
            {
                "messages": [HumanMessage(content=message)],
                "session_id": session_id,
                "user_id": user_id,
            },
            config=config,
            version="v2",
        ):
            event_type = event.get("event", "")

            if event_type == "on_chat_model_start":
                yield {
                    "type": "thinking_start",
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                }

            elif event_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk", {})
                if hasattr(chunk, "content") and chunk.content:
                    yield {
                        "type": "token",
                        "content": chunk.content,
                        "session_id": session_id,
                    }

            elif event_type == "on_tool_start":
                tool_name = event.get("name", "unknown")
                yield {
                    "type": "tool_start",
                    "tool": tool_name,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                }

            elif event_type == "on_tool_end":
                tool_name = event.get("name", "unknown")
                yield {
                    "type": "tool_end",
                    "tool": tool_name,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                }

            elif event_type == "on_chain_end":
                if event.get("name") == "LangGraph":
                    output = event.get("data", {}).get("output", {})
                    yield {
                        "type": "complete",
                        "session_id": session_id,
                        "response": output.get("messages", [{}])[-1].content if output.get("messages") else "",
                        "timestamp": datetime.now().isoformat(),
                    }

    def get_session_state(self, session_id: str) -> dict[str, Any]:
        """Get current state for a session."""
        config = {"configurable": {"thread_id": session_id}}
        state = self.graph.get_state(config)

        if state and state.values:
            return {
                "session_id": session_id,
                "phase": state.values.get("current_phase", SDLCPhase.REQUIREMENTS).value,
                "todos": [t.model_dump() for t in state.values.get("todos", [])],
                "requirements": len(state.values.get("requirements", [])),
                "code_files": len(state.values.get("code_files", {})),
                "test_cases": len(state.values.get("test_cases", [])),
                "security_issues": len(state.values.get("security_issues", [])),
            }
        return {"session_id": session_id, "state": "not_found"}

    def transition_phase(self, session_id: str, new_phase: SDLCPhase) -> dict[str, Any]:
        """Transition to a new SDLC phase."""
        config = {"configurable": {"thread_id": session_id}}
        state = self.graph.get_state(config)

        if state and state.values:
            current = state.values.get("current_phase", SDLCPhase.REQUIREMENTS)
            return {
                "previous_phase": current.value,
                "new_phase": new_phase.value,
                "transitioned": True,
            }

        return {"error": "Session not found"}


def create_software_dev_agent(
    model: str = "gpt-4o-mini",
    model_provider: Literal["openai", "anthropic", "auto"] = "auto",
    **kwargs,
) -> SoftwareDevDeepAgent:
    """Factory function to create a Software Development DeepAgent.

    Args:
        model: LLM model to use.
        model_provider: LLM provider.
        **kwargs: Additional config options.

    Returns:
        Configured SoftwareDevDeepAgent instance.
    """
    config = SoftwareDevAgentConfig(
        model=model,
        **kwargs,
    )

    return SoftwareDevDeepAgent(
        config=config,
        model_provider=model_provider,
    )
