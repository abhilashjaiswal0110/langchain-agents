"""Software Development DeepAgent State Management.

This module extends the base DeepAgentState with SDLC-specific context tracking
for managing software development workflows.
"""

from datetime import datetime
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.deepagents.config.software_dev_config import (
    ArchitecturePattern,
    CodeLanguage,
    RequirementPriority,
    RequirementType,
    SDLCPhase,
    SecuritySeverity,
    TestType,
)
from app.deepagents.core.types import FileEntry, SubAgentResult, Todo


class Requirement(BaseModel):
    """A software requirement."""

    id: str
    title: str
    description: str
    type: RequirementType = RequirementType.FUNCTIONAL
    priority: RequirementPriority = RequirementPriority.SHOULD_HAVE
    acceptance_criteria: list[str] = Field(default_factory=list)
    status: str = "draft"  # draft, approved, implemented, verified
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserStory(BaseModel):
    """A user story derived from requirements."""

    id: str
    title: str
    as_a: str  # As a [user type]
    i_want: str  # I want [goal]
    so_that: str  # So that [benefit]
    acceptance_criteria: list[str] = Field(default_factory=list)
    story_points: int | None = None
    requirement_ids: list[str] = Field(default_factory=list)


class ArchitectureComponent(BaseModel):
    """An architecture component."""

    id: str
    name: str
    type: str  # service, database, queue, gateway, etc.
    description: str
    technologies: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    interfaces: list[dict[str, Any]] = Field(default_factory=list)


class APIEndpoint(BaseModel):
    """An API endpoint specification."""

    path: str
    method: str  # GET, POST, PUT, DELETE, PATCH
    description: str
    request_schema: dict[str, Any] | None = None
    response_schema: dict[str, Any] | None = None
    auth_required: bool = True
    parameters: list[dict[str, Any]] = Field(default_factory=list)


class CodeFile(BaseModel):
    """A code file in the project."""

    path: str
    language: CodeLanguage
    content: str
    line_count: int = 0
    test_coverage: float = 0.0
    last_modified: datetime = Field(default_factory=datetime.now)
    dependencies: list[str] = Field(default_factory=list)


class CodeReviewIssue(BaseModel):
    """An issue found during code review."""

    id: str
    file_path: str
    line_number: int | None = None
    severity: str  # critical, high, medium, low, info
    category: str  # style, bug, security, performance, maintainability
    message: str
    suggestion: str | None = None
    auto_fixable: bool = False


class TestCase(BaseModel):
    """A test case specification."""

    id: str
    name: str
    type: TestType
    description: str
    target_file: str
    target_function: str | None = None
    test_code: str | None = None
    status: str = "pending"  # pending, passed, failed, skipped
    execution_time: float | None = None


class SecurityIssue(BaseModel):
    """A security vulnerability or issue."""

    id: str
    severity: SecuritySeverity
    category: str  # OWASP category or custom
    title: str
    description: str
    file_path: str | None = None
    line_number: int | None = None
    cwe_id: str | None = None
    recommendation: str
    fixed: bool = False


class BuildPipeline(BaseModel):
    """A CI/CD pipeline definition."""

    name: str
    platform: str  # github-actions, gitlab-ci, jenkins, etc.
    stages: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    status: str = "draft"  # draft, active, running, passed, failed


class DebugSession(BaseModel):
    """A debugging session context."""

    id: str
    issue_description: str
    stack_trace: str | None = None
    affected_files: list[str] = Field(default_factory=list)
    root_cause: str | None = None
    proposed_fix: str | None = None
    status: str = "investigating"  # investigating, identified, fixed, verified


class DocumentationEntry(BaseModel):
    """A documentation entry."""

    id: str
    type: str  # api, readme, architecture, inline, changelog
    title: str
    content: str
    target_path: str | None = None
    status: str = "draft"  # draft, review, published


class SoftwareDevState(BaseModel):
    """State for Software Development DeepAgent execution.

    Extends the base DeepAgentState with SDLC-specific context:
    - Requirements and user stories
    - Architecture components and API specs
    - Code files and review issues
    - Test cases and security issues
    - CI/CD pipelines and debugging sessions
    - Documentation entries
    """

    # Conversation
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # Planning (TodoList middleware)
    todos: list[Todo] = Field(default_factory=list)
    current_todo_id: str | None = None

    # File System (Filesystem middleware)
    files: dict[str, FileEntry] = Field(default_factory=dict)
    working_directory: str = "/"

    # Subagent Results (SubAgent middleware)
    subagent_results: list[SubAgentResult] = Field(default_factory=list)
    active_subagents: list[str] = Field(default_factory=list)

    # Session metadata
    session_id: str | None = None
    user_id: str | None = None
    project_name: str | None = None
    started_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)

    # SDLC Phase tracking
    current_phase: SDLCPhase = SDLCPhase.REQUIREMENTS
    phase_history: list[tuple[SDLCPhase, datetime]] = Field(default_factory=list)

    # Requirements context
    requirements: list[Requirement] = Field(default_factory=list)
    user_stories: list[UserStory] = Field(default_factory=list)

    # Architecture context
    architecture_pattern: ArchitecturePattern | None = None
    components: list[ArchitectureComponent] = Field(default_factory=list)
    api_endpoints: list[APIEndpoint] = Field(default_factory=list)
    tech_stack: dict[str, str] = Field(default_factory=dict)

    # Code context
    code_files: dict[str, CodeFile] = Field(default_factory=dict)
    primary_language: CodeLanguage = CodeLanguage.PYTHON
    dependencies: list[str] = Field(default_factory=list)

    # Review context
    review_issues: list[CodeReviewIssue] = Field(default_factory=list)
    review_status: str = "pending"  # pending, in_progress, approved, rejected

    # Testing context
    test_cases: list[TestCase] = Field(default_factory=list)
    test_coverage: float = 0.0
    test_results: dict[str, Any] = Field(default_factory=dict)

    # Security context
    security_issues: list[SecurityIssue] = Field(default_factory=list)
    security_scan_status: str = "pending"
    compliance_status: dict[str, bool] = Field(default_factory=dict)

    # DevOps context
    pipelines: list[BuildPipeline] = Field(default_factory=list)
    deployment_status: str = "not_deployed"
    environments: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Debugging context
    debug_sessions: list[DebugSession] = Field(default_factory=list)
    active_debug_session: str | None = None

    # Documentation context
    documentation: list[DocumentationEntry] = Field(default_factory=list)

    # Execution metadata
    iteration_count: int = 0
    total_tool_calls: int = 0
    thinking_steps: list[str] = Field(default_factory=list)

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    # Helper methods

    def get_pending_todos(self) -> list[Todo]:
        """Get all pending todos."""
        from app.deepagents.core.types import TodoStatus

        return [t for t in self.todos if t.status == TodoStatus.PENDING]

    def get_phase_summary(self) -> str:
        """Get a summary of current SDLC phase progress."""
        phase_counts = {
            SDLCPhase.REQUIREMENTS: len(self.requirements),
            SDLCPhase.DESIGN: len(self.components),
            SDLCPhase.IMPLEMENTATION: len(self.code_files),
            SDLCPhase.TESTING: len(self.test_cases),
            SDLCPhase.SECURITY: len(self.security_issues),
            SDLCPhase.DEVOPS: len(self.pipelines),
            SDLCPhase.DEBUGGING: len(self.debug_sessions),
            SDLCPhase.DOCUMENTATION: len(self.documentation),
            SDLCPhase.REVIEW: len(self.review_issues),
        }

        current = self.current_phase.value
        count = phase_counts.get(self.current_phase, 0)
        return f"Phase: {current} | Items: {count}"

    def get_requirements_summary(self) -> str:
        """Get requirements summary."""
        if not self.requirements:
            return "No requirements captured yet."

        by_priority = {}
        for req in self.requirements:
            p = req.priority.value
            by_priority[p] = by_priority.get(p, 0) + 1

        parts = [f"{len(self.requirements)} requirements:"]
        for p, c in by_priority.items():
            parts.append(f"  {p}: {c}")

        return "\n".join(parts)

    def get_code_summary(self) -> str:
        """Get code files summary."""
        if not self.code_files:
            return "No code files in context."

        total_lines = sum(f.line_count for f in self.code_files.values())
        by_lang = {}
        for f in self.code_files.values():
            l = f.language.value
            by_lang[l] = by_lang.get(l, 0) + 1

        parts = [f"{len(self.code_files)} files, {total_lines} lines:"]
        for lang, count in by_lang.items():
            parts.append(f"  {lang}: {count} files")

        return "\n".join(parts)

    def get_test_summary(self) -> str:
        """Get test cases summary."""
        if not self.test_cases:
            return "No test cases defined."

        by_status = {}
        for tc in self.test_cases:
            s = tc.status
            by_status[s] = by_status.get(s, 0) + 1

        passed = by_status.get("passed", 0)
        total = len(self.test_cases)
        coverage = self.test_coverage

        return f"Tests: {passed}/{total} passed | Coverage: {coverage:.1f}%"

    def get_security_summary(self) -> str:
        """Get security issues summary."""
        if not self.security_issues:
            return "No security issues found."

        by_severity = {}
        for issue in self.security_issues:
            s = issue.severity.value
            by_severity[s] = by_severity.get(s, 0) + 1

        unfixed = sum(1 for i in self.security_issues if not i.fixed)
        parts = [f"{len(self.security_issues)} issues ({unfixed} unfixed):"]
        for sev, count in sorted(by_severity.items()):
            parts.append(f"  {sev}: {count}")

        return "\n".join(parts)

    def get_project_context(self) -> str:
        """Get overall project context summary."""
        parts = [
            f"Project: {self.project_name or 'Unnamed'}",
            f"Phase: {self.current_phase.value}",
            f"Language: {self.primary_language.value}",
            "",
            self.get_requirements_summary(),
            "",
            self.get_code_summary(),
            "",
            self.get_test_summary(),
            "",
            self.get_security_summary(),
        ]

        return "\n".join(parts)

    def add_thinking_step(self, thought: str) -> None:
        """Add a thinking step for transparency."""
        self.thinking_steps.append(f"[{datetime.now().isoformat()}] {thought}")

    def transition_phase(self, new_phase: SDLCPhase) -> None:
        """Transition to a new SDLC phase."""
        self.phase_history.append((self.current_phase, datetime.now()))
        self.current_phase = new_phase
        self.last_activity = datetime.now()
