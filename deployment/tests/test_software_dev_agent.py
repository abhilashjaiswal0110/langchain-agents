"""Comprehensive tests for Software Development DeepAgent.

Tests cover:
- Configuration and state management
- Tool functionality for all 9 specialized modules
- Subagent definitions and routing
- Main agent initialization
- REST API endpoints
"""

import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# Configuration tests
from app.deepagents.config.software_dev_config import (
    SDLCPhase,
    CodeLanguage,
    TestType,
    SecuritySeverity,
    RequirementType,
    RequirementPriority,
    ArchitecturePattern,
    SoftwareDevAgentConfig,
    QualityGateConfig,
    TestGenerationConfig,
    SecurityScanConfig,
    CICDConfig,
    OWASP_TOP_10,
)


class TestConfiguration:
    """Tests for software development agent configuration."""

    def test_sdlc_phases(self):
        """Test all SDLC phases are defined."""
        phases = list(SDLCPhase)
        assert len(phases) == 9
        assert SDLCPhase.REQUIREMENTS in phases
        assert SDLCPhase.DESIGN in phases
        assert SDLCPhase.IMPLEMENTATION in phases
        assert SDLCPhase.REVIEW in phases
        assert SDLCPhase.TESTING in phases
        assert SDLCPhase.SECURITY in phases
        assert SDLCPhase.DEVOPS in phases
        assert SDLCPhase.DEBUGGING in phases
        assert SDLCPhase.DOCUMENTATION in phases

    def test_code_languages(self):
        """Test supported code languages."""
        languages = list(CodeLanguage)
        assert CodeLanguage.PYTHON in languages
        assert CodeLanguage.TYPESCRIPT in languages
        assert CodeLanguage.JAVASCRIPT in languages
        assert CodeLanguage.GO in languages
        assert CodeLanguage.JAVA in languages
        assert CodeLanguage.RUST in languages

    def test_test_types(self):
        """Test supported test types."""
        test_types = list(TestType)
        assert TestType.UNIT in test_types
        assert TestType.INTEGRATION in test_types
        assert TestType.E2E in test_types
        assert TestType.PERFORMANCE in test_types
        assert TestType.SECURITY in test_types

    def test_security_severity_levels(self):
        """Test security severity levels."""
        severities = list(SecuritySeverity)
        assert SecuritySeverity.CRITICAL in severities
        assert SecuritySeverity.HIGH in severities
        assert SecuritySeverity.MEDIUM in severities
        assert SecuritySeverity.LOW in severities
        assert SecuritySeverity.INFO in severities

    def test_requirement_types(self):
        """Test requirement types."""
        req_types = list(RequirementType)
        assert RequirementType.FUNCTIONAL in req_types
        assert RequirementType.NON_FUNCTIONAL in req_types
        assert RequirementType.TECHNICAL in req_types
        assert RequirementType.BUSINESS in req_types

    def test_requirement_priorities(self):
        """Test MoSCoW requirement priorities."""
        priorities = list(RequirementPriority)
        assert RequirementPriority.MUST_HAVE in priorities
        assert RequirementPriority.SHOULD_HAVE in priorities
        assert RequirementPriority.COULD_HAVE in priorities
        assert RequirementPriority.WONT_HAVE in priorities

    def test_architecture_patterns(self):
        """Test architecture patterns."""
        patterns = list(ArchitecturePattern)
        assert ArchitecturePattern.MICROSERVICES in patterns
        assert ArchitecturePattern.MONOLITHIC in patterns
        assert ArchitecturePattern.SERVERLESS in patterns
        assert ArchitecturePattern.EVENT_DRIVEN in patterns
        assert ArchitecturePattern.LAYERED in patterns

    def test_quality_gates_defaults(self):
        """Test quality gates default values."""
        gates = QualityGateConfig()
        assert gates.min_test_coverage == 80.0
        assert gates.max_complexity == 10
        assert gates.max_code_smells == 5
        assert gates.require_security_scan is True
        assert gates.require_documentation is True
        assert gates.require_code_review is True

    def test_test_config_defaults(self):
        """Test test configuration defaults."""
        config = TestGenerationConfig()
        assert config.min_unit_tests == 3
        assert config.target_coverage == 80.0
        assert config.include_edge_cases is True

    def test_security_config_defaults(self):
        """Test security configuration defaults."""
        config = SecurityScanConfig()
        assert config.check_dependencies is True
        assert config.check_owasp_top10 is True
        assert config.check_secrets is True
        assert config.severity_threshold == SecuritySeverity.MEDIUM

    def test_cicd_config_defaults(self):
        """Test CI/CD configuration defaults."""
        config = CICDConfig()
        assert config.platform == "github-actions"
        assert config.enable_auto_deploy is False
        assert "test" in config.build_stages
        assert "build" in config.build_stages

    def test_agent_config_defaults(self):
        """Test main agent configuration defaults."""
        config = SoftwareDevAgentConfig()
        assert config.model == "gpt-4o-mini"
        assert config.max_iterations == 50
        assert config.recursion_limit == 25
        assert CodeLanguage.PYTHON in config.supported_languages

    def test_owasp_top_10(self):
        """Test OWASP Top 10 definitions."""
        assert len(OWASP_TOP_10) == 10
        assert "A01" in OWASP_TOP_10
        assert "Broken Access Control" in OWASP_TOP_10["A01"]


# State management tests
from app.deepagents.software_dev.state import (
    Requirement,
    UserStory,
    ArchitectureComponent,
    APIEndpoint,
    CodeFile,
    CodeReviewIssue,
    TestCase,
    SecurityIssue,
    BuildPipeline,
    DebugSession,
    DocumentationEntry,
    SoftwareDevState,
)


class TestStateModels:
    """Tests for state management models."""

    def test_requirement_model(self):
        """Test Requirement model."""
        req = Requirement(
            id="REQ-001",
            title="User Authentication",
            description="Users should be able to authenticate",
            type=RequirementType.FUNCTIONAL,
            priority=RequirementPriority.MUST_HAVE,
        )
        assert req.id == "REQ-001"
        assert req.type == RequirementType.FUNCTIONAL
        assert req.priority == RequirementPriority.MUST_HAVE
        assert req.status == "draft"

    def test_user_story_model(self):
        """Test UserStory model."""
        story = UserStory(
            id="US-001",
            title="Login Feature",
            as_a="user",
            i_want="to log in",
            so_that="I can access my account",
        )
        assert story.id == "US-001"
        assert story.as_a == "user"
        assert story.story_points is None

    def test_architecture_component_model(self):
        """Test ArchitectureComponent model."""
        component = ArchitectureComponent(
            id="comp-001",
            name="API Gateway",
            type="service",
            description="Handles API routing",
            dependencies=["auth-service"],
        )
        assert component.name == "API Gateway"
        assert "auth-service" in component.dependencies

    def test_api_endpoint_model(self):
        """Test APIEndpoint model."""
        endpoint = APIEndpoint(
            path="/api/users",
            method="GET",
            description="List users",
            request_schema={"type": "object"},
            response_schema={"type": "array"},
        )
        assert endpoint.path == "/api/users"
        assert endpoint.method == "GET"

    def test_code_file_model(self):
        """Test CodeFile model."""
        code_file = CodeFile(
            path="src/main.py",
            language=CodeLanguage.PYTHON,
            content="print('hello')",
        )
        assert code_file.path == "src/main.py"
        assert code_file.language == CodeLanguage.PYTHON
        assert code_file.test_coverage == 0.0

    def test_code_review_issue_model(self):
        """Test CodeReviewIssue model."""
        issue = CodeReviewIssue(
            id="CR-001",
            file_path="src/main.py",
            line_number=10,
            severity="high",
            category="security",
            message="Hardcoded password",
            suggestion="Use environment variable",
        )
        assert issue.id == "CR-001"
        assert issue.severity == "high"
        assert issue.auto_fixable is False

    def test_test_case_model(self):
        """Test TestCase model."""
        test_case = TestCase(
            id="TC-001",
            name="test_login",
            type=TestType.UNIT,
            description="Test login function",
            target_file="tests/test_auth.py",
        )
        assert test_case.id == "TC-001"
        assert test_case.type == TestType.UNIT
        assert test_case.status == "pending"

    def test_security_issue_model(self):
        """Test SecurityIssue model."""
        issue = SecurityIssue(
            id="SEC-001",
            title="SQL Injection",
            severity=SecuritySeverity.CRITICAL,
            category="A03",
            description="SQL injection vulnerability",
            file_path="src/db.py",
            line_number=25,
            recommendation="Use parameterized queries",
        )
        assert issue.id == "SEC-001"
        assert issue.severity == SecuritySeverity.CRITICAL
        assert issue.fixed is False

    def test_build_pipeline_model(self):
        """Test BuildPipeline model."""
        pipeline = BuildPipeline(
            name="CI Pipeline",
            platform="github-actions",
            stages=["build", "test", "deploy"],
        )
        assert pipeline.name == "CI Pipeline"
        assert "test" in pipeline.stages

    def test_debug_session_model(self):
        """Test DebugSession model."""
        session = DebugSession(
            id="DBG-001",
            issue_description="KeyError: 'user'",
            stack_trace="...",
        )
        assert session.id == "DBG-001"
        assert session.root_cause is None
        assert session.status == "investigating"

    def test_documentation_entry_model(self):
        """Test DocumentationEntry model."""
        doc = DocumentationEntry(
            id="DOC-001",
            title="API Documentation",
            type="api",
            content="# API Reference",
            target_path="docs/api.md",
        )
        assert doc.id == "DOC-001"
        assert doc.type == "api"


class TestSoftwareDevState:
    """Tests for main state management."""

    def test_initial_state(self):
        """Test initial state values."""
        state = SoftwareDevState(messages=[])
        assert state.current_phase == SDLCPhase.REQUIREMENTS
        assert len(state.requirements) == 0
        assert len(state.code_files) == 0
        assert state.iteration_count == 0

    def test_get_phase_summary(self):
        """Test phase summary generation."""
        state = SoftwareDevState(messages=[])
        summary = state.get_phase_summary()
        assert "requirements" in summary.lower()
        assert "Phase" in summary

    def test_get_requirements_summary(self):
        """Test requirements summary."""
        state = SoftwareDevState(
            messages=[],
            requirements=[
                Requirement(
                    id="REQ-001",
                    title="Test",
                    description="Test req",
                    type=RequirementType.FUNCTIONAL,
                    priority=RequirementPriority.MUST_HAVE,
                )
            ],
        )
        summary = state.get_requirements_summary()
        assert "1 requirements" in summary
        assert "must_have" in summary

    def test_get_code_summary(self):
        """Test code summary."""
        state = SoftwareDevState(
            messages=[],
            code_files={
                "test.py": CodeFile(
                    path="test.py",
                    language=CodeLanguage.PYTHON,
                    content="print('test')",
                )
            },
        )
        summary = state.get_code_summary()
        assert "1 files" in summary
        assert "python" in summary.lower()

    def test_get_test_summary(self):
        """Test test summary."""
        state = SoftwareDevState(
            messages=[],
            test_cases=[
                TestCase(
                    id="TC-001",
                    name="test_example",
                    type=TestType.UNIT,
                    description="Test",
                    target_file="test.py",
                    status="passed",
                )
            ],
        )
        summary = state.get_test_summary()
        assert "1/1 passed" in summary

    def test_get_security_summary(self):
        """Test security summary."""
        state = SoftwareDevState(
            messages=[],
            security_issues=[
                SecurityIssue(
                    id="SEC-001",
                    title="Test Issue",
                    severity=SecuritySeverity.HIGH,
                    category="A01",
                    description="Test",
                    file_path="test.py",
                    line_number=1,
                    recommendation="Fix it",
                )
            ],
        )
        summary = state.get_security_summary()
        assert "1 issues" in summary
        assert "high" in summary


# Requirements tools smoke tests
from app.deepagents.software_dev.tools.requirements_tools import (
    analyze_requirements,
    extract_user_stories,
    validate_requirements,
    detect_ambiguities,
    generate_acceptance_criteria,
)


class TestRequirementsTools:
    """Smoke tests for requirements intelligence tools."""

    def test_analyze_requirements(self):
        """Test requirements analysis returns valid JSON."""
        result = json.loads(analyze_requirements.invoke({
            "requirements_text": "Users must be able to log in"
        }))
        assert "total_requirements" in result
        assert "requirements" in result

    def test_extract_user_stories(self):
        """Test user story extraction returns valid JSON."""
        result = json.loads(extract_user_stories.invoke({
            "requirements_text": "Login feature"
        }))
        assert "total_stories" in result
        assert "user_stories" in result

    def test_detect_ambiguities(self):
        """Test ambiguity detection returns valid JSON."""
        result = json.loads(detect_ambiguities.invoke({
            "text": "The system should be fast"
        }))
        assert "total_ambiguities" in result
        assert "ambiguities" in result

    def test_generate_acceptance_criteria(self):
        """Test acceptance criteria generation returns valid JSON."""
        result = json.loads(generate_acceptance_criteria.invoke({
            "requirement_text": "User login"
        }))
        assert "acceptance_criteria" in result


# Architecture tools smoke tests
from app.deepagents.software_dev.tools.architecture_tools import (
    design_architecture,
    create_api_spec,
    suggest_tech_stack,
)


class TestArchitectureTools:
    """Smoke tests for architecture tools."""

    def test_design_architecture(self):
        """Test architecture design returns valid JSON."""
        result = json.loads(design_architecture.invoke({
            "requirements_summary": "E-commerce platform",
            "pattern": "microservices",
        }))
        assert "components" in result
        assert "pattern" in result

    def test_create_api_spec(self):
        """Test API spec creation returns valid JSON."""
        result = json.loads(create_api_spec.invoke({
            "resource_name": "users",
        }))
        assert "endpoints" in result

    def test_suggest_tech_stack(self):
        """Test tech stack suggestion returns valid JSON."""
        result = json.loads(suggest_tech_stack.invoke({
            "project_type": "web_app",
        }))
        assert isinstance(result, dict)
        # Check that we have tech recommendations
        assert "recommended_stack" in result
        assert "project_type" in result


# Code generation tools smoke tests
from app.deepagents.software_dev.tools.codegen_tools import (
    generate_code,
    format_code,
)


class TestCodeGenTools:
    """Smoke tests for code generation tools."""

    def test_generate_code(self):
        """Test code generation returns valid JSON."""
        result = json.loads(generate_code.invoke({
            "description": "Calculate sum of numbers",
            "language": "python",
        }))
        assert "code" in result

    def test_format_code(self):
        """Test code formatting returns valid JSON."""
        result = json.loads(format_code.invoke({
            "code": "def foo():pass",
            "language": "python",
        }))
        assert "formatted_code" in result


# Code review tools smoke tests
from app.deepagents.software_dev.tools.review_tools import (
    review_code,
    check_code_style,
)


class TestReviewTools:
    """Smoke tests for code review tools."""

    def test_review_code(self):
        """Test code review returns valid JSON."""
        result = json.loads(review_code.invoke({
            "code": "print('hello')",
            "language": "python",
        }))
        assert isinstance(result, dict)

    def test_check_code_style(self):
        """Test code style check returns valid JSON."""
        result = json.loads(check_code_style.invoke({
            "code": "def foo():pass",
            "language": "python",
        }))
        assert isinstance(result, dict)


# Testing tools smoke tests
from app.deepagents.software_dev.tools.testing_tools import (
    generate_unit_tests,
    generate_test_data,
)


class TestTestingTools:
    """Smoke tests for testing automation tools."""

    def test_generate_unit_tests(self):
        """Test unit test generation returns valid JSON."""
        result = json.loads(generate_unit_tests.invoke({
            "code": "def add(a, b): return a + b",
            "language": "python",
        }))
        assert "tests" in result
        assert "framework" in result

    def test_generate_test_data(self):
        """Test test data generation returns valid JSON."""
        result = json.loads(generate_test_data.invoke({
            "data_schema": {"name": "string", "age": "integer"},
            "count": 3,
        }))
        assert "data" in result
        assert len(result["data"]) == 3


# Security tools smoke tests
from app.deepagents.software_dev.tools.security_tools import (
    scan_security_issues,
    detect_secrets,
)


class TestSecurityTools:
    """Smoke tests for security tools."""

    def test_scan_security_issues(self):
        """Test security scanning returns valid JSON."""
        result = json.loads(scan_security_issues.invoke({
            "code": "password = 'secret123'",
            "language": "python",
        }))
        assert "issues" in result

    def test_detect_secrets(self):
        """Test secret detection returns valid JSON."""
        result = json.loads(detect_secrets.invoke({
            "code": "api_key = 'sk-123'",
        }))
        assert "secrets" in result


# DevOps tools smoke tests
from app.deepagents.software_dev.tools.devops_tools import (
    generate_dockerfile,
    create_ci_pipeline,
)


class TestDevOpsTools:
    """Smoke tests for DevOps tools."""

    def test_generate_dockerfile(self):
        """Test Dockerfile generation returns valid JSON."""
        result = json.loads(generate_dockerfile.invoke({
            "language": "python",
            "framework": "fastapi",
        }))
        assert "dockerfile" in result

    def test_create_ci_pipeline(self):
        """Test CI pipeline creation returns valid JSON."""
        result = json.loads(create_ci_pipeline.invoke({
            "project_name": "my-project",
            "platform": "github-actions",
            "language": "python",
        }))
        assert "pipeline" in result or "config" in result


# Debugging tools smoke tests
from app.deepagents.software_dev.tools.debugging_tools import (
    analyze_error,
    propose_fix,
)


class TestDebuggingTools:
    """Smoke tests for debugging tools."""

    def test_analyze_error(self):
        """Test error analysis returns valid JSON."""
        result = json.loads(analyze_error.invoke({
            "error_message": "KeyError: 'user'",
            "language": "python",
        }))
        assert isinstance(result, dict)

    def test_propose_fix(self):
        """Test fix proposal returns valid JSON."""
        result = json.loads(propose_fix.invoke({
            "issue_description": "NullPointerError when accessing None object",
            "language": "python",
        }))
        assert isinstance(result, dict)
        assert "proposed_fixes" in result
        assert "issue" in result


# Documentation tools smoke tests
from app.deepagents.software_dev.tools.documentation_tools import (
    create_readme,
    add_inline_comments,
)


class TestDocumentationTools:
    """Smoke tests for documentation tools."""

    def test_create_readme(self):
        """Test README creation returns valid JSON."""
        result = json.loads(create_readme.invoke({
            "project_name": "My Project",
            "description": "A test project",
        }))
        assert "content" in result or "readme" in result
        assert result.get("project_name") == "My Project"

    def test_add_inline_comments(self):
        """Test inline comment addition returns valid JSON."""
        result = json.loads(add_inline_comments.invoke({
            "code": "def add(a, b): return a + b",
            "language": "python",
        }))
        assert "commented_code" in result


# Subagent tests
from app.deepagents.software_dev.subagents import (
    REQUIREMENTS_AGENT,
    ARCHITECTURE_AGENT,
    CODEGEN_AGENT,
    REVIEW_AGENT,
    TESTING_AGENT,
    DEBUGGING_AGENT,
    SECURITY_AGENT,
    DEVOPS_AGENT,
    DOCUMENTATION_AGENT,
    get_all_subagents,
    get_subagent_by_name,
    get_subagent_tools,
    get_subagent_for_phase,
)


class TestSubagents:
    """Tests for subagent definitions."""

    def test_all_subagents_defined(self):
        """Test all 9 subagents are defined."""
        subagents = get_all_subagents()
        assert len(subagents) == 9

    def test_requirements_agent(self):
        """Test requirements agent definition."""
        assert REQUIREMENTS_AGENT.name == "requirements-intelligence"
        assert len(REQUIREMENTS_AGENT.tools) == 6
        assert "requirements" in REQUIREMENTS_AGENT.description.lower()

    def test_architecture_agent(self):
        """Test architecture agent definition."""
        assert ARCHITECTURE_AGENT.name == "architecture-design"
        assert len(ARCHITECTURE_AGENT.tools) == 6
        assert "architecture" in ARCHITECTURE_AGENT.description.lower()

    def test_codegen_agent(self):
        """Test code generation agent definition."""
        assert CODEGEN_AGENT.name == "code-generator"
        assert len(CODEGEN_AGENT.tools) == 6

    def test_review_agent(self):
        """Test code review agent definition."""
        assert REVIEW_AGENT.name == "code-reviewer"
        assert len(REVIEW_AGENT.tools) == 6

    def test_testing_agent(self):
        """Test testing agent definition."""
        assert TESTING_AGENT.name == "testing-automation"
        assert len(TESTING_AGENT.tools) == 6

    def test_debugging_agent(self):
        """Test debugging agent definition."""
        assert DEBUGGING_AGENT.name == "debugging-optimization"
        assert len(DEBUGGING_AGENT.tools) == 6

    def test_security_agent(self):
        """Test security agent definition."""
        assert SECURITY_AGENT.name == "security-compliance"
        assert len(SECURITY_AGENT.tools) == 6

    def test_devops_agent(self):
        """Test DevOps agent definition."""
        assert DEVOPS_AGENT.name == "devops-integration"
        assert len(DEVOPS_AGENT.tools) == 6

    def test_documentation_agent(self):
        """Test documentation agent definition."""
        assert DOCUMENTATION_AGENT.name == "documentation"
        assert len(DOCUMENTATION_AGENT.tools) == 6

    def test_get_subagent_by_name(self):
        """Test getting subagent by name."""
        agent = get_subagent_by_name("requirements-intelligence")
        assert agent is not None
        assert agent.name == "requirements-intelligence"

        # Non-existent agent
        agent = get_subagent_by_name("non-existent")
        assert agent is None

    def test_get_subagent_tools(self):
        """Test getting subagent tools."""
        tools = get_subagent_tools("code-generator")
        assert len(tools) == 6

        # Non-existent agent
        tools = get_subagent_tools("non-existent")
        assert len(tools) == 0

    def test_get_subagent_for_phase(self):
        """Test getting subagent for SDLC phase."""
        agent = get_subagent_for_phase(SDLCPhase.REQUIREMENTS)
        assert agent.name == "requirements-intelligence"

        agent = get_subagent_for_phase(SDLCPhase.DESIGN)
        assert agent.name == "architecture-design"

        agent = get_subagent_for_phase(SDLCPhase.IMPLEMENTATION)
        assert agent.name == "code-generator"

        agent = get_subagent_for_phase(SDLCPhase.TESTING)
        assert agent.name == "testing-automation"

        agent = get_subagent_for_phase(SDLCPhase.SECURITY)
        assert agent.name == "security-compliance"


# Main agent tests
class TestSoftwareDevAgent:
    """Tests for main Software Development DeepAgent."""

    def test_agent_import(self):
        """Test agent can be imported."""
        from app.deepagents.software_dev.software_dev_agent import (
            SoftwareDevDeepAgent,
            create_software_dev_agent,
        )
        assert SoftwareDevDeepAgent is not None
        assert create_software_dev_agent is not None


# API Routes tests
class TestAPIRoutes:
    """Tests for REST API routes."""

    def test_routes_import(self):
        """Test routes can be imported."""
        from app.deepagents.software_dev.routes import router
        assert router is not None
        assert router.prefix == "/api/software-dev-agent"

    def test_request_models(self):
        """Test request/response models."""
        from app.deepagents.software_dev.routes import (
            StartSessionRequest,
            ChatRequest,
            ChatResponse,
            SessionStateResponse,
            PhaseTransitionRequest,
        )

        # Test StartSessionRequest
        req = StartSessionRequest(
            user_id="user123",
            project_name="Test Project",
        )
        assert req.user_id == "user123"
        assert req.project_name == "Test Project"

        # Test ChatRequest
        chat_req = ChatRequest(
            session_id="session123",
            message="Hello",
        )
        assert chat_req.session_id == "session123"
        assert chat_req.message == "Hello"

        # Test ChatResponse
        chat_resp = ChatResponse(
            session_id="session123",
            response="Hi there!",
            phase="requirements",
        )
        assert chat_resp.session_id == "session123"
        assert chat_resp.todos == []

        # Test SessionStateResponse
        state_resp = SessionStateResponse(
            session_id="session123",
            phase="design",
        )
        assert state_resp.requirements_count == 0

        # Test PhaseTransitionRequest
        phase_req = PhaseTransitionRequest(
            session_id="session123",
            new_phase="implementation",
        )
        assert phase_req.new_phase == "implementation"


# Integration test - verify full import chain works
class TestIntegration:
    """Integration tests for the complete module."""

    def test_full_import_chain(self):
        """Test that all modules can be imported together."""
        from app.deepagents.config.software_dev_config import SoftwareDevAgentConfig
        from app.deepagents.software_dev.state import SoftwareDevState
        from app.deepagents.software_dev.subagents import get_all_subagents
        from app.deepagents.software_dev.tools import (
            analyze_requirements,
            design_architecture,
            generate_code,
            review_code,
            generate_unit_tests,
            scan_security_issues,
            create_ci_pipeline,
            analyze_error,
            create_readme,
        )
        from app.deepagents.software_dev.routes import router

        # Verify counts
        assert len(get_all_subagents()) == 9
        assert router.prefix == "/api/software-dev-agent"

    def test_tool_count(self):
        """Test that all 54 tools are available."""
        from app.deepagents.software_dev.tools import __all__ as all_tools
        assert len(all_tools) == 54

    def test_phase_subagent_mapping(self):
        """Test all phases have corresponding subagents."""
        from app.deepagents.software_dev.subagents import get_subagent_for_phase

        for phase in SDLCPhase:
            agent = get_subagent_for_phase(phase)
            assert agent is not None, f"No subagent for phase {phase}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
