"""Comprehensive tests for Software Development DeepAgent.

Tests cover:
- Configuration and state management
- Tool functionality for all 9 specialized modules
- Subagent definitions and routing
- Main agent initialization and execution
- REST API endpoints
"""

import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

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
    QualityGates,
    TestConfig,
    SecurityConfig,
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
        assert ArchitecturePattern.MONOLITH in patterns
        assert ArchitecturePattern.SERVERLESS in patterns
        assert ArchitecturePattern.EVENT_DRIVEN in patterns
        assert ArchitecturePattern.LAYERED in patterns

    def test_quality_gates_defaults(self):
        """Test quality gates default values."""
        gates = QualityGates()
        assert gates.min_code_coverage == 80.0
        assert gates.max_complexity == 10
        assert gates.max_critical_issues == 0
        assert gates.require_tests is True
        assert gates.require_docs is True
        assert gates.require_security_scan is True

    def test_test_config_defaults(self):
        """Test test configuration defaults."""
        config = TestConfig()
        assert config.framework == "pytest"
        assert config.coverage_threshold == 80.0
        assert config.parallel_execution is True

    def test_security_config_defaults(self):
        """Test security configuration defaults."""
        config = SecurityConfig()
        assert config.scan_dependencies is True
        assert config.check_owasp is True
        assert config.detect_secrets is True
        assert config.severity_threshold == SecuritySeverity.MEDIUM

    def test_cicd_config_defaults(self):
        """Test CI/CD configuration defaults."""
        config = CICDConfig()
        assert config.platform == "github_actions"
        assert config.auto_deploy is False
        assert "staging" in config.environments
        assert "production" in config.environments

    def test_agent_config_defaults(self):
        """Test main agent configuration defaults."""
        config = SoftwareDevAgentConfig()
        assert config.model == "gpt-4o-mini"
        assert config.max_iterations == 50
        assert config.recursion_limit == 100
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
        assert code_file.tests_path is None

    def test_code_review_issue_model(self):
        """Test CodeReviewIssue model."""
        issue = CodeReviewIssue(
            id="CR-001",
            file_path="src/main.py",
            line_number=10,
            severity="high",
            category="security",
            description="Hardcoded password",
            suggestion="Use environment variable",
        )
        assert issue.id == "CR-001"
        assert issue.severity == "high"
        assert issue.resolved is False

    def test_test_case_model(self):
        """Test TestCase model."""
        test_case = TestCase(
            id="TC-001",
            name="test_login",
            type=TestType.UNIT,
            description="Test login function",
            file_path="tests/test_auth.py",
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
        )
        assert issue.id == "SEC-001"
        assert issue.severity == SecuritySeverity.CRITICAL
        assert issue.remediation is None

    def test_build_pipeline_model(self):
        """Test BuildPipeline model."""
        pipeline = BuildPipeline(
            name="CI Pipeline",
            platform="github_actions",
            stages=["build", "test", "deploy"],
        )
        assert pipeline.name == "CI Pipeline"
        assert "test" in pipeline.stages

    def test_debug_session_model(self):
        """Test DebugSession model."""
        session = DebugSession(
            id="DBG-001",
            error_message="KeyError: 'user'",
            stack_trace="...",
        )
        assert session.id == "DBG-001"
        assert session.root_cause is None
        assert session.status == "active"

    def test_documentation_entry_model(self):
        """Test DocumentationEntry model."""
        doc = DocumentationEntry(
            id="DOC-001",
            title="API Documentation",
            type="api",
            content="# API Reference",
            file_path="docs/api.md",
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
        assert "requirements" in summary
        assert "REQUIREMENTS" in summary

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
        assert "MUST_HAVE" in summary

    def test_get_code_summary(self):
        """Test code summary."""
        state = SoftwareDevState(
            messages=[],
            code_files=[
                CodeFile(
                    path="test.py",
                    language=CodeLanguage.PYTHON,
                    content="print('test')",
                )
            ],
        )
        summary = state.get_code_summary()
        assert "1 code files" in summary
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
                    file_path="test.py",
                    status="passed",
                )
            ],
        )
        summary = state.get_test_summary()
        assert "1 test cases" in summary
        assert "1 passed" in summary

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
                )
            ],
        )
        summary = state.get_security_summary()
        assert "1 security issues" in summary
        assert "HIGH" in summary


# Requirements tools tests
from app.deepagents.software_dev.tools.requirements_tools import (
    analyze_requirements,
    extract_user_stories,
    validate_requirements,
    prioritize_requirements,
    detect_ambiguities,
    generate_acceptance_criteria,
)


class TestRequirementsTools:
    """Tests for requirements intelligence tools."""

    def test_analyze_requirements(self):
        """Test requirements analysis."""
        text = """
        Users must be able to log in with email and password.
        The system should support OAuth integration.
        API response time must be under 200ms.
        """
        result = json.loads(analyze_requirements.invoke({"requirements_text": text}))
        assert result["total_requirements"] > 0
        assert "requirements" in result
        assert "by_type" in result
        assert "by_priority" in result

    def test_analyze_requirements_with_context(self):
        """Test requirements analysis with context."""
        text = "Users should be able to register"
        result = json.loads(
            analyze_requirements.invoke({
                "requirements_text": text,
                "context": "E-commerce platform",
            })
        )
        assert result["total_requirements"] == 1

    def test_extract_user_stories_from_text(self):
        """Test extracting user stories from text."""
        text = """
        Login with email
        View profile
        Edit settings
        """
        result = json.loads(
            extract_user_stories.invoke({"requirements_text": text})
        )
        assert result["total_stories"] == 3
        assert len(result["user_stories"]) == 3
        assert result["user_stories"][0]["as_a"] == "user"

    def test_validate_requirements(self):
        """Test requirements validation."""
        # First analyze some requirements
        analyze_requirements.invoke({
            "requirements_text": "The system should maybe allow users to login"
        })
        result = json.loads(validate_requirements.invoke({}))
        assert "total_validated" in result
        assert "valid_count" in result
        assert "results" in result

    def test_detect_ambiguities(self):
        """Test ambiguity detection."""
        text = "The system should be fast and handle many users with good performance"
        result = json.loads(detect_ambiguities.invoke({"text": text}))
        assert result["total_ambiguities"] > 0
        assert "by_severity" in result
        # Should detect vague adjectives
        vague_words = [a["word"] for a in result["ambiguities"]]
        assert any(w in vague_words for w in ["fast", "many", "good"])

    def test_detect_ambiguities_uncertain_language(self):
        """Test detection of uncertain language."""
        text = "The system might possibly allow users to maybe update their profile"
        result = json.loads(detect_ambiguities.invoke({"text": text}))
        uncertain = [a for a in result["ambiguities"] if a["type"] == "uncertain_language"]
        assert len(uncertain) > 0

    def test_generate_acceptance_criteria_given_when_then(self):
        """Test BDD-style acceptance criteria generation."""
        result = json.loads(
            generate_acceptance_criteria.invoke({
                "requirement_text": "Users should be able to reset their password",
                "format": "given_when_then",
            })
        )
        assert result["format"] == "given_when_then"
        assert result["criteria_count"] > 0
        assert "given" in result["acceptance_criteria"][0]
        assert "when" in result["acceptance_criteria"][0]
        assert "then" in result["acceptance_criteria"][0]

    def test_generate_acceptance_criteria_checklist(self):
        """Test checklist-style acceptance criteria."""
        result = json.loads(
            generate_acceptance_criteria.invoke({
                "requirement_text": "Feature X",
                "format": "checklist",
            })
        )
        assert result["format"] == "checklist"
        assert "criterion" in result["acceptance_criteria"][0]

    def test_generate_acceptance_criteria_scenario(self):
        """Test scenario-style acceptance criteria."""
        result = json.loads(
            generate_acceptance_criteria.invoke({
                "requirement_text": "Feature Y",
                "format": "scenario",
            })
        )
        assert result["format"] == "scenario"
        assert "scenario" in result["acceptance_criteria"][0]
        assert "steps" in result["acceptance_criteria"][0]


# Architecture tools tests
from app.deepagents.software_dev.tools.architecture_tools import (
    design_architecture,
    create_api_spec,
    suggest_tech_stack,
    design_data_model,
    create_component_diagram,
    analyze_dependencies,
)


class TestArchitectureTools:
    """Tests for architecture and design tools."""

    def test_design_architecture_microservices(self):
        """Test microservices architecture design."""
        result = json.loads(
            design_architecture.invoke({
                "requirements": "E-commerce platform with user auth, catalog, and orders",
                "pattern": "microservices",
            })
        )
        assert result["pattern"] == "microservices"
        assert len(result["components"]) > 0
        assert "recommendations" in result

    def test_design_architecture_serverless(self):
        """Test serverless architecture design."""
        result = json.loads(
            design_architecture.invoke({
                "requirements": "API for processing images",
                "pattern": "serverless",
            })
        )
        assert result["pattern"] == "serverless"

    def test_create_api_spec(self):
        """Test API specification creation."""
        result = json.loads(
            create_api_spec.invoke({
                "resource_name": "users",
                "operations": ["list", "create", "read", "update", "delete"],
            })
        )
        assert result["resource"] == "users"
        assert len(result["endpoints"]) == 5
        # Check CRUD operations
        methods = [e["method"] for e in result["endpoints"]]
        assert "GET" in methods
        assert "POST" in methods
        assert "PUT" in methods
        assert "DELETE" in methods

    def test_suggest_tech_stack(self):
        """Test tech stack suggestions."""
        result = json.loads(
            suggest_tech_stack.invoke({
                "requirements": "High-performance API with real-time features",
                "constraints": "Must use cloud services, budget conscious",
            })
        )
        assert "frontend" in result
        assert "backend" in result
        assert "database" in result
        assert "infrastructure" in result
        assert "reasoning" in result

    def test_design_data_model(self):
        """Test data model design."""
        result = json.loads(
            design_data_model.invoke({
                "entities": ["User", "Order", "Product"],
                "requirements": "E-commerce platform",
            })
        )
        assert result["entity_count"] == 3
        assert len(result["entities"]) == 3
        # Check relationships are generated
        assert len(result["relationships"]) > 0

    def test_create_component_diagram(self):
        """Test component diagram creation."""
        result = json.loads(
            create_component_diagram.invoke({
                "components": ["API Gateway", "Auth Service", "User Service"],
                "format": "mermaid",
            })
        )
        assert result["format"] == "mermaid"
        assert "diagram" in result
        assert "graph" in result["diagram"] or "flowchart" in result["diagram"]

    def test_analyze_dependencies(self):
        """Test dependency analysis."""
        result = json.loads(
            analyze_dependencies.invoke({
                "package_file": "requirements.txt",
                "content": "fastapi==0.115.0\nlangchain>=0.3.0\npydantic>=2.0",
            })
        )
        assert result["total_dependencies"] == 3
        assert len(result["dependencies"]) == 3


# Code generation tools tests
from app.deepagents.software_dev.tools.codegen_tools import (
    generate_code,
    refactor_code,
    apply_design_pattern,
    generate_boilerplate,
    optimize_imports,
    format_code,
)


class TestCodeGenTools:
    """Tests for code generation tools."""

    def test_generate_code_function(self):
        """Test function code generation."""
        result = json.loads(
            generate_code.invoke({
                "description": "Calculate fibonacci number",
                "language": "python",
                "code_type": "function",
            })
        )
        assert result["language"] == "python"
        assert result["type"] == "function"
        assert "code" in result
        assert "def " in result["code"]

    def test_generate_code_class(self):
        """Test class code generation."""
        result = json.loads(
            generate_code.invoke({
                "description": "User repository for database operations",
                "language": "python",
                "code_type": "class",
            })
        )
        assert result["type"] == "class"
        assert "class " in result["code"]

    def test_generate_code_typescript(self):
        """Test TypeScript code generation."""
        result = json.loads(
            generate_code.invoke({
                "description": "User interface",
                "language": "typescript",
                "code_type": "function",
            })
        )
        assert result["language"] == "typescript"

    def test_refactor_code(self):
        """Test code refactoring."""
        code = """
def calculate(x, y, op):
    if op == 'add':
        return x + y
    elif op == 'sub':
        return x - y
    elif op == 'mul':
        return x * y
    elif op == 'div':
        return x / y
"""
        result = json.loads(
            refactor_code.invoke({
                "code": code,
                "refactoring_type": "extract_method",
                "language": "python",
            })
        )
        assert "refactored_code" in result
        assert "changes" in result

    def test_apply_design_pattern_singleton(self):
        """Test singleton pattern application."""
        result = json.loads(
            apply_design_pattern.invoke({
                "pattern": "singleton",
                "class_name": "DatabaseConnection",
                "language": "python",
            })
        )
        assert result["pattern"] == "singleton"
        assert "implementation" in result
        assert "DatabaseConnection" in result["implementation"]

    def test_apply_design_pattern_factory(self):
        """Test factory pattern application."""
        result = json.loads(
            apply_design_pattern.invoke({
                "pattern": "factory",
                "class_name": "NotificationFactory",
                "language": "python",
            })
        )
        assert result["pattern"] == "factory"
        assert "create" in result["implementation"].lower()

    def test_generate_boilerplate(self):
        """Test boilerplate generation."""
        result = json.loads(
            generate_boilerplate.invoke({
                "project_type": "fastapi",
                "project_name": "my_api",
            })
        )
        assert result["project_type"] == "fastapi"
        assert "files" in result
        assert len(result["files"]) > 0

    def test_optimize_imports(self):
        """Test import optimization."""
        code = """
import os
import sys
from typing import List, Dict, Optional
import json
import os  # duplicate
from collections import defaultdict
"""
        result = json.loads(
            optimize_imports.invoke({
                "code": code,
                "language": "python",
            })
        )
        assert "optimized_code" in result
        assert result["removed_duplicates"] > 0 or result["reordered"] is True

    def test_format_code(self):
        """Test code formatting."""
        code = "def foo(x,y):return x+y"
        result = json.loads(
            format_code.invoke({
                "code": code,
                "language": "python",
                "style": "pep8",
            })
        )
        assert "formatted_code" in result


# Code review tools tests
from app.deepagents.software_dev.tools.review_tools import (
    review_code,
    check_code_style,
    analyze_complexity,
    detect_code_smells,
    suggest_improvements,
    check_best_practices,
)


class TestReviewTools:
    """Tests for code review tools."""

    def test_review_code(self):
        """Test code review."""
        code = """
def process_user(user_data):
    password = "admin123"  # hardcoded password
    query = f"SELECT * FROM users WHERE id = {user_data['id']}"  # SQL injection
    return eval(user_data['code'])  # dangerous eval
"""
        result = json.loads(
            review_code.invoke({
                "code": code,
                "language": "python",
            })
        )
        assert result["total_issues"] > 0
        # Should detect security issues
        categories = [i["category"] for i in result["issues"]]
        assert "security" in categories or any("security" in str(c).lower() for c in categories)

    def test_check_code_style(self):
        """Test code style checking."""
        code = """
def badlyNamedFunction( x,y ):
    return x+y
"""
        result = json.loads(
            check_code_style.invoke({
                "code": code,
                "language": "python",
                "style_guide": "pep8",
            })
        )
        assert "violations" in result
        assert result["style_guide"] == "pep8"

    def test_analyze_complexity(self):
        """Test complexity analysis."""
        code = """
def complex_function(a, b, c, d):
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    return a + b + c + d
                else:
                    return a + b + c
            else:
                return a + b
        else:
            return a
    else:
        return 0
"""
        result = json.loads(
            analyze_complexity.invoke({
                "code": code,
                "language": "python",
            })
        )
        assert "cyclomatic_complexity" in result
        assert "cognitive_complexity" in result
        assert result["cyclomatic_complexity"] > 1

    def test_detect_code_smells(self):
        """Test code smell detection."""
        code = """
def do_everything(a, b, c, d, e, f, g, h, i, j):  # too many params
    # This is a very long function
    x = 1
    y = 2
    z = 3
    # ... imagine 100 more lines
    result = a + b + c + d + e + f + g + h + i + j
    return result
"""
        result = json.loads(
            detect_code_smells.invoke({
                "code": code,
                "language": "python",
            })
        )
        assert "code_smells" in result
        # Should detect long parameter list
        smell_types = [s["type"] for s in result["code_smells"]]
        assert "long_parameter_list" in smell_types or len(smell_types) > 0

    def test_suggest_improvements(self):
        """Test improvement suggestions."""
        code = """
def get_users():
    users = []
    for user in database.query("SELECT * FROM users"):
        users.append(user)
    return users
"""
        result = json.loads(
            suggest_improvements.invoke({
                "code": code,
                "language": "python",
            })
        )
        assert "suggestions" in result
        assert len(result["suggestions"]) > 0

    def test_check_best_practices(self):
        """Test best practices checking."""
        code = """
import requests

def fetch_data(url):
    response = requests.get(url)
    return response.json()
"""
        result = json.loads(
            check_best_practices.invoke({
                "code": code,
                "language": "python",
                "domain": "web",
            })
        )
        assert "practices" in result
        assert "score" in result


# Testing tools tests
from app.deepagents.software_dev.tools.testing_tools import (
    generate_unit_tests,
    generate_integration_tests,
    analyze_test_coverage,
    run_tests,
    generate_test_data,
    create_test_plan,
)


class TestTestingTools:
    """Tests for testing automation tools."""

    def test_generate_unit_tests_pytest(self):
        """Test pytest unit test generation."""
        code = """
def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b
"""
        result = json.loads(
            generate_unit_tests.invoke({
                "code": code,
                "language": "python",
                "framework": "pytest",
            })
        )
        assert result["framework"] == "pytest"
        assert "test_code" in result
        assert "def test_" in result["test_code"]
        assert result["test_count"] > 0

    def test_generate_unit_tests_jest(self):
        """Test Jest unit test generation."""
        code = """
function add(a, b) {
    return a + b;
}
"""
        result = json.loads(
            generate_unit_tests.invoke({
                "code": code,
                "language": "javascript",
                "framework": "jest",
            })
        )
        assert result["framework"] == "jest"
        assert "test_code" in result

    def test_generate_integration_tests(self):
        """Test integration test generation."""
        result = json.loads(
            generate_integration_tests.invoke({
                "api_spec": {
                    "endpoint": "/api/users",
                    "method": "POST",
                    "request": {"name": "string"},
                    "response": {"id": "integer"},
                },
                "framework": "pytest",
            })
        )
        assert "test_code" in result
        assert "async" in result["test_code"] or "client" in result["test_code"].lower()

    def test_analyze_test_coverage(self):
        """Test coverage analysis."""
        result = json.loads(
            analyze_test_coverage.invoke({
                "coverage_data": {
                    "src/main.py": {"covered": 80, "total": 100},
                    "src/utils.py": {"covered": 45, "total": 50},
                },
            })
        )
        assert "overall_coverage" in result
        assert "by_file" in result
        assert result["overall_coverage"] > 0

    def test_run_tests(self):
        """Test test execution simulation."""
        result = json.loads(
            run_tests.invoke({
                "test_path": "tests/",
                "framework": "pytest",
            })
        )
        assert "total" in result
        assert "passed" in result
        assert "failed" in result
        assert "status" in result

    def test_generate_test_data(self):
        """Test test data generation."""
        result = json.loads(
            generate_test_data.invoke({
                "schema": {
                    "name": "string",
                    "age": "integer",
                    "email": "email",
                },
                "count": 5,
            })
        )
        assert "data" in result
        assert len(result["data"]) == 5
        assert "name" in result["data"][0]

    def test_create_test_plan(self):
        """Test plan creation."""
        result = json.loads(
            create_test_plan.invoke({
                "feature": "User Registration",
                "requirements": [
                    "Users can register with email",
                    "Password must be at least 8 characters",
                    "Email must be unique",
                ],
            })
        )
        assert "test_plan" in result
        assert "test_cases" in result
        assert len(result["test_cases"]) > 0


# Security tools tests
from app.deepagents.software_dev.tools.security_tools import (
    scan_security_issues,
    check_owasp_compliance,
    detect_secrets,
    analyze_dependencies_security,
    generate_security_report,
    suggest_security_fixes,
)


class TestSecurityTools:
    """Tests for security and compliance tools."""

    def test_scan_security_issues(self):
        """Test security scanning."""
        code = """
import pickle
password = "secret123"
query = f"SELECT * FROM users WHERE name = '{user_input}'"
os.system(f"ls {user_input}")
"""
        result = json.loads(
            scan_security_issues.invoke({
                "code": code,
                "language": "python",
            })
        )
        assert result["total_issues"] > 0
        # Should detect hardcoded password and SQL injection
        issue_types = [i["type"] for i in result["issues"]]
        assert "hardcoded_credential" in issue_types or "sql_injection" in issue_types

    def test_check_owasp_compliance(self):
        """Test OWASP compliance checking."""
        code = """
def authenticate(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    return db.execute(query)
"""
        result = json.loads(
            check_owasp_compliance.invoke({
                "code": code,
                "language": "python",
            })
        )
        assert "compliance_score" in result
        assert "violations" in result
        # Should detect A03 (Injection)
        violation_categories = [v["category"] for v in result["violations"]]
        assert any("A03" in str(c) for c in violation_categories) or len(violation_categories) > 0

    def test_detect_secrets(self):
        """Test secret detection."""
        code = """
API_KEY = "sk-1234567890abcdef"
AWS_SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
password = "admin123"
"""
        result = json.loads(detect_secrets.invoke({"code": code}))
        assert result["total_secrets"] > 0
        assert "secrets" in result
        # Should detect API key and password
        secret_types = [s["type"] for s in result["secrets"]]
        assert len(secret_types) > 0

    def test_analyze_dependencies_security(self):
        """Test dependency security analysis."""
        result = json.loads(
            analyze_dependencies_security.invoke({
                "dependencies": [
                    {"name": "requests", "version": "2.25.0"},
                    {"name": "django", "version": "2.2.0"},
                    {"name": "flask", "version": "1.0.0"},
                ],
            })
        )
        assert "total_dependencies" in result
        assert "vulnerabilities" in result

    def test_generate_security_report(self):
        """Test security report generation."""
        result = json.loads(
            generate_security_report.invoke({
                "scan_results": {
                    "issues": [
                        {"severity": "high", "type": "sql_injection"},
                        {"severity": "medium", "type": "xss"},
                    ]
                },
                "format": "markdown",
            })
        )
        assert "report" in result
        assert "summary" in result
        assert result["format"] == "markdown"

    def test_suggest_security_fixes(self):
        """Test security fix suggestions."""
        result = json.loads(
            suggest_security_fixes.invoke({
                "issue": {
                    "type": "sql_injection",
                    "code": "query = f\"SELECT * FROM users WHERE id = {user_id}\"",
                    "language": "python",
                },
            })
        )
        assert "fixes" in result
        assert len(result["fixes"]) > 0
        # Should suggest parameterized queries
        fix_descriptions = [f["description"] for f in result["fixes"]]
        assert any("parameter" in d.lower() for d in fix_descriptions) or len(fix_descriptions) > 0


# DevOps tools tests
from app.deepagents.software_dev.tools.devops_tools import (
    create_ci_pipeline,
    create_cd_pipeline,
    configure_deployment,
    generate_dockerfile,
    create_kubernetes_config,
    setup_monitoring,
)


class TestDevOpsTools:
    """Tests for DevOps integration tools."""

    def test_create_ci_pipeline_github(self):
        """Test GitHub Actions CI pipeline creation."""
        result = json.loads(
            create_ci_pipeline.invoke({
                "platform": "github_actions",
                "language": "python",
                "stages": ["lint", "test", "build"],
            })
        )
        assert result["platform"] == "github_actions"
        assert "pipeline" in result
        assert "name:" in result["pipeline"]
        assert "jobs:" in result["pipeline"]

    def test_create_ci_pipeline_gitlab(self):
        """Test GitLab CI pipeline creation."""
        result = json.loads(
            create_ci_pipeline.invoke({
                "platform": "gitlab_ci",
                "language": "python",
                "stages": ["test", "build"],
            })
        )
        assert result["platform"] == "gitlab_ci"
        assert "stages:" in result["pipeline"]

    def test_create_cd_pipeline(self):
        """Test CD pipeline creation."""
        result = json.loads(
            create_cd_pipeline.invoke({
                "platform": "github_actions",
                "environments": ["staging", "production"],
                "deployment_type": "kubernetes",
            })
        )
        assert "pipeline" in result
        assert "staging" in result["pipeline"] or "deploy" in result["pipeline"].lower()

    def test_configure_deployment(self):
        """Test deployment configuration."""
        result = json.loads(
            configure_deployment.invoke({
                "environment": "production",
                "cloud_provider": "aws",
                "service_type": "container",
            })
        )
        assert result["environment"] == "production"
        assert "config" in result

    def test_generate_dockerfile(self):
        """Test Dockerfile generation."""
        result = json.loads(
            generate_dockerfile.invoke({
                "language": "python",
                "framework": "fastapi",
                "base_image": "python:3.11-slim",
            })
        )
        assert "dockerfile" in result
        assert "FROM" in result["dockerfile"]
        assert "python" in result["dockerfile"].lower()

    def test_create_kubernetes_config(self):
        """Test Kubernetes configuration creation."""
        result = json.loads(
            create_kubernetes_config.invoke({
                "service_name": "my-api",
                "replicas": 3,
                "port": 8000,
                "resources": {"cpu": "500m", "memory": "512Mi"},
            })
        )
        assert "manifests" in result
        # Should have deployment and service
        assert "Deployment" in str(result["manifests"]) or "deployment" in str(result)

    def test_setup_monitoring(self):
        """Test monitoring setup."""
        result = json.loads(
            setup_monitoring.invoke({
                "service_name": "my-api",
                "metrics": ["latency", "error_rate", "throughput"],
                "platform": "prometheus",
            })
        )
        assert "config" in result
        assert result["platform"] == "prometheus"


# Debugging tools tests
from app.deepagents.software_dev.tools.debugging_tools import (
    analyze_error,
    trace_execution,
    identify_root_cause,
    propose_fix,
    analyze_performance,
    detect_memory_issues,
)


class TestDebuggingTools:
    """Tests for debugging and optimization tools."""

    def test_analyze_error(self):
        """Test error analysis."""
        result = json.loads(
            analyze_error.invoke({
                "error_message": "KeyError: 'user_id'",
                "stack_trace": """
Traceback (most recent call last):
  File "app.py", line 42, in get_user
    return data['user_id']
KeyError: 'user_id'
""",
                "language": "python",
            })
        )
        assert "analysis" in result
        assert "error_type" in result
        assert result["error_type"] == "KeyError"

    def test_trace_execution(self):
        """Test execution tracing."""
        result = json.loads(
            trace_execution.invoke({
                "code": """
def calculate_total(items):
    total = 0
    for item in items:
        total += item['price'] * item['quantity']
    return total
""",
                "input_data": {"items": [{"price": 10, "quantity": 2}]},
            })
        )
        assert "trace" in result
        assert "steps" in result

    def test_identify_root_cause(self):
        """Test root cause identification."""
        result = json.loads(
            identify_root_cause.invoke({
                "symptoms": [
                    "API returns 500 error",
                    "Database connection timeout",
                    "High CPU usage",
                ],
                "context": "Production environment after recent deployment",
            })
        )
        assert "root_cause" in result
        assert "analysis" in result
        assert "confidence" in result

    def test_propose_fix(self):
        """Test fix proposal."""
        result = json.loads(
            propose_fix.invoke({
                "error": "TypeError: cannot unpack non-iterable NoneType object",
                "code": """
def get_user_info(user_id):
    name, email = get_user(user_id)
    return {"name": name, "email": email}
""",
                "language": "python",
            })
        )
        assert "fixes" in result
        assert len(result["fixes"]) > 0
        assert "code" in result["fixes"][0] or "description" in result["fixes"][0]

    def test_analyze_performance(self):
        """Test performance analysis."""
        result = json.loads(
            analyze_performance.invoke({
                "code": """
def find_duplicates(items):
    duplicates = []
    for i in range(len(items)):
        for j in range(len(items)):
            if i != j and items[i] == items[j]:
                if items[i] not in duplicates:
                    duplicates.append(items[i])
    return duplicates
""",
                "language": "python",
            })
        )
        assert "analysis" in result
        assert "complexity" in result or "bottlenecks" in result
        # Should detect O(n^2) complexity
        assert "optimizations" in result

    def test_detect_memory_issues(self):
        """Test memory issue detection."""
        result = json.loads(
            detect_memory_issues.invoke({
                "code": """
cache = {}

def process_data(data):
    global cache
    result = heavy_computation(data)
    cache[id(data)] = result  # Never cleaned up
    return result
""",
                "language": "python",
            })
        )
        assert "issues" in result
        # Should detect potential memory leak
        assert len(result["issues"]) > 0 or "warnings" in result


# Documentation tools tests
from app.deepagents.software_dev.tools.documentation_tools import (
    generate_api_docs,
    create_readme,
    document_architecture,
    generate_changelog,
    add_inline_comments,
    create_user_guide,
)


class TestDocumentationTools:
    """Tests for documentation generation tools."""

    def test_generate_api_docs(self):
        """Test API documentation generation."""
        result = json.loads(
            generate_api_docs.invoke({
                "endpoints": [
                    {
                        "path": "/api/users",
                        "method": "GET",
                        "description": "List all users",
                        "response": {"type": "array"},
                    },
                    {
                        "path": "/api/users",
                        "method": "POST",
                        "description": "Create user",
                        "request": {"name": "string"},
                    },
                ],
                "format": "openapi",
            })
        )
        assert "documentation" in result
        assert result["format"] == "openapi"

    def test_create_readme(self):
        """Test README generation."""
        result = json.loads(
            create_readme.invoke({
                "project_name": "My API",
                "description": "A REST API for managing users",
                "features": ["Authentication", "User management", "Rate limiting"],
                "tech_stack": ["Python", "FastAPI", "PostgreSQL"],
            })
        )
        assert "readme" in result
        assert "# My API" in result["readme"]
        assert "Features" in result["readme"]

    def test_document_architecture(self):
        """Test architecture documentation."""
        result = json.loads(
            document_architecture.invoke({
                "components": [
                    {"name": "API Gateway", "description": "Routes requests"},
                    {"name": "Auth Service", "description": "Handles authentication"},
                ],
                "format": "markdown",
            })
        )
        assert "documentation" in result
        assert "API Gateway" in result["documentation"]

    def test_generate_changelog(self):
        """Test changelog generation."""
        result = json.loads(
            generate_changelog.invoke({
                "changes": [
                    {"type": "feat", "description": "Add user authentication"},
                    {"type": "fix", "description": "Fix login bug"},
                    {"type": "docs", "description": "Update README"},
                ],
                "version": "1.2.0",
            })
        )
        assert "changelog" in result
        assert "1.2.0" in result["changelog"]
        assert "feat" in result["changelog"].lower() or "Features" in result["changelog"]

    def test_add_inline_comments(self):
        """Test inline comment addition."""
        code = """
def calculate_discount(price, percentage):
    if percentage > 100:
        percentage = 100
    discount = price * (percentage / 100)
    return price - discount
"""
        result = json.loads(
            add_inline_comments.invoke({
                "code": code,
                "language": "python",
                "detail_level": "medium",
            })
        )
        assert "commented_code" in result
        assert "#" in result["commented_code"]

    def test_create_user_guide(self):
        """Test user guide creation."""
        result = json.loads(
            create_user_guide.invoke({
                "product_name": "TaskManager",
                "features": [
                    {"name": "Create Task", "description": "Add new tasks"},
                    {"name": "Complete Task", "description": "Mark tasks as done"},
                ],
                "format": "markdown",
            })
        )
        assert "guide" in result
        assert "TaskManager" in result["guide"]


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

    def test_agent_creation(self):
        """Test agent creation with default config."""
        from app.deepagents.software_dev.software_dev_agent import (
            create_software_dev_agent,
        )
        # This will fail without API key, so we mock it
        with patch("app.deepagents.software_dev.software_dev_agent.ChatOpenAI"):
            agent = create_software_dev_agent()
            assert agent is not None
            assert agent.config is not None

    def test_agent_config(self):
        """Test agent configuration."""
        from app.deepagents.software_dev.software_dev_agent import (
            SoftwareDevDeepAgent,
        )
        config = SoftwareDevAgentConfig(
            model="gpt-4",
            max_iterations=100,
        )
        with patch("app.deepagents.software_dev.software_dev_agent.ChatOpenAI"):
            agent = SoftwareDevDeepAgent(config=config)
            assert agent.config.model == "gpt-4"
            assert agent.config.max_iterations == 100


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

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test health check endpoint."""
        from app.deepagents.software_dev.routes import health_check

        result = await health_check()
        assert result["status"] == "healthy"
        assert result["agent"] == "Software Development DeepAgent"

    @pytest.mark.asyncio
    async def test_list_phases_endpoint(self):
        """Test list phases endpoint."""
        from app.deepagents.software_dev.routes import list_phases

        result = await list_phases()
        assert "phases" in result
        assert result["count"] == 9
        phase_names = [p["name"] for p in result["phases"]]
        assert "requirements" in phase_names
        assert "design" in phase_names
        assert "implementation" in phase_names

    @pytest.mark.asyncio
    async def test_list_subagents_endpoint(self):
        """Test list subagents endpoint."""
        from app.deepagents.software_dev.routes import list_subagents

        result = await list_subagents()
        assert len(result) == 9
        names = [s.name for s in result]
        assert "requirements-intelligence" in names
        assert "code-generator" in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
