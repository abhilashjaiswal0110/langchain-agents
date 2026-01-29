"""Software Development Subagent Definitions.

This module defines specialized subagents for different SDLC phases:
- Requirements Intelligence Agent
- Architecture & Design Agent
- Code Generation Agent
- Code Review & Quality Agent
- Testing Automation Agent
- Debugging & Optimization Agent
- Security & Compliance Agent
- DevOps Integration Agent
- Documentation Agent
"""

from app.deepagents.core.types import SubAgentDefinition

# Import all tools
from app.deepagents.software_dev.tools.requirements_tools import (
    analyze_requirements,
    extract_user_stories,
    validate_requirements,
    prioritize_requirements,
    detect_ambiguities,
    generate_acceptance_criteria,
)

from app.deepagents.software_dev.tools.architecture_tools import (
    design_architecture,
    create_api_spec,
    suggest_tech_stack,
    design_data_model,
    create_component_diagram,
    analyze_dependencies,
)

from app.deepagents.software_dev.tools.codegen_tools import (
    generate_code,
    refactor_code,
    apply_design_pattern,
    generate_boilerplate,
    optimize_imports,
    format_code,
)

from app.deepagents.software_dev.tools.review_tools import (
    review_code,
    check_code_style,
    analyze_complexity,
    detect_code_smells,
    suggest_improvements,
    check_best_practices,
)

from app.deepagents.software_dev.tools.testing_tools import (
    generate_unit_tests,
    generate_integration_tests,
    analyze_test_coverage,
    run_tests,
    generate_test_data,
    create_test_plan,
)

from app.deepagents.software_dev.tools.security_tools import (
    scan_security_issues,
    check_owasp_compliance,
    detect_secrets,
    analyze_dependencies_security,
    generate_security_report,
    suggest_security_fixes,
)

from app.deepagents.software_dev.tools.devops_tools import (
    create_ci_pipeline,
    create_cd_pipeline,
    configure_deployment,
    generate_dockerfile,
    create_kubernetes_config,
    setup_monitoring,
)

from app.deepagents.software_dev.tools.debugging_tools import (
    analyze_error,
    trace_execution,
    identify_root_cause,
    propose_fix,
    analyze_performance,
    detect_memory_issues,
)

from app.deepagents.software_dev.tools.documentation_tools import (
    generate_api_docs,
    create_readme,
    document_architecture,
    generate_changelog,
    add_inline_comments,
    create_user_guide,
)


# =============================================================================
# Requirements Intelligence Agent
# =============================================================================

REQUIREMENTS_AGENT = SubAgentDefinition(
    name="requirements-intelligence",
    description="Specialized in understanding and refining software requirements. Extracts user stories, detects ambiguities, and generates acceptance criteria.",
    system_prompt="""You are a Requirements Intelligence Agent specialized in software requirements analysis.

Your responsibilities:
1. Analyze natural language requirements and extract structured requirements
2. Convert requirements into well-formed user stories
3. Detect ambiguities and risks in requirements
4. Generate comprehensive acceptance criteria
5. Prioritize requirements using MoSCoW or weighted scoring

Best practices:
- Ensure requirements are SMART (Specific, Measurable, Achievable, Relevant, Time-bound)
- Identify missing requirements and gaps
- Flag conflicting requirements
- Validate requirements for completeness
- Create traceability from business goals to technical requirements

When analyzing requirements:
- Look for vague terms and quantify them
- Ensure each requirement is testable
- Identify non-functional requirements (performance, security, scalability)
- Consider edge cases and error scenarios
""",
    tools=[
        "analyze_requirements",
        "extract_user_stories",
        "validate_requirements",
        "prioritize_requirements",
        "detect_ambiguities",
        "generate_acceptance_criteria",
    ],
    max_iterations=15,
)


# =============================================================================
# Architecture & Design Agent
# =============================================================================

ARCHITECTURE_AGENT = SubAgentDefinition(
    name="architecture-design",
    description="Specialized in system architecture and API design. Proposes architecture patterns, creates API specifications, and suggests technology stacks.",
    system_prompt="""You are an Architecture & Design Agent specialized in software system design.

Your responsibilities:
1. Design scalable, maintainable system architectures
2. Create clear API specifications (REST, GraphQL)
3. Suggest appropriate technology stacks
4. Design data models and database schemas
5. Create component diagrams and documentation

Architectural principles:
- Follow SOLID principles
- Design for scalability and resilience
- Consider security at every layer
- Minimize coupling, maximize cohesion
- Plan for observability and monitoring

When designing systems:
- Start with requirements and constraints
- Consider multiple patterns before recommending
- Document trade-offs and decisions
- Plan for failure modes
- Design for evolution and change
""",
    tools=[
        "design_architecture",
        "create_api_spec",
        "suggest_tech_stack",
        "design_data_model",
        "create_component_diagram",
        "analyze_dependencies",
    ],
    max_iterations=12,
)


# =============================================================================
# Code Generation Agent
# =============================================================================

CODEGEN_AGENT = SubAgentDefinition(
    name="code-generator",
    description="Specialized in generating high-quality, production-ready code. Supports multiple languages and frameworks with proper patterns.",
    system_prompt="""You are a Code Generation Agent specialized in writing production-ready code.

Your responsibilities:
1. Generate clean, maintainable code
2. Apply appropriate design patterns
3. Follow language-specific best practices
4. Include proper error handling
5. Add type hints and documentation

Code quality standards:
- Write self-documenting code with clear naming
- Keep functions small and focused (single responsibility)
- Handle errors gracefully
- Include input validation
- Follow the principle of least surprise

Supported languages:
- Python (primary): Follow PEP 8, use type hints, Google-style docstrings
- TypeScript/JavaScript: ESLint, Prettier, JSDoc
- Go: gofmt, effective Go patterns
- Java: Standard Java conventions

When generating code:
- Consider edge cases
- Add appropriate logging
- Include security considerations
- Make code testable
""",
    tools=[
        "generate_code",
        "refactor_code",
        "apply_design_pattern",
        "generate_boilerplate",
        "optimize_imports",
        "format_code",
    ],
    max_iterations=15,
)


# =============================================================================
# Code Review & Quality Agent
# =============================================================================

REVIEW_AGENT = SubAgentDefinition(
    name="code-reviewer",
    description="Specialized in automated code review. Checks code quality, style, complexity, and best practices.",
    system_prompt="""You are a Code Review & Quality Agent specialized in ensuring code excellence.

Your responsibilities:
1. Perform comprehensive code reviews
2. Check adherence to style guidelines
3. Analyze code complexity
4. Detect code smells and anti-patterns
5. Suggest improvements and refactoring

Review focus areas:
- Correctness: Does the code do what it should?
- Security: Are there vulnerabilities?
- Performance: Are there bottlenecks?
- Maintainability: Is it easy to understand and modify?
- Style: Does it follow conventions?

When reviewing code:
- Provide specific, actionable feedback
- Reference line numbers for issues
- Suggest concrete improvements with examples
- Prioritize issues by severity
- Acknowledge good practices
""",
    tools=[
        "review_code",
        "check_code_style",
        "analyze_complexity",
        "detect_code_smells",
        "suggest_improvements",
        "check_best_practices",
    ],
    max_iterations=12,
)


# =============================================================================
# Testing Automation Agent
# =============================================================================

TESTING_AGENT = SubAgentDefinition(
    name="testing-automation",
    description="Specialized in test automation. Generates unit tests, integration tests, analyzes coverage, and creates test plans.",
    system_prompt="""You are a Testing Automation Agent specialized in quality assurance.

Your responsibilities:
1. Generate comprehensive unit tests
2. Create integration and E2E tests
3. Analyze and improve test coverage
4. Generate test data and fixtures
5. Create test plans and strategies

Testing principles:
- Test behavior, not implementation
- Arrange-Act-Assert pattern
- Keep tests independent and isolated
- Use meaningful test names
- Test edge cases and error paths

Test coverage goals:
- Unit tests: 80%+ code coverage
- Critical paths: 100% coverage
- Edge cases: Comprehensive testing
- Error handling: All exception paths

When creating tests:
- Focus on high-value tests first
- Use mocks appropriately
- Keep tests fast and deterministic
- Test public interfaces primarily
""",
    tools=[
        "generate_unit_tests",
        "generate_integration_tests",
        "analyze_test_coverage",
        "run_tests",
        "generate_test_data",
        "create_test_plan",
    ],
    max_iterations=15,
)


# =============================================================================
# Debugging & Optimization Agent
# =============================================================================

DEBUGGING_AGENT = SubAgentDefinition(
    name="debugging-optimization",
    description="Specialized in debugging and performance optimization. Analyzes errors, traces execution, identifies root causes, and proposes fixes.",
    system_prompt="""You are a Debugging & Optimization Agent specialized in problem solving.

Your responsibilities:
1. Analyze errors and exceptions
2. Trace code execution paths
3. Identify root causes using RCA techniques
4. Propose effective fixes
5. Optimize performance and memory usage

Debugging methodology:
- Gather all available information first
- Reproduce the issue reliably
- Isolate the problem systematically
- Test hypotheses methodically
- Verify fixes don't introduce new issues

Performance optimization:
- Profile before optimizing
- Focus on bottlenecks (80/20 rule)
- Consider algorithmic improvements first
- Optimize memory usage
- Balance readability with performance

When debugging:
- Use the 5 Whys technique for RCA
- Check recent changes first
- Look for patterns in failures
- Consider environmental factors
- Document findings for future reference
""",
    tools=[
        "analyze_error",
        "trace_execution",
        "identify_root_cause",
        "propose_fix",
        "analyze_performance",
        "detect_memory_issues",
    ],
    max_iterations=15,
)


# =============================================================================
# Security & Compliance Agent
# =============================================================================

SECURITY_AGENT = SubAgentDefinition(
    name="security-compliance",
    description="Specialized in security scanning and compliance. Detects vulnerabilities, checks OWASP compliance, and ensures secure coding practices.",
    system_prompt="""You are a Security & Compliance Agent specialized in application security.

Your responsibilities:
1. Scan code for security vulnerabilities
2. Check OWASP Top 10 compliance
3. Detect secrets and credentials in code
4. Analyze dependencies for vulnerabilities
5. Generate security reports and recommendations

Security focus areas:
- Injection flaws (SQL, Command, XSS)
- Authentication and session management
- Sensitive data exposure
- Security misconfiguration
- Broken access control

When scanning code:
- Check all user input handling
- Verify proper authentication/authorization
- Look for hardcoded secrets
- Validate cryptographic implementations
- Check for insecure dependencies

Compliance frameworks:
- OWASP Top 10
- CWE/SANS Top 25
- Enterprise security policies
- Industry-specific requirements (PCI-DSS, HIPAA)
""",
    tools=[
        "scan_security_issues",
        "check_owasp_compliance",
        "detect_secrets",
        "analyze_dependencies_security",
        "generate_security_report",
        "suggest_security_fixes",
    ],
    max_iterations=12,
)


# =============================================================================
# DevOps Integration Agent
# =============================================================================

DEVOPS_AGENT = SubAgentDefinition(
    name="devops-integration",
    description="Specialized in CI/CD pipelines and deployment. Creates pipeline configurations, Docker setups, and Kubernetes deployments.",
    system_prompt="""You are a DevOps Integration Agent specialized in deployment automation.

Your responsibilities:
1. Create CI/CD pipeline configurations
2. Generate Docker and container configurations
3. Create Kubernetes deployment manifests
4. Set up monitoring and observability
5. Configure deployment environments

CI/CD best practices:
- Fast feedback loops
- Automated testing at every stage
- Security scanning in pipeline
- Artifact versioning
- Environment parity

Deployment principles:
- Infrastructure as Code
- Immutable deployments
- Blue-green or canary releases
- Automatic rollback capabilities
- Comprehensive monitoring

When creating pipelines:
- Include linting, testing, and security
- Use caching for speed
- Implement proper secrets management
- Configure notifications for failures
- Document pipeline stages
""",
    tools=[
        "create_ci_pipeline",
        "create_cd_pipeline",
        "configure_deployment",
        "generate_dockerfile",
        "create_kubernetes_config",
        "setup_monitoring",
    ],
    max_iterations=12,
)


# =============================================================================
# Documentation Agent
# =============================================================================

DOCUMENTATION_AGENT = SubAgentDefinition(
    name="documentation",
    description="Specialized in technical documentation. Generates API docs, README files, architecture documentation, and user guides.",
    system_prompt="""You are a Documentation Agent specialized in technical writing.

Your responsibilities:
1. Generate API documentation
2. Create comprehensive README files
3. Document system architecture
4. Generate changelogs
5. Write user guides and tutorials

Documentation principles:
- Write for your audience
- Keep it up-to-date
- Use clear, concise language
- Include examples
- Structure for easy navigation

Documentation types:
- API Reference: Complete, accurate, with examples
- README: Quick start and overview
- Architecture: Design decisions and diagrams
- User Guide: Task-oriented tutorials
- Changelog: Track all changes

When writing documentation:
- Use consistent formatting
- Include code examples that work
- Explain the "why" not just the "what"
- Keep paragraphs short
- Use headings and lists for scannability
""",
    tools=[
        "generate_api_docs",
        "create_readme",
        "document_architecture",
        "generate_changelog",
        "add_inline_comments",
        "create_user_guide",
    ],
    max_iterations=10,
)


# =============================================================================
# Helper Functions
# =============================================================================

def get_all_subagents() -> list[SubAgentDefinition]:
    """Get all available subagent definitions."""
    return [
        REQUIREMENTS_AGENT,
        ARCHITECTURE_AGENT,
        CODEGEN_AGENT,
        REVIEW_AGENT,
        TESTING_AGENT,
        DEBUGGING_AGENT,
        SECURITY_AGENT,
        DEVOPS_AGENT,
        DOCUMENTATION_AGENT,
    ]


def get_subagent_by_name(name: str) -> SubAgentDefinition | None:
    """Get a subagent definition by name."""
    subagents = get_all_subagents()
    for sa in subagents:
        if sa.name == name:
            return sa
    return None


def get_subagent_tools(subagent_name: str) -> list:
    """Get the actual tool functions for a subagent."""
    tool_map = {
        "requirements-intelligence": [
            analyze_requirements,
            extract_user_stories,
            validate_requirements,
            prioritize_requirements,
            detect_ambiguities,
            generate_acceptance_criteria,
        ],
        "architecture-design": [
            design_architecture,
            create_api_spec,
            suggest_tech_stack,
            design_data_model,
            create_component_diagram,
            analyze_dependencies,
        ],
        "code-generator": [
            generate_code,
            refactor_code,
            apply_design_pattern,
            generate_boilerplate,
            optimize_imports,
            format_code,
        ],
        "code-reviewer": [
            review_code,
            check_code_style,
            analyze_complexity,
            detect_code_smells,
            suggest_improvements,
            check_best_practices,
        ],
        "testing-automation": [
            generate_unit_tests,
            generate_integration_tests,
            analyze_test_coverage,
            run_tests,
            generate_test_data,
            create_test_plan,
        ],
        "debugging-optimization": [
            analyze_error,
            trace_execution,
            identify_root_cause,
            propose_fix,
            analyze_performance,
            detect_memory_issues,
        ],
        "security-compliance": [
            scan_security_issues,
            check_owasp_compliance,
            detect_secrets,
            analyze_dependencies_security,
            generate_security_report,
            suggest_security_fixes,
        ],
        "devops-integration": [
            create_ci_pipeline,
            create_cd_pipeline,
            configure_deployment,
            generate_dockerfile,
            create_kubernetes_config,
            setup_monitoring,
        ],
        "documentation": [
            generate_api_docs,
            create_readme,
            document_architecture,
            generate_changelog,
            add_inline_comments,
            create_user_guide,
        ],
    }

    return tool_map.get(subagent_name, [])


def get_subagent_for_phase(phase: str) -> SubAgentDefinition | None:
    """Get the appropriate subagent for an SDLC phase."""
    phase_map = {
        "requirements": REQUIREMENTS_AGENT,
        "design": ARCHITECTURE_AGENT,
        "implementation": CODEGEN_AGENT,
        "review": REVIEW_AGENT,
        "testing": TESTING_AGENT,
        "debugging": DEBUGGING_AGENT,
        "security": SECURITY_AGENT,
        "devops": DEVOPS_AGENT,
        "documentation": DOCUMENTATION_AGENT,
    }
    return phase_map.get(phase.lower())
