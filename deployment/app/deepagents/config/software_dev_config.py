"""Software Development DeepAgent Configuration.

This module defines configuration settings for the Software Development DeepAgent,
including SDLC phases, quality gates, and agent-specific settings.
"""

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class SDLCPhase(str, Enum):
    """Software Development Lifecycle phases."""

    REQUIREMENTS = "requirements"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    SECURITY = "security"
    DEVOPS = "devops"
    DEBUGGING = "debugging"
    DOCUMENTATION = "documentation"
    REVIEW = "review"


class CodeLanguage(str, Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    KOTLIN = "kotlin"
    SWIFT = "swift"


class TestType(str, Enum):
    """Types of tests supported."""

    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"
    SECURITY = "security"
    API = "api"


class SecuritySeverity(str, Enum):
    """Security issue severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RequirementType(str, Enum):
    """Types of software requirements."""

    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    TECHNICAL = "technical"
    BUSINESS = "business"


class RequirementPriority(str, Enum):
    """Requirement priority levels."""

    MUST_HAVE = "must_have"
    SHOULD_HAVE = "should_have"
    COULD_HAVE = "could_have"
    WONT_HAVE = "wont_have"


class ArchitecturePattern(str, Enum):
    """Common architecture patterns."""

    MICROSERVICES = "microservices"
    MONOLITHIC = "monolithic"
    SERVERLESS = "serverless"
    EVENT_DRIVEN = "event_driven"
    LAYERED = "layered"
    HEXAGONAL = "hexagonal"
    CQRS = "cqrs"
    MVC = "mvc"


class CodeQualityMetric(BaseModel):
    """Code quality metric definition."""

    name: str
    description: str
    target_value: float
    weight: float = 1.0


class QualityGateConfig(BaseModel):
    """Quality gate configuration for code review."""

    min_test_coverage: float = Field(default=80.0, description="Minimum test coverage %")
    max_complexity: int = Field(default=10, description="Max cyclomatic complexity")
    max_code_smells: int = Field(default=5, description="Max code smells allowed")
    require_security_scan: bool = Field(default=True)
    require_code_review: bool = Field(default=True)
    require_documentation: bool = Field(default=True)
    block_on_critical: bool = Field(default=True, description="Block on critical issues")


class TestGenerationConfig(BaseModel):
    """Configuration for test generation."""

    min_unit_tests: int = Field(default=3, description="Min unit tests per function")
    include_edge_cases: bool = Field(default=True)
    include_error_cases: bool = Field(default=True)
    generate_mocks: bool = Field(default=True)
    target_coverage: float = Field(default=80.0)


class SecurityScanConfig(BaseModel):
    """Configuration for security scanning."""

    check_owasp_top10: bool = Field(default=True)
    check_dependencies: bool = Field(default=True)
    check_secrets: bool = Field(default=True)
    check_sql_injection: bool = Field(default=True)
    check_xss: bool = Field(default=True)
    check_input_validation: bool = Field(default=True)
    severity_threshold: SecuritySeverity = Field(default=SecuritySeverity.MEDIUM)


class CICDConfig(BaseModel):
    """CI/CD pipeline configuration."""

    platform: str = Field(default="github-actions", description="CI/CD platform")
    build_stages: list[str] = Field(
        default_factory=lambda: ["lint", "test", "build", "deploy"]
    )
    enable_auto_deploy: bool = Field(default=False)
    require_approval_prod: bool = Field(default=True)
    enable_rollback: bool = Field(default=True)


class DocumentationType(str, Enum):
    """Types of documentation to generate."""

    API = "api"
    README = "readme"
    ARCHITECTURE = "architecture"
    USER_GUIDE = "user_guide"
    CHANGELOG = "changelog"
    INLINE_COMMENTS = "inline_comments"


class SoftwareDevAgentConfig(BaseModel):
    """Main configuration for Software Development DeepAgent."""

    # General settings
    name: str = Field(default="software_dev_agent", description="Agent name")
    model: str = Field(default="gpt-4o-mini", description="LLM model")
    temperature: float = Field(default=0.1, description="LLM temperature")
    max_iterations: int = Field(default=50, description="Max agent iterations")
    recursion_limit: int = Field(default=25, description="Max recursion depth")

    # Language support
    supported_languages: list[CodeLanguage] = Field(
        default_factory=lambda: [
            CodeLanguage.PYTHON,
            CodeLanguage.JAVASCRIPT,
            CodeLanguage.TYPESCRIPT,
            CodeLanguage.JAVA,
            CodeLanguage.GO,
        ]
    )
    primary_language: CodeLanguage = Field(default=CodeLanguage.PYTHON)

    # Quality gates
    quality_gates: QualityGateConfig = Field(default_factory=QualityGateConfig)

    # Test configuration
    test_config: TestGenerationConfig = Field(default_factory=TestGenerationConfig)

    # Security configuration
    security_config: SecurityScanConfig = Field(default_factory=SecurityScanConfig)

    # CI/CD configuration
    cicd_config: CICDConfig = Field(default_factory=CICDConfig)

    # Context limits
    max_file_size: int = Field(default=100000, description="Max file size in chars")
    max_context_files: int = Field(default=50, description="Max files in context")

    # Subagent settings
    max_concurrent_subagents: int = Field(default=3)
    subagent_timeout: int = Field(default=300, description="Subagent timeout seconds")

    # Memory settings
    enable_long_term_memory: bool = Field(default=True)
    memory_namespace: str = Field(default="software_dev")


# Default quality metrics
DEFAULT_QUALITY_METRICS: list[CodeQualityMetric] = [
    CodeQualityMetric(
        name="test_coverage",
        description="Percentage of code covered by tests",
        target_value=80.0,
        weight=1.5,
    ),
    CodeQualityMetric(
        name="cyclomatic_complexity",
        description="Average cyclomatic complexity",
        target_value=5.0,
        weight=1.0,
    ),
    CodeQualityMetric(
        name="code_duplication",
        description="Percentage of duplicated code",
        target_value=3.0,
        weight=1.0,
    ),
    CodeQualityMetric(
        name="documentation_coverage",
        description="Percentage of documented public APIs",
        target_value=90.0,
        weight=0.8,
    ),
    CodeQualityMetric(
        name="type_coverage",
        description="Percentage of typed code",
        target_value=95.0,
        weight=0.7,
    ),
]


# OWASP Top 10 Categories for security scanning
OWASP_TOP_10: dict[str, str] = {
    "A01": "Broken Access Control",
    "A02": "Cryptographic Failures",
    "A03": "Injection",
    "A04": "Insecure Design",
    "A05": "Security Misconfiguration",
    "A06": "Vulnerable Components",
    "A07": "Authentication Failures",
    "A08": "Software and Data Integrity Failures",
    "A09": "Security Logging and Monitoring Failures",
    "A10": "Server-Side Request Forgery (SSRF)",
}


def get_default_config() -> SoftwareDevAgentConfig:
    """Get default software development agent configuration."""
    return SoftwareDevAgentConfig()


def get_quality_metrics() -> list[CodeQualityMetric]:
    """Get default quality metrics."""
    return DEFAULT_QUALITY_METRICS.copy()


def get_owasp_categories() -> dict[str, str]:
    """Get OWASP Top 10 categories."""
    return OWASP_TOP_10.copy()
