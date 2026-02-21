# Global development guidelines for the LangChain monorepo

This document provides context to understand the LangChain Python project and assist with development.

## Project architecture and context

### Monorepo structure

This is a Python monorepo with multiple independently versioned packages that use `uv`.

```txt
langchain/
├── deployment/               # Enterprise Agents Platform (production-ready)
│   ├── app/                 # Main application code
│   │   ├── agents/          # Agent implementations
│   │   ├── deepagents/      # Deep Agents framework
│   │   │   ├── core/        # Core components (middleware, types)
│   │   │   ├── config/      # Agent configurations
│   │   │   ├── software_dev/ # Software Development Deep Agent
│   │   │   │   ├── tools/    # 54+ SDLC tools including bash execution
│   │   │   │   │   ├── bash_execution_tools.py  # Secure command execution
│   │   │   │   │   └── azure_integration.py     # Azure deployment tools
│   │   │   ├── recruitment_agent.py # Recruitment Deep Agent
│   │   │   └── it_operations_agent.py # IT Ops Deep Agent
│   │   ├── auth/            # Authentication
│   │   ├── chains/          # LangChain chains
│   │   ├── governance/      # Governance framework
│   │   ├── integrations/    # External integrations
│   │   ├── memory/          # Session and conversation memory
│   │   ├── static/          # Web UI files
│   │   └── server.py        # FastAPI server
│   ├── data/                # Data storage
│   ├── docs/                # Documentation
│   ├── infrastructure/      # Azure Bicep IaC
│   ├── tests/               # Test suite
│   ├── pyproject.toml       # Dependencies
│   ├── langgraph.json       # LangGraph Studio config
│   └── README.md            # Deployment guide
├── libs/
│   ├── core/                # `langchain-core` primitives and base abstractions
│   ├── langchain/           # `langchain-classic` (legacy, no new features)
│   ├── langchain_v1/        # Actively maintained `langchain` package
│   ├── partners/            # Third-party integrations
│   │   ├── openai/          # OpenAI models and embeddings
│   │   ├── anthropic/       # Anthropic (Claude) integration
│   │   ├── ollama/          # Local model support
│   │   └── ...              # Other integrations maintained by the LangChain team
│   ├── text-splitters/      # Document chunking utilities
│   ├── standard-tests/      # Shared test suite for integrations
│   └── model-profiles/      # Model configuration profiles
├── .github/                 # CI/CD workflows and templates
├── .vscode/                 # VSCode IDE standard settings and recommended extensions
└── README.md                # Information about LangChain
```

- **Deployment layer** (`deployment/`): Production-ready enterprise agents platform with 10 specialized agents, REST APIs, Web UI, and comprehensive documentation. See [deployment/README.md](deployment/README.md) for details.
- **Core layer** (`langchain-core`): Base abstractions, interfaces, and protocols. Users should not need to know about this layer directly.
- **Implementation layer** (`langchain`): Concrete implementations and high-level public utilities
- **Integration layer** (`partners/`): Third-party service integrations. Note that this monorepo is not exhaustive of all LangChain integrations; some are maintained in separate repos, such as `langchain-ai/langchain-google` and `langchain-ai/langchain-aws`. Usually these repos are cloned at the same level as this monorepo, so if needed, you can refer to their code directly by navigating to `../langchain-google/` from this monorepo.
- **Testing layer** (`standard-tests/`): Standardized integration tests for partner integrations

## Enterprise Agents Platform

The `deployment/` folder contains a production-ready enterprise agents platform built with LangChain and LangGraph. This is a separate application layer on top of the core LangChain libraries.

### Key Features

- **12 Production Agents**: Research, Content Generation, Data Analysis, Document Processing, Multilingual RAG, HITL IT Support, ServiceNow ITSM, Code Assistant, Document Intelligence, Employee Experience, Recruitment, Software Development
- **Deep Agents Framework**: Three advanced agents with planning, subagent delegation, and context management (IT Operations, Sales Intelligence, Recruitment)
- **IT Support Agents**: Four conversational agents with session memory (IT Helpdesk, ServiceNow, Document Intelligence, Employee Experience)
- **REST API**: FastAPI server with LangServe integration
- **Web UI**: Interactive chat interfaces for agent testing
- **LangGraph Studio**: Visual development and debugging interface
- **Security**: API key authentication, CORS, secrets management
- **Observability**: LangSmith tracing and evaluation framework
- **Docker Deployment**: Production-ready containerization

### Verified Agent Inventory (2026-02-20)

All agents below confirmed loaded and live-tested against Azure OpenAI (`o4-mini`):

| Category | Agent | API Path | Status |
|----------|-------|----------|--------|
| Enterprise | Research | `/api/enterprise/research/invoke` | ✅ Live |
| Enterprise | Content | `/api/enterprise/content/invoke` | ✅ Loaded |
| Enterprise | Data Analyst | `/api/enterprise/data-analyst/invoke` | ✅ Loaded |
| Enterprise | Document | `/api/enterprise/documents/invoke` | ✅ Loaded |
| Enterprise | Multilingual RAG | `/api/enterprise/rag/invoke` | ✅ Loaded |
| Enterprise | HITL Support | `/api/enterprise/support/invoke` | ✅ Loaded |
| Enterprise | Code Assistant | `/api/enterprise/code/invoke` | ✅ Loaded |
| Enterprise | Document Intelligence | `/api/enterprise/document-intelligence/invoke` | ✅ Loaded |
| Deep Agent | IT Operations | `/api/deepagent/start` + `/api/deepagent/chat` | ✅ Live |
| Deep Agent | Sales Intelligence | `/api/sales-agent/start` + `/api/sales-agent/chat` | ✅ Loaded |
| Deep Agent | Recruitment | `/api/recruitment-agent/start` + `/api/recruitment-agent/chat` | ✅ Loaded |
| IT Support | IT Helpdesk | `/api/conversation/start` (`it_helpdesk`) | ✅ Live |
| IT Support | ServiceNow | `/api/conversation/start` (`servicenow`) | ✅ Loaded |
| IT Support | Document Intelligence | `/api/conversation/start` (`document_intelligence`) | ✅ Loaded |
| IT Support | Employee Experience | `/api/conversation/start` (`employee_experience`) | ✅ Loaded |
| Software Dev | Software Dev Agent | `/api/software-dev-agent/start` | ✅ Loaded |

### Development Context

When working on enterprise agents in the `deployment/` folder, follow these additional guidelines:

**File Organization:**
- Agent implementations: `deployment/app/agents/` or `deployment/app/deepagents/`
- API routes: `deployment/app/server.py` or agent-specific `routes.py` files
- Tests: `deployment/tests/` (mirror source structure)
- Documentation: `deployment/docs/` and `deployment/KNOWLEDGE.md`

**Agent Development Patterns:**
- Inherit from `BaseAgent` for standard agents
- Use LangGraph StateGraph for workflow orchestration
- Implement streaming endpoints for real-time feedback
- Add comprehensive tests with pytest
- Update `deployment/KNOWLEDGE.md` with architecture details

**Testing Requirements:**
- Unit tests: No external dependencies
- Integration tests: Test with real LLM providers
- Evaluation tests: Use LangSmith datasets for agent performance
- All tests in `deployment/tests/` with clear naming

**Commit Scope for Deployment:**
Use `deployment` or specific agent scopes for enterprise agents work:
```txt
feat(deployment): add new enterprise agent
feat(software-dev-agent): add code generation tools
fix(recruitment-agent): resolve SharePoint authentication
docs(deployment): update architecture documentation
```

### Development tools & commands

- `uv` – Fast Python package installer and resolver (replaces pip/poetry)
- `make` – Task runner for common development commands. Feel free to look at the `Makefile` for available commands and usage patterns.
- `ruff` – Fast Python linter and formatter
- `mypy` – Static type checking
- `pytest` – Testing framework

This monorepo uses `uv` for dependency management. Local development uses editable installs: `[tool.uv.sources]`

Each package in `libs/` has its own `pyproject.toml` and `uv.lock`.

Before running your tests, setup all packages by running:

```bash
# For all groups
uv sync --all-groups

# or, to install a specific group only:
uv sync --group test
```

```bash
# Run unit tests (no network)
make test

# Run specific test file
uv run --group test pytest tests/unit_tests/test_specific.py
```

```bash
# Lint code
make lint

# Format code
make format

# Type checking
uv run --group lint mypy .
```

#### Key config files

- pyproject.toml: Main workspace configuration with dependency groups
- uv.lock: Locked dependencies for reproducible builds
- Makefile: Development tasks

#### Commit standards

Suggest PR titles that follow Conventional Commits format. Refer to .github/workflows/pr_lint for allowed types and scopes. Note that all commit/PR titles should be in lowercase with the exception of proper nouns/named entities. All PR titles should include a scope with no exceptions. For example:

```txt
feat(langchain): add new chat completion feature
fix(core): resolve type hinting issue in vector store
chore(anthropic): update infrastructure dependencies
```

Note how `feat(langchain)` includes a scope even though it is the main package and name of the repo.

#### Pull request guidelines

- Always add a disclaimer to the PR description mentioning how AI agents are involved with the contribution.
- Describe the "why" of the changes, why the proposed solution is the right one. Limit prose.
- Highlight areas of the proposed changes that require careful review.

## Core development principles

### Maintain stable public interfaces

CRITICAL: Always attempt to preserve function signatures, argument positions, and names for exported/public methods. Do not make breaking changes.
You should warn the developer for any function signature changes, regardless of whether they look breaking or not.

**Before making ANY changes to public APIs:**

- Check if the function/class is exported in `__init__.py`
- Look for existing usage patterns in tests and examples
- Use keyword-only arguments for new parameters: `*, new_param: str = "default"`
- Mark experimental features clearly with docstring warnings (using MkDocs Material admonitions, like `!!! warning`)

Ask: "Would this change break someone's code if they used it last week?"

### Code quality standards

All Python code MUST include type hints and return types.

```python title="Example"
def filter_unknown_users(users: list[str], known_users: set[str]) -> list[str]:
    """Single line description of the function.

    Any additional context about the function can go here.

    Args:
        users: List of user identifiers to filter.
        known_users: Set of known/valid user identifiers.

    Returns:
        List of users that are not in the `known_users` set.
    """
```

- Use descriptive, self-explanatory variable names.
- Follow existing patterns in the codebase you're modifying
- Attempt to break up complex functions (>20 lines) into smaller, focused functions where it makes sense

### Testing requirements

Every new feature or bugfix MUST be covered by unit tests.

- Unit tests: `tests/unit_tests/` (no network calls allowed)
- Integration tests: `tests/integration_tests/` (network calls permitted)
- We use `pytest` as the testing framework; if in doubt, check other existing tests for examples.
- The testing file structure should mirror the source code structure.

**Checklist:**

- [ ] Tests fail when your new logic is broken
- [ ] Happy path is covered
- [ ] Edge cases and error conditions are tested
- [ ] Use fixtures/mocks for external dependencies
- [ ] Tests are deterministic (no flaky tests)
- [ ] Does the test suite fail if your new logic is broken?

### Security and risk assessment

- No `eval()`, `exec()`, or `pickle` on user-controlled input
- Proper exception handling (no bare `except:`) and use a `msg` variable for error messages
- Remove unreachable/commented code before committing
- Race conditions or resource leaks (file handles, sockets, threads).
- Ensure proper resource cleanup (file handles, connections)

### Documentation standards

Use Google-style docstrings with Args section for all public functions.

```python title="Example"
def send_email(to: str, msg: str, *, priority: str = "normal") -> bool:
    """Send an email to a recipient with specified priority.

    Any additional context about the function can go here.

    Args:
        to: The email address of the recipient.
        msg: The message body to send.
        priority: Email priority level.

    Returns:
        `True` if email was sent successfully, `False` otherwise.

    Raises:
        InvalidEmailError: If the email address format is invalid.
        SMTPConnectionError: If unable to connect to email server.
    """
```

- Types go in function signatures, NOT in docstrings
  - If a default is present, DO NOT repeat it in the docstring unless there is post-processing or it is set conditionally.
- Focus on "why" rather than "what" in descriptions
- Document all parameters, return values, and exceptions
- Keep descriptions concise but clear
- Ensure American English spelling (e.g., "behavior", not "behaviour")
- Do NOT use Sphinx-style double backtick formatting (` ``code`` `). Use single backticks (`` `code` ``) for inline code references in docstrings and comments.

## Software Development Deep Agent - Bash Execution & Azure Integration

### Overview

The Software Development Deep Agent now includes secure bash execution capabilities and Azure cloud integration for automated SDLC workflows.

### Bash Execution Tools

**Location**: `deployment/app/deepagents/software_dev/tools/bash_execution_tools.py`

**Key Features**:
- **Multi-platform support**: Bash (Linux/macOS), PowerShell (Windows), CMD fallback
- **Security validation**: Blocks dangerous commands (rm -rf /, fork bombs, dd to devices)
- **Warning system**: Flags risky operations (sudo, recursive deletes, curl | bash)
- **Cross-platform detection**: Automatic shell selection based on OS
- **Timeout protection**: Configurable execution timeout (default 30s)
- **Command history**: Tracks executed commands for auditing

**Available Tools** (4 total):
| Tool | Purpose |
|------|---------|
| `execute_bash_command` | Execute shell commands with security validation |
| `execute_python_code` | Run Python code snippets in isolated environment |
| `execute_tests_real` | Run test suites (pytest, npm test, cargo test) |
| `install_dependencies` | Install package dependencies (pip, npm, cargo) |

**Security Patterns**:
```python
# Blocked dangerous commands
rm -rf /                    # Root directory deletion
:(){ :|:& };:              # Fork bomb
dd if=... of=/dev/...      # Writing to devices
mkfs.ext4 /dev/...         # Filesystem formatting

# Warned risky commands
rm -rf /tmp/mydir          # Recursive deletion
sudo apt-get install       # Elevated privileges
curl ... | bash            # Piping web content to shell
```

**Usage Example**:
```python
from app.deepagents.software_dev.tools.bash_execution_tools import (
    execute_bash_command,
    execute_python_code,
    execute_tests_real,
)

# Execute a bash command
result = execute_bash_command.invoke({
    "command": "pytest tests/",
    "timeout": 60,
    "working_directory": "/path/to/project"
})

# Run Python code
result = execute_python_code.invoke({
    "code": "print('Hello, World!')",
    "timeout": 10
})

# Run tests
result = execute_tests_real.invoke({
    "test_framework": "pytest",
    "test_path": "tests/unit/",
    "additional_args": "-v --cov"
})
```

### Azure Integration

**Location**: `deployment/app/deepagents/software_dev/tools/azure_integration.py`

**Purpose**: Enable Azure cloud-based execution for bash commands and deployments.

**Supported Azure Services**:
| Service | Purpose | Configuration |
|---------|---------|---------------|
| **Azure Container Instances (ACI)** | Ephemeral command execution | `ACI_CONTAINER_GROUP_NAME`, `ACI_CONTAINER_IMAGE` |
| **Azure Functions** | Serverless command execution | `AZURE_FUNCTIONS_APP_NAME`, `AZURE_FUNCTIONS_RUNTIME` |
| **Azure Kubernetes Service (AKS)** | Production-grade container orchestration | `AKS_CLUSTER_NAME`, `AKS_DEPLOYMENT_NAME` |
| **Azure App Service** | Web app deployment and execution | `AZURE_APP_SERVICE_NAME`, `AZURE_APP_SERVICE_PLAN` |
| **Azure Key Vault** | Secrets management for credentials | `AZURE_KEY_VAULT_NAME`, `AZURE_KEY_VAULT_URI` |

**Configuration**:
1. Copy `.azure.config.example` to `.azure.config`
2. Fill in your Azure subscription details:
```bash
# Azure subscription
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=rg-langchain-agents
AZURE_LOCATION=eastus

# Container Instances
ACI_CONTAINER_IMAGE=myregistry.azurecr.io/bash-executor:latest
ACI_CPU_CORES=1.0
ACI_MEMORY_GB=1.5

# Azure Functions
AZURE_FUNCTIONS_APP_NAME=func-langchain-bash-executor
AZURE_FUNCTIONS_RUNTIME=python
```

**Security Considerations**:
- `.azure.config` is excluded from version control via `.gitignore`
- Use Azure Key Vault for sensitive credentials
- Configure managed identities for passwordless authentication
- Implement RBAC policies for least-privilege access
- Enable Azure Monitor for audit logging

**Integration with Code Generation**:
The Code Generator subagent now has access to bash execution tools for:
- Running code formatters (black, prettier, gofmt)
- Executing linters (ruff, eslint, clippy)
- Verifying generated code through execution
- Installing dependencies automatically

**Testing**:
Comprehensive test suite available at `deployment/tests/test_bash_execution_tools.py`:
- Security validation tests (dangerous/risky command detection)
- Cross-platform shell detection
- Command execution with various scenarios
- Error handling and timeout protection
- Python code execution
- Test framework integration

**Best Practices**:
1. Always validate commands before execution
2. Use appropriate timeouts for long-running operations
3. Specify working directory for context-aware execution
4. Review security warnings before proceeding with risky commands
5. Use Azure Key Vault for production credentials
6. Monitor execution logs via LangSmith tracing

## Additional resources

- **LangChain Documentation:** https://docs.langchain.com/oss/python/langchain/overview and source at https://github.com/langchain-ai/docs or `../docs/`. Prefer the local install and use file search tools for best results. If needed, use the docs MCP server as defined in `.mcp.json` for programmatic access.
- **Contributing Guide:** [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview)
- **Enterprise Agents Platform:**
  - [Deployment Guide](deployment/README.md) – Quick start and feature overview
  - [Knowledge Base](deployment/KNOWLEDGE.md) – Comprehensive technical documentation
  - [Architecture Blueprint](deployment/docs/Project_Architecture_Blueprint.md) – System design and patterns
  - [API Reference](deployment/docs/api/README.md) – Complete endpoint documentation
  - [LangGraph Setup](deployment/LANGGRAPH_SETUP.md) – Visual development with LangGraph Studio
