<div align="center">
  <a href="https://www.langchain.com/">
    <picture>
      <source media="(prefers-color-scheme: light)" srcset=".github/images/logo-dark.svg">
      <source media="(prefers-color-scheme: dark)" srcset=".github/images/logo-light.svg">
      <img alt="LangChain Logo" src=".github/images/logo-dark.svg" width="80%">
    </picture>
  </a>
</div>

<div align="center">
  <h3>The platform for reliable agents.</h3>
</div>

<div align="center">
  <a href="https://opensource.org/licenses/MIT" target="_blank"><img src="https://img.shields.io/pypi/l/langchain" alt="PyPI - License"></a>
  <a href="https://pypistats.org/packages/langchain" target="_blank"><img src="https://img.shields.io/pepy/dt/langchain" alt="PyPI - Downloads"></a>
  <a href="https://pypi.org/project/langchain/#history" target="_blank"><img src="https://img.shields.io/pypi/v/langchain?label=%20" alt="Version"></a>
  <a href="https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/langchain-ai/langchain" target="_blank"><img src="https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode" alt="Open in Dev Containers"></a>
  <a href="https://codespaces.new/langchain-ai/langchain" target="_blank"><img src="https://github.com/codespaces/badge.svg" alt="Open in Github Codespace" title="Open in Github Codespace" width="150" height="20"></a>
  <a href="https://codspeed.io/langchain-ai/langchain" target="_blank"><img src="https://img.shields.io/endpoint?url=https://codspeed.io/badge.json" alt="CodSpeed Badge"></a>
  <a href="https://x.com/langchain" target="_blank"><img src="https://img.shields.io/twitter/url/https/twitter.com/langchain.svg?style=social&label=Follow%20%40LangChain" alt="Twitter / X"></a>
</div>

LangChain is a framework for building agents and LLM-powered applications. It helps you chain together interoperable components and third-party integrations to simplify AI application development – all while future-proofing decisions as the underlying technology evolves.

```bash
pip install langchain
```

If you're looking for more advanced customization or agent orchestration, check out [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview), our framework for building controllable agent workflows.

---

**Documentation**:

- [docs.langchain.com](https://docs.langchain.com/oss/python/langchain/overview) – Comprehensive documentation, including conceptual overviews and guides
- [reference.langchain.com/python](https://reference.langchain.com/python) – API reference docs for LangChain packages
- [Chat LangChain](https://chat.langchain.com/) – Chat with the LangChain documentation and get answers to your questions

**Discussions**: Visit the [LangChain Forum](https://forum.langchain.com) to connect with the community and share all of your technical questions, ideas, and feedback.

> [!NOTE]
> Looking for the JS/TS library? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Why use LangChain?

LangChain helps developers build applications powered by LLMs through a standard interface for models, embeddings, vector stores, and more.

Use LangChain for:

- **Real-time data augmentation**. Easily connect LLMs to diverse data sources and external/internal systems, drawing from LangChain's vast library of integrations with model providers, tools, vector stores, retrievers, and more.
- **Model interoperability**. Swap models in and out as your engineering team experiments to find the best choice for your application's needs. As the industry frontier evolves, adapt quickly – LangChain's abstractions keep you moving without losing momentum.
- **Rapid prototyping**. Quickly build and iterate on LLM applications with LangChain's modular, component-based architecture. Test different approaches and workflows without rebuilding from scratch, accelerating your development cycle.
- **Production-ready features**. Deploy reliable applications with built-in support for monitoring, evaluation, and debugging through integrations like LangSmith. Scale with confidence using battle-tested patterns and best practices.
- **Vibrant community and ecosystem**. Leverage a rich ecosystem of integrations, templates, and community-contributed components. Benefit from continuous improvements and stay up-to-date with the latest AI developments through an active open-source community.
- **Flexible abstraction layers**. Work at the level of abstraction that suits your needs - from high-level chains for quick starts to low-level components for fine-grained control. LangChain grows with your application's complexity.

## LangChain ecosystem

While the LangChain framework can be used standalone, it also integrates seamlessly with any LangChain product, giving developers a full suite of tools when building LLM applications.

To improve your LLM application development, pair LangChain with:

- [Deep Agents](https://github.com/langchain-ai/deepagents) *(new!)* – Build agents that can plan, use subagents, and leverage file systems for complex tasks
- [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) – Build agents that can reliably handle complex tasks with LangGraph, our low-level agent orchestration framework. LangGraph offers customizable architecture, long-term memory, and human-in-the-loop workflows – and is trusted in production by companies like LinkedIn, Uber, Klarna, and GitLab.
- [Integrations](https://docs.langchain.com/oss/python/integrations/providers/overview) – List of LangChain integrations, including chat & embedding models, tools & toolkits, and more
- [LangSmith](https://www.langchain.com/langsmith) – Helpful for agent evals and observability. Debug poor-performing LLM app runs, evaluate agent trajectories, gain visibility in production, and improve performance over time.
- [LangSmith Deployment](https://docs.langchain.com/langsmith/deployments) – Deploy and scale agents effortlessly with a purpose-built deployment platform for long-running, stateful workflows. Discover, reuse, configure, and share agents across teams – and iterate quickly with visual prototyping in [LangSmith Studio](https://docs.langchain.com/langsmith/studio).

## Enterprise Agents Deployment

This repository includes a production-ready **Enterprise Agents Platform** in the [deployment/](deployment/) folder, built with LangChain and LangGraph for real-world AI agent applications.

### 🚀 Key Features

- **12 Production Agents**: Research, Content Generation (HITL), Data Analysis, Document Processing, Multilingual RAG, IT Support (HITL), ServiceNow ITSM, Code Assistant, Document Intelligence, Employee Experience, Recruitment, **Software Development**
- **3 Deep Agents**: IT Operations (6 subagents), Sales Intelligence, Recruitment (5 subagents) — all with planning, streaming, and context persistence
- **4 IT Support Agents**: IT Helpdesk, ServiceNow, Document Intelligence, Employee Experience — conversational agents with session memory
- **🆕 Software Development Deep Agent**: AI-powered SDLC automation with 9 specialized subagents, 54 purpose-built tools, end-to-end workflow from requirements to deployment
- **Sales Intelligence Deep Agent**: AI-powered sales analysis with CRM, competitor, pricing, and knowledge tools
- **Recruitment Deep Agent**: AI-powered end-to-end hiring automation with SharePoint integration, 5 specialized subagents, L1/L2/L3 screening, technical assessments, and Excel reporting
- **IT Operations Deep Agent**: Advanced planning agent with 6 specialized subagents, streaming responses, and reasoning model support
- **⚡ Real-time Streaming**: Server-Sent Events (SSE) for live progress updates and tool execution visibility
- **🧠 Reasoning Models**: Native support for OpenAI o1/o3/o4 series with automatic temperature bypass
- **ServiceNow Integration**: Full ITSM operations with 10 tools - incidents, change requests, service requests, CMDB, SLA monitoring, knowledge base
- **LangGraph Orchestration**: State-based agent workflows with human-in-the-loop capabilities
- **REST API**: FastAPI server with LangServe endpoints for seamless integration
- **Microsoft Copilot Studio**: Ready-to-use webhooks for enterprise chatbot integration
- **Security**: API key authentication, CORS configuration, secrets management
- **Observability**: LangSmith tracing for debugging and performance monitoring
- **Evaluation Framework**: Automated agent testing with custom metrics
- **Docker Deployment**: Production-ready containerization with multi-stage builds

### 📚 Quick Start

```bash
cd deployment

# Install dependencies
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your API keys (Azure OpenAI or OpenAI/Anthropic + LangSmith)

# Run locally (Linux/macOS)
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000

# Run locally (Windows — use deployment .venv directly to avoid uv conflicts)
.venv\Scripts\uvicorn.exe app.server:app --host 0.0.0.0 --port 8000

# Or use Docker
docker-compose up --build

# Or use LangGraph Studio UI for visual development
cd deployment
.\start_studio.ps1
# Access at: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

> **Windows tip**: If `uv run uvicorn` fails with `VIRTUAL_ENV` conflicts or `Access denied`
> errors, use `.venv\Scripts\uvicorn.exe` directly. See [docs/SETUP.md](docs/SETUP.md#windows-startup-issues) for details.

### 🎨 LangGraph Studio UI

**Visual development interface** for building and debugging agents without Docker:

```bash
.venv\Scripts\python.exe -m langgraph_cli dev --port 2024 --allow-blocking
```

**Features:**
- 🔍 Visual graph editor with real-time workflow visualization
- 🐛 Interactive debugging with step-by-step execution
- 🔧 Tool call inspection and state management
- ⚡ Hot reload for rapid iteration
- 🚀 No Docker required - runs entirely in-memory

**Access:**
- Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- API: http://127.0.0.1:2024
- Docs: http://127.0.0.1:2024/docs

**Available Agents:**
- ServiceNow ITSM Agent
- Document Processing Agent
- IT Helpdesk Agent
- IT Operations Deep Agent (with 6 subagents)
- Sales Intelligence Deep Agent
- Recruitment Deep Agent (with 5 subagents)
- **Software Development Deep Agent** (with 9 subagents) - **NEW!**

See [LANGGRAPH_SETUP.md](deployment/LANGGRAPH_SETUP.md) for detailed setup instructions.

### 🎯 Recruitment Deep Agent - AI-Powered Hiring Automation

**Complete end-to-end recruitment workflow automation** with SharePoint integration:

**Quick Start:**
```bash
# Configure SharePoint (or skip for demo mode)
export SHAREPOINT_SITE_URL=https://yourcompany.sharepoint.com/sites/Recruitment
export SHAREPOINT_TENANT_ID=your-tenant-id
export SHAREPOINT_CLIENT_ID=your-client-id
export SHAREPOINT_CLIENT_SECRET=your-secret

# Start Recruitment Agent
POST /api/recruitment-agent/start
{
  "user_id": "hr_manager",
  "job_description_id": "JD-001"
}

# Chat with agent
POST /api/recruitment-agent/chat
{
  "session_id": "rec_abc123",
  "message": "Screen all resumes and generate shortlist for Python Developer position"
}
```

**5 Specialized Subagents:**
1. **Document Manager**: SharePoint operations (list, download, upload, search)
2. **Resume Screener**: L1/L2/L3 candidate screening with weighted scoring
3. **Question Generator**: Technical interview questions (MCQ, Coding, Scenario)
4. **Answer Evaluator**: Automated grading with constructive feedback
5. **Report Generator**: Excel exports and shortlist production

**Key Features:**
- 📄 **SharePoint Integration**: Azure AD authentication, document lifecycle management
- 🎯 **Multi-Level Screening**: L1/L2/L3 classification (60%/70%/80% thresholds)
- 📝 **Technical Assessments**: Skill-matched questions with difficulty distributions
- 📊 **Excel Reporting**: CSV exports with rankings and comprehensive analytics
- ⚙️ **Configurable**: Adjust weights, thresholds, question counts
- 🔒 **Secure**: PII handling, session isolation, role-based access
- 📈 **Analytics**: Candidate rankings, skill gap analysis, recommendations

**Workflow Example:**
```python
# 1. Parse job description
jd = parse_job_description("Python Developer - 5+ years Django/AWS")

# 2. Screen candidates from SharePoint
candidates = batch_screen_resumes(jd_id=jd.id)
# → 15 candidates, 8 meet requirements (53% pass rate)

# 3. Generate interview questions
for candidate in shortlisted:
    questions = generate_interview_questions(
        candidate_id=candidate.id,
        skills=candidate.skills,
        level=candidate.level  # L1/L2/L3
    )

# 4. Evaluate submitted answers
evaluation = evaluate_candidate_answers(set_id=questions.set_id)
# → 82% score, PASSED, recommend for L2 interview

# 5. Generate final reports
report = generate_scoring_report(jd_id=jd.id)
excel = export_scoring_excel(jd_id=jd.id)
shortlist = generate_shortlist_report(jd_id=jd.id)
```

**Configuration** (`deployment/.env`):
```bash
# Passing scores by level
L1_PASSING_SCORE=60  # Junior (0-3 years)
L2_PASSING_SCORE=70  # Mid-level (3-7 years)
L3_PASSING_SCORE=80  # Senior (7+ years)

# Score weights
TECHNICAL_WEIGHT=0.40    # 40% - Technical skills match
EXPERIENCE_WEIGHT=0.25   # 25% - Years of experience
EDUCATION_WEIGHT=0.15    # 15% - Education level
SOFT_SKILLS_WEIGHT=0.10  # 10% - Soft skills
CERTIFICATION_WEIGHT=0.10 # 10% - Certifications
```

**Testing:**
```bash
# Run comprehensive test suite (52 tests)
pytest tests/test_recruitment_agent.py -v

# Test coverage:
# - SharePoint tools (7 tests)
# - Resume screening (8 tests)
# - Interview generation (8 tests)
# - Scoring & reporting (5 tests)
# - E2E workflows (4 tests)
```

See [deployment/KNOWLEDGE.md](deployment/KNOWLEDGE.md#recruitment-deep-agent) for complete documentation.

### � Software Development Deep Agent - AI-Powered SDLC Automation

**Comprehensive end-to-end Software Development Lifecycle automation** with intelligent orchestration:

**Quick Start:**
```bash
# Access the Software Development Agent UI
http://localhost:8000/software-dev-chat

# Or use API
POST /api/software-dev-agent/start
{
  "user_id": "developer_001"
}

# Chat with agent
POST /api/software-dev-agent/chat
{
  "session_id": "sdlc_xyz789",
  "message": "Create a REST API for user authentication with JWT tokens and rate limiting"
}
```

**9 Specialized Subagents:**
1. **Requirements Intelligence**: Extract and validate software requirements
2. **Architecture Design**: Design system architecture and APIs
3. **Code Generator**: Generate production-ready code
4. **Code Reviewer**: Perform automated code reviews
5. **Testing Automation**: Create and run comprehensive tests
6. **Debugging & Optimization**: Debug issues and optimize performance
7. **Security Compliance**: Scan for vulnerabilities and ensure compliance
8. **CI/CD Pipeline**: Setup and manage deployment pipelines
9. **Documentation Generator**: Create technical documentation

**Key Features:**
- 🎯 **End-to-End SDLC**: Complete automation from requirements to deployment
- 🛠️ **54 Purpose-Built Tools**: Specialized tools for every development phase
- 📝 **Automatic Phase Transitions**: Intelligent workflow progression based on task completion
- ⚡ **Real-Time Streaming**: Live visibility into agent thinking and tool execution
- 🔒 **Security-First**: Built-in vulnerability scanning and security best practices
- 🧪 **Test Automation**: Unit, integration, and E2E test generation
- 📊 **Code Quality**: Automated reviews with actionable feedback
- 🚀 **CI/CD Integration**: Pipeline setup with GitHub Actions, GitLab CI, Jenkins
- 📚 **Auto Documentation**: Technical specs, API docs, and architecture diagrams

**SDLC Phases:**
1. **Requirements Analysis**: User story creation, acceptance criteria, technical specifications
2. **Architecture Design**: System design, API specs, database schema, technology selection
3. **Implementation**: Code generation, module development, integration
4. **Code Review**: Static analysis, security scanning, best practices validation
5. **Testing**: Unit tests, integration tests, E2E tests, performance testing
6. **Debugging**: Issue diagnosis, root cause analysis, performance optimization
7. **Security**: Vulnerability scanning, dependency audits, compliance checks
8. **Deployment**: CI/CD pipeline setup, containerization, infrastructure as code
9. **Documentation**: README, API docs, architecture diagrams, user guides

**Workflow Example:**
```python
# 1. Analyze requirements
requirements = analyze_requirements(
    "Build a microservice for order processing with event-driven architecture"
)
# → 8 user stories, 15 acceptance criteria, tech stack recommendations

# 2. Design architecture
design = design_architecture(requirements_id=requirements.id)
# → System diagram, API specs, database schema, service boundaries

# 3. Generate code
code = generate_code(design_id=design.id, component="order-service")
# → FastAPI service with SQLAlchemy models, Pydantic schemas, async handlers

# 4. Run code review
review = review_code(code_id=code.id)
# → 12 suggestions: 3 security issues, 5 best practices, 4 optimizations

# 5. Generate tests
tests = generate_tests(code_id=code.id, coverage_target=80)
# → 45 unit tests, 12 integration tests, 85% coverage achieved

# 6. Setup CI/CD
pipeline = setup_cicd(project_id=code.project_id, platform="github-actions")
# → GitHub Actions workflow: lint, test, build, deploy to staging/prod

# 7. Generate documentation
docs = generate_documentation(project_id=code.project_id)
# → README, API docs (OpenAPI), architecture diagrams, deployment guide
```

**Configuration** (`deployment/.env`):
```bash
# Software Development Agent
SOFTWARE_DEV_MODEL=gpt-4o          # Main orchestrator model
SOFTWARE_DEV_SUBAGENT_MODEL=gpt-4o-mini  # Subagent model
SOFTWARE_DEV_TEMPERATURE=0.7       # Creativity level

# Code Generation
DEFAULT_LANGUAGE=python            # Primary language
CODE_STYLE=google                  # Style guide
MAX_CODE_LENGTH=10000              # Max tokens per generation

# Testing
MIN_TEST_COVERAGE=80               # Coverage threshold
TEST_FRAMEWORKS=pytest,unittest    # Supported frameworks

# Security
SECURITY_SCAN_ENABLED=true         # Enable vulnerability scanning
DEPENDENCY_AUDIT=true              # Check dependency vulnerabilities
```

**Testing:**
```bash
# Run comprehensive test suite (64 tests)
pytest tests/test_software_dev_agent.py -v

# Test coverage:
# - Requirements intelligence (8 tests)
# - Architecture design (7 tests)
# - Code generation (12 tests)
# - Code review (9 tests)
# - Testing automation (8 tests)
# - Debugging & optimization (6 tests)
# - Security compliance (7 tests)
# - CI/CD setup (5 tests)
# - Documentation generation (2 tests)
```

#### 🔧 Bash Execution & Azure Integration - NEW!

The Software Development Deep Agent now includes **secure bash execution** capabilities and **Azure cloud integration** for automated SDLC workflows.

**Bash Execution Tools** (`deployment/app/deepagents/software_dev/tools/bash_execution_tools.py`):

**Key Features:**
- ✅ **Multi-platform Support**: Bash (Linux/macOS), PowerShell (Windows), CMD fallback
- 🛡️ **Security Validation**: Blocks dangerous commands (rm -rf /, fork bombs, dd to devices)
- ⚠️ **Warning System**: Flags risky operations (sudo, recursive deletes, curl | bash)
- 🔄 **Cross-platform Detection**: Automatic shell selection based on OS
- ⏱️ **Timeout Protection**: Configurable execution timeout (default 30s)
- 📜 **Command History**: Tracks executed commands for auditing

**Available Tools** (4 total):

| Tool | Purpose |
|------|---------|
| `execute_bash_command` | Execute shell commands with security validation |
| `execute_python_code` | Run Python code snippets in isolated environment |
| `execute_tests_real` | Run test suites (pytest, npm test, cargo test) |
| `install_dependencies` | Install package dependencies (pip, npm, cargo) |

**Security Patterns:**

```python
# 🚫 Blocked dangerous commands
rm -rf /                    # Root directory deletion
:(){ :|:& };:              # Fork bomb
dd if=... of=/dev/...      # Writing to devices
mkfs.ext4 /dev/...         # Filesystem formatting

# ⚠️ Warned risky commands
rm -rf /tmp/mydir          # Recursive deletion
sudo apt-get install       # Elevated privileges
curl ... | bash            # Piping web content to shell
```

**Usage Example:**

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
# → { "success": true, "stdout": "...", "exit_code": 0, "shell_type": "bash" }

# Run Python code
result = execute_python_code.invoke({
    "code": "print('Hello, World!')",
    "timeout": 10
})
# → { "success": true, "stdout": "Hello, World!\n", "stderr": "", "exit_code": 0, "code": "print('Hello, World!')" }

# Run tests
result = execute_tests_real.invoke({
    "test_framework": "pytest",
    "test_path": "tests/unit/",
    "additional_args": "-v --cov"
})
# → { "success": true, "stdout": "...", "stderr": "", "exit_code": 0, "command": "pytest tests/unit/ -v --cov", "shell_type": "bash", "test_framework": "pytest", "test_path": "tests/unit/" }
```

**Azure Integration** (`deployment/app/deepagents/software_dev/tools/azure_integration.py`):

**Supported Azure Services:**

| Service | Purpose | Configuration |
|---------|---------|---------------|
| **Azure Container Instances (ACI)** | Ephemeral command execution | `ACI_CONTAINER_GROUP_NAME`, `ACI_CONTAINER_IMAGE` |
| **Azure Functions** | Serverless command execution | `AZURE_FUNCTIONS_APP_NAME`, `AZURE_FUNCTIONS_RUNTIME` |
| **Azure Kubernetes Service (AKS)** | Production-grade container orchestration | `AKS_CLUSTER_NAME`, `AKS_DEPLOYMENT_NAME` |
| **Azure App Service** | Web app deployment and execution | `AZURE_APP_SERVICE_NAME`, `AZURE_APP_SERVICE_PLAN` |
| **Azure Key Vault** | Secrets management for credentials | `AZURE_KEY_VAULT_NAME`, `AZURE_KEY_VAULT_URI` |

**Configuration** (`.azure.config`):

```bash
# Copy example and configure your Azure resources
cp deployment/.azure.config.example deployment/.azure.config

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

**Security Best Practices:**
- ✅ `.azure.config` excluded from version control via `.gitignore`
- ✅ Use Azure Key Vault for sensitive credentials
- ✅ Configure managed identities for passwordless authentication
- ✅ Implement RBAC policies for least-privilege access
- ✅ Enable Azure Monitor for audit logging

**Integration with Code Generation:**

The Code Generator subagent now has access to bash execution tools for:
- 🎨 Running code formatters (black, prettier, gofmt)
- 🔍 Executing linters (ruff, eslint, clippy)
- ✅ Verifying generated code through execution
- 📦 Installing dependencies automatically

**Testing:**

```bash
# Run comprehensive bash execution test suite
pytest tests/test_bash_execution_tools.py -v

# Test coverage (52 tests):
# - Security validation (10 tests) - dangerous/risky command detection
# - Cross-platform shell detection (4 tests)
# - Command execution (15 tests) - various scenarios
# - Error handling and timeout (8 tests)
# - Python code execution (8 tests)
# - Test framework integration (7 tests)
```

**Real-World Use Cases:**
- 🚀 **Automated Formatting**: Run black, prettier, gofmt on generated code
- 🔍 **Lint Enforcement**: Execute ruff, eslint, clippy to catch issues early
- ✅ **Continuous Testing**: Run pytest, npm test, cargo test after code changes
- 📦 **Dependency Management**: Auto-install packages with pip, npm, cargo
- 🏗️ **Build Automation**: Execute build commands (npm run build, cargo build)
- 🔄 **Git Operations**: Automate git commands (status, commit, push)
- ☁️ **Cloud Deployment**: Deploy to Azure via ACI, Functions, AKS, App Service

**Real-World Use Cases:**
- 🚀 **Rapid Prototyping**: Generate MVP in hours instead of weeks
- 🏢 **Enterprise Development**: Maintain consistency across large teams
- 🔄 **Legacy Modernization**: Refactor and upgrade legacy systems
- 🛡️ **Security Hardening**: Automated security reviews and remediation
- 📈 **Performance Optimization**: Identify and fix bottlenecks
- 📦 **Microservices**: Design and implement service architectures
- 🔧 **DevOps Automation**: Complete CI/CD pipeline setup

See [deployment/KNOWLEDGE.md](deployment/KNOWLEDGE.md#software-development-deep-agent) for complete documentation.

### �🔗 Integration Examples

**Copilot Studio Webhook:**
```
POST /webhook/research
POST /webhook/content
POST /webhook/data_analyst
# ... (7 total endpoints)
```

**Python SDK:**
```python
from langserve import RemoteRunnable

agent = RemoteRunnable("http://localhost:8000/research")
response = agent.invoke({"messages": [{"role": "user", "content": "Research AI trends"}]})
```

### 📖 Documentation

Comprehensive documentation available in [deployment/docs/](deployment/docs/):

- [Architecture Blueprint](deployment/docs/Project_Architecture_Blueprint.md) – System design, patterns, and extension points
- [Deployment Guide](deployment/docs/DEPLOYMENT.md) – Local, Docker, Azure deployment strategies
- [API Reference](deployment/docs/api/README.md) – Complete endpoint documentation
- [Security Guide](deployment/docs/SECURITY.md) – Authentication, secrets, compliance
- [Operations Manual](deployment/docs/OPERATIONS.md) – Monitoring, troubleshooting, incident response
- [Setup Guide](deployment/docs/SETUP.md) – Developer onboarding and prerequisites

### 🏗️ Architecture

The platform follows a **layered architecture** with clear separation of concerns:

```
Presentation Layer (FastAPI endpoints)
    ↓
Middleware Layer (Authentication, CORS, Error Handling)
    ↓
Application Layer (Agent orchestration, State management)
    ↓
Domain Layer (Agent implementations with LangGraph)
    ↓
Infrastructure Layer (LLM providers, Vector stores, External APIs)
```

**Design Patterns**: Template Method (agent base), Abstract Factory (agent creation), State Pattern (LangGraph), Strategy (tool selection), Decorator (middleware), Facade (API), Observer (tracing)

### 🤖 Deep Agent Architecture

The **IT Operations Deep Agent** represents the next generation of enterprise AI agents with advanced capabilities:

#### Core Features

- **📋 Planning & Task Management**: Multi-step task decomposition with todo tracking and status updates
- **📁 Context Management**: Virtual file system for maintaining analysis context across conversations
- **🔀 Subagent Delegation**: Six specialized subagents for domain-specific operations:
  - **Incident Agent**: Incident search, creation, updates, and escalation
  - **Change Agent**: Change request validation and risk assessment
  - **Problem Agent**: Problem investigation and known error management
  - **Asset Agent**: CMDB queries and CI relationship mapping
  - **SLA Agent**: SLA monitoring, breach prediction, and reporting
  - **Knowledge Agent**: Knowledge base search and article creation
- **⚡ Real-time Streaming**: SSE endpoint for live progress visibility
- **🧠 Reasoning Model Support**: Automatic detection and configuration for OpenAI o1/o3/o4 models
- **💾 Persistent Storage**: File-based session storage with context isolation

#### Streaming API

```javascript
const eventSource = new EventSource('/api/deepagent/chat/stream', {
  method: 'POST',
  body: JSON.stringify({
    session_id: 'deepagent-123',
    message: 'Analyze P1 incidents this week'
  })
});

eventSource.addEventListener('thinking', (e) => {
  console.log('Agent thinking:', JSON.parse(e.data).content);
});

eventSource.addEventListener('tool_call', (e) => {
  const { tool, args, description } = JSON.parse(e.data);
  console.log(`Executing ${tool}:`, description);
});

eventSource.addEventListener('content', (e) => {
  console.log('Response:', JSON.parse(e.data).response);
});
```

#### LLM Configuration

```env
# Standard Models (with temperature control)
DEEP_AGENT_MODEL=gpt-4o        # Fast, cost-effective
DEEP_AGENT_MODEL=gpt-4o-mini   # Lightweight option

# Reasoning Models (temperature auto-bypassed)
DEEP_AGENT_MODEL=o1            # Advanced reasoning
DEEP_AGENT_MODEL=o3-mini       # Balanced reasoning
DEEP_AGENT_MODEL=o4-mini       # Latest reasoning model
```

The agent automatically detects reasoning models by prefix (o1, o3, o4) and bypasses temperature settings to comply with OpenAI API requirements.

#### Usage Example

```bash
# Start session
curl -X POST http://localhost:8000/api/deepagent/start \
  -H "Content-Type: application/json" \
  -d '{"user_id": "ops_user"}'

# Stream agent response
curl -N -X POST http://localhost:8000/api/deepagent/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "deepagent-abc123",
    "message": "Investigate INC0010001 and check for related incidents"
  }'
```

**Real-world Scenarios:**
- Complex incident pattern analysis across multiple systems
- Change impact assessment with CI dependency mapping
- Root cause investigation with automated problem record creation
- SLA breach prediction and proactive escalation
- Knowledge base enrichment from resolved incidents

### 🎯 Use Cases

- **Enterprise Knowledge Management**: RAG-based document querying with multilingual support
- **Content Creation Workflows**: Human-in-the-loop content generation with approval gates
- **Data Analysis Automation**: Automated insights generation with visualization
- **IT Service Desk**: Intelligent ticket routing and resolution assistance
- **ServiceNow ITSM Operations**: Incident, change request, and service request management
- **Developer Productivity**: Code generation, review, and debugging assistance
- **Research Automation**: Multi-source information gathering and synthesis

### 🔧 Extending the Platform

Add new agents by following the established patterns:

1. Create agent class inheriting from `BaseAgent` in [deployment/app/agents/](deployment/app/agents/)
2. Define LangGraph workflow with StateGraph
3. Register API endpoint in [deployment/app/server.py](deployment/app/server.py)
4. Add evaluation tests in [deployment/app/agents/evals/](deployment/app/agents/evals/)
5. Update documentation

See [Architecture Blueprint](deployment/docs/Project_Architecture_Blueprint.md#agent-implementation-template) for complete implementation templates.

---

## Additional resources

- [API Reference](https://reference.langchain.com/python) – Detailed reference on navigating base packages and integrations for LangChain.
- [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview) – Learn how to contribute to LangChain projects and find good first issues.
- [Code of Conduct](https://github.com/langchain-ai/langchain/?tab=coc-ov-file) – Our community guidelines and standards for participation.
- [LangChain Academy](https://academy.langchain.com/) – Comprehensive, free courses on LangChain libraries and products, made by the LangChain team.
