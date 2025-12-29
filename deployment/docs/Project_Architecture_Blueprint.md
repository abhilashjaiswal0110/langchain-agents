# Project Architecture Blueprint

> **Document Version**: 1.0.0
> **Last Updated**: December 19, 2025
> **Status**: Production-Ready
> **Classification**: Internal Use

---

## Executive Summary

This document provides a comprehensive architectural blueprint for the **LangChain Enterprise Agents Platform**, a production-ready microservices application that exposes AI agents as REST APIs. The platform implements a modular, extensible architecture using LangGraph state machines, FastAPI for API serving, and LangSmith for full observability.

### Key Architectural Characteristics

| Characteristic | Implementation |
|----------------|----------------|
| **Architecture Pattern** | Microservices with Agent-Based Design |
| **State Management** | LangGraph StateGraph with Pydantic Models |
| **API Framework** | FastAPI with LangServe Integration |
| **Observability** | LangSmith Distributed Tracing |
| **Security Model** | API Key Middleware with CORS |
| **Deployment Model** | Containerized (Docker) |
| **Programming Language** | Python 3.10+ |

---

## Table of Contents

1. [Architectural Overview](#architectural-overview)
2. [Component Architecture](#component-architecture)
3. [Design Patterns](#design-patterns)
4. [Data Architecture](#data-architecture)
5. [API Architecture](#api-architecture)
6. [Security Architecture](#security-architecture)
7. [Agent Implementation Patterns](#agent-implementation-patterns)
8. [Extension Points](#extension-points)
9. [Deployment Architecture](#deployment-architecture)
10. [Technology Stack](#technology-stack)

---

## Architectural Overview

### System Context Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         External Consumers                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ Web Clients  │  │ Mobile Apps  │  │ CLI Tools    │  │ 3rd Party   │ │
│  │              │  │              │  │              │  │ Platforms   │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘ │
│         │                 │                 │                 │         │
└─────────┼─────────────────┼─────────────────┼─────────────────┼─────────┘
          │                 │                 │                 │
          └─────────────────┴─────────────────┴─────────────────┘
                                    │
                          ┌─────────▼──────────┐
                          │   API Gateway      │
                          │  (Optional ngrok)  │
                          └─────────┬──────────┘
                                    │
          ┌─────────────────────────▼─────────────────────────┐
          │                                                    │
          │        LangChain Enterprise Agents Platform       │
          │                                                    │
          │  ┌──────────────────────────────────────────────┐ │
          │  │        FastAPI Application Layer             │ │
          │  │  - API Key Middleware                        │ │
          │  │  - CORS Middleware                           │ │
          │  │  - Request Validation (Pydantic)             │ │
          │  └──────────────────────────────────────────────┘ │
          │                      │                            │
          │  ┌──────────────────┴────────────────────────┐   │
          │  │                                            │   │
          │  │         Agent Orchestration Layer         │   │
          │  │                                            │   │
          │  │  ┌──────────────┐  ┌──────────────┐       │   │
          │  │  │ Enterprise   │  │ LangServe    │       │   │
          │  │  │ Agents       │  │ Routes       │       │   │
          │  │  │ (7 agents)   │  │ (Legacy)     │       │   │
          │  │  └──────────────┘  └──────────────┘       │   │
          │  │                                            │   │
          │  └────────────────────────────────────────────┘   │
          │                      │                            │
          │  ┌──────────────────┴────────────────────────┐   │
          │  │                                            │   │
          │  │         LangGraph State Machine           │   │
          │  │                                            │   │
          │  │  - StateGraph Execution                   │   │
          │  │  - Tool Node Invocation                   │   │
          │  │  - Conditional Routing                    │   │
          │  │  - Checkpointing (MemorySaver)            │   │
          │  │                                            │   │
          │  └────────────────────────────────────────────┘   │
          │                      │                            │
          │  ┌──────────────────┴────────────────────────┐   │
          │  │                                            │   │
          │  │         LangSmith Tracing Layer           │   │
          │  │                                            │   │
          │  │  - Request/Response Tracing               │   │
          │  │  - Tool Execution Tracking                │   │
          │  │  - Performance Metrics                    │   │
          │  │                                            │   │
          │  └────────────────────────────────────────────┘   │
          │                                                    │
          └────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
    ┌──────────┐            ┌──────────┐            ┌──────────┐
    │   LLM    │            │ External │            │  3rd     │
    │ Providers│            │  Tools   │            │ Party    │
    │          │            │          │            │ Services │
    │ • OpenAI │            │ • Tavily │            │          │
    │ • Anthropic          │ • WebFetch│            │ • Copilot│
    └──────────┘            └──────────┘            │ • Azure  │
                                                    │ • AWS    │
                                                    └──────────┘
```

### High-Level Architecture Principles

1. **Separation of Concerns**: Clear boundaries between API layer, agent layer, and execution layer
2. **Dependency Inversion**: Abstract base classes enable multiple LLM providers
3. **Stateful Design**: LangGraph StateGraph manages conversation state
4. **Observable by Default**: LangSmith tracing on all executions
5. **Security First**: API key authentication, no hardcoded secrets
6. **Modular Extensibility**: New agents follow BaseAgent pattern

---

## Component Architecture

### Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Presentation Layer                          │
│  - FastAPI Routes                                               │
│  - OpenAPI/Swagger UI                                           │
│  - Static Web UI                                                │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────────┐
│                     Middleware Layer                           │
│  - APIKeyMiddleware (Authentication)                           │
│  - CORSMiddleware (Cross-origin)                               │
│  - ExceptionHandlers                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────────┐
│                   Application Layer                            │
│  - Enterprise Agent Endpoints (/api/enterprise/*)              │
│  - Webhook Integration Endpoints (/api/webhooks/*)            │
│  - Conversation Management (/api/conversation/*)               │
│  - LangServe Routes (Legacy)                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────────┐
│                     Domain Layer                               │
│  - BaseAgent (Abstract)                                        │
│  - 7 Enterprise Agents (Concrete)                              │
│  - AgentConfig                                                 │
│  - State Models (Pydantic)                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────────┐
│                   Infrastructure Layer                         │
│  - LangGraph StateGraph                                        │
│  - LLM Clients (OpenAI/Anthropic)                              │
│  - Tool Execution Engine                                       │
│  - Checkpointer (MemorySaver)                                  │
│  - LangSmith Tracer                                            │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Server Module (`app/server.py`)

**Purpose**: FastAPI application entry point and routing

**Key Responsibilities**:
- Initialize tracing configuration
- Load and register agents
- Define API endpoints
- Configure middleware
- Manage application lifecycle

**Dependencies**:
```python
- fastapi (Web framework)
- langserve (LangChain serving)
- pydantic (Data validation)
- dotenv (Configuration)
```

**Component Structure**:
```
server.py
├── setup_langsmith_tracing()      # Tracing initialization
├── APIKeyMiddleware               # Security middleware
├── load_chains()                  # Legacy chain loading
├── load_enterprise_agents()       # Agent initialization
├── API Endpoints
│   ├── Health/Status              # /health, /ready
│   ├── Enterprise Agents          # /api/enterprise/*
│   ├── Webhooks                   # /api/webhooks/*
│   └── LangServe Routes           # /chat, /rag, /agent
└── Application Lifespan Manager   # Startup/Shutdown
```

#### 2. BaseAgent Module (`app/agents/base/agent_base.py`)

**Purpose**: Abstract base class for all agents

**Key Components**:

```python
@dataclass
class AgentConfig:
    """Configuration for agent initialization"""
    model_provider: Literal["openai", "anthropic", "auto"]
    model_name: str | None
    temperature: float = 0.7
    max_tokens: int = 4096
    checkpointer: BaseCheckpointSaver | None
    tracing_enabled: bool = True
```

```python
class BaseAgentState(BaseModel):
    """Base state schema with message history"""
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str | None
    user_id: str | None
    metadata: dict[str, Any]
```

```python
class BaseAgent(ABC):
    """Abstract base with common functionality"""

    @abstractmethod
    def _build_graph(self) -> StateGraph:
        """Subclasses define workflow graph"""

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Subclasses define system prompt"""

    def invoke(self, message: str, **kwargs) -> dict:
        """Execute agent synchronously"""

    async def astream(self, message: str, **kwargs) -> AsyncIterator:
        """Stream agent execution"""
```

**Pattern**: Template Method + Abstract Factory

#### 3. Tool Utilities (`app/agents/base/tools.py`)

**Purpose**: Shared tool creation and error handling

**Key Functions**:
```python
def tool_error_handler(func: F) -> F:
    """Decorator for graceful tool error handling"""

def sanitize_output(output: str) -> str:
    """Remove potentially sensitive information"""

def chunk_text(text: str, chunk_size: int = 1000) -> list[str]:
    """Split text into manageable chunks"""
```

**Pattern**: Decorator Pattern for Cross-cutting Concerns

#### 4. Enterprise Agents

Seven specialized agents implementing BaseAgent:

| Agent | Module | Primary Function |
|-------|--------|------------------|
| **ResearchAgent** | `research/` | Web search and synthesis |
| **ContentAgent** | `content/` | Content creation with HITL |
| **DataAnalystAgent** | `data_analyst/` | Data analysis and visualization |
| **DocumentAgent** | `documents/` | SOP/WLI/Policy generation |
| **MultilingualRAGAgent** | `rag/` | Multi-language document Q&A |
| **HITLSupportAgent** | `it_support/` | IT support with approval gates |
| **CodeAssistantAgent** | `code_assistant/` | Code review and modernization |

**Common Structure**:
```
agent_module/
├── agent_name.py           # Agent implementation
├── __init__.py             # Exports
└── [domain_specific_tools.py]  # Optional tools
```

---

## Design Patterns

### 1. Template Method Pattern

**Location**: `BaseAgent` class

**Purpose**: Define algorithm skeleton, let subclasses provide specifics

**Implementation**:
```python
class BaseAgent(ABC):
    def invoke(self, message: str, **kwargs) -> dict:
        # Template method
        if self._compiled_graph is None:
            self.compile()  # Step 1

        input_state = self._build_input(message, **kwargs)  # Step 2
        result = self._compiled_graph.invoke(input_state)   # Step 3
        return result

    @abstractmethod
    def _build_graph(self) -> StateGraph:
        # Subclass defines workflow
        pass
```

### 2. Abstract Factory Pattern

**Location**: Agent initialization in `BaseAgent._init_llm()`

**Purpose**: Create LLM instances without specifying concrete classes

**Implementation**:
```python
def _init_llm(self) -> None:
    provider = self.config.model_provider

    if provider == "auto":
        # Auto-detection logic
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"

    if provider == "openai":
        self._llm = ChatOpenAI(...)  # OpenAI factory
    elif provider == "anthropic":
        self._llm = ChatAnthropic(...)  # Anthropic factory
```

### 3. State Pattern

**Location**: LangGraph StateGraph nodes

**Purpose**: Encapsulate state-dependent behavior

**Implementation**:
```python
def _build_graph(self) -> StateGraph:
    graph = StateGraph(AgentState)

    # States as nodes
    graph.add_node("agent", self._call_model)
    graph.add_node("tools", ToolNode(self._tools))

    # Transitions as edges
    graph.add_conditional_edges(
        "agent",
        self._should_continue,  # State transition logic
        {"tools": "tools", "end": END}
    )

    return graph
```

### 4. Strategy Pattern

**Location**: LLM provider selection

**Purpose**: Select algorithm at runtime

**Implementation**: AgentConfig allows switching between OpenAI/Anthropic without code changes

### 5. Decorator Pattern

**Location**: Tool error handling and tracing

**Purpose**: Add behavior without modifying original functions

**Implementation**:
```python
@tool
@tool_error_handler
def search_web(query: str) -> str:
    """Tool with error handling and LangChain registration"""
    return perform_search(query)
```

### 6. Observer Pattern (Implicit)

**Location**: LangSmith tracing

**Purpose**: Monitor execution without modifying agent logic

**Implementation**: `@traceable` decorator observes method execution

### 7. Facade Pattern

**Location**: Server API endpoints

**Purpose**: Provide simplified interface to complex agent subsystem

**Implementation**:
```python
@app.post("/api/enterprise/research/invoke")
async def research_invoke(request: ResearchAgentRequest):
    # Facade hiding complexity of agent initialization,
    # state management, and graph execution
    result = research_agent.invoke(message=request.query)
    return EnterpriseAgentResponse(...)
```

---

## Data Architecture

### State Management Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  State Lifecycle                            │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│  │  Input   │───>│ State    │───>│  Output  │            │
│  │ Message  │    │Transform │    │ Message  │            │
│  └──────────┘    └──────────┘    └──────────┘            │
│                         │                                  │
│                         ▼                                  │
│                  ┌──────────┐                              │
│                  │Checkpointer                             │
│                  │(MemorySaver)                            │
│                  └──────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

### State Schema Hierarchy

```python
BaseAgentState (Pydantic BaseModel)
    ├── messages: Annotated[list[BaseMessage], add_messages]
    ├── session_id: str | None
    ├── user_id: str | None
    └── metadata: dict[str, Any]

ResearchState(BaseAgentState)
    ├── [inherits base fields]
    ├── query: str
    ├── research_plan: list[str]
    ├── sources: list[dict[str, Any]]
    └── findings: list[str]

ContentState(BaseAgentState)
    ├── [inherits base fields]
    ├── topic: str
    ├── platform: Literal["linkedin", "x", "blog"]
    ├── draft_content: str | None
    ├── feedback: str | None
    └── status: Literal["planning", "drafting", ...]

DataAnalystState(BaseAgentState)
    ├── [inherits base fields]
    ├── data_source: str
    ├── data_type: Literal["excel", "csv", "database"]
    ├── columns: list[str]
    └── analysis_results: dict[str, Any] | None

[... other agent states follow same pattern]
```

### Data Flow Patterns

#### 1. Request Flow

```
HTTP Request (JSON)
    │
    ▼
Pydantic Request Model (Validation)
    │
    ▼
Agent.invoke(message, **kwargs)
    │
    ▼
Build Input State
    │
    ▼
StateGraph.invoke(input_state)
    │
    ▼
Execute Nodes (agent → tools → agent)
    │
    ▼
Extract Response from State
    │
    ▼
Pydantic Response Model
    │
    ▼
HTTP Response (JSON)
```

#### 2. State Update Pattern

```python
# Pydantic models with add_messages reducer
messages: Annotated[list[BaseMessage], add_messages]

# State updates are additive:
def agent_node(state: AgentState) -> dict:
    response = llm.invoke(state.messages)
    # Returns partial update - messages are appended
    return {"messages": [response]}
```

#### 3. Checkpointing Pattern

```python
# MemorySaver stores state by thread_id (session_id)
checkpointer = MemorySaver()

# State persists across invocations
result = graph.invoke(
    input_state,
    config={"configurable": {"thread_id": session_id}}
)
```

### Data Validation

All API inputs/outputs use Pydantic models:

```python
class ResearchAgentRequest(BaseModel):
    query: str  # Required
    session_id: str | None = None  # Optional

class EnterpriseAgentResponse(BaseModel):
    success: bool
    response: str | None
    session_id: str | None
    agent_type: str
    error: str | None
```

**Benefits**:
- Automatic validation
- Type safety
- Auto-generated OpenAPI schema
- Clear API contracts

---

## API Architecture

### API Endpoint Structure

```
├── / (GET)
│   └── Redirect to /docs
│
├── /docs (GET)
│   └── OpenAPI/Swagger UI
│
├── /health (GET)
│   └── Health check with component status
│
├── /ready (GET)
│   └── Kubernetes readiness probe
│
├── /api/enterprise/
│   ├── agents (GET)
│   │   └── List all available agents
│   │
│   ├── research/
│   │   ├── invoke (POST)
│   │   └── stream (POST)
│   │
│   ├── content/
│   │   ├── invoke (POST)
│   │   └── stream (POST)
│   │
│   ├── data-analyst/
│   │   ├── invoke (POST)
│   │   └── stream (POST)
│   │
│   ├── documents/
│   │   ├── invoke (POST)
│   │   └── stream (POST)
│   │
│   ├── multilingual-rag/
│   │   ├── invoke (POST)
│   │   └── stream (POST)
│   │
│   ├── support/
│   │   ├── invoke (POST)
│   │   └── stream (POST)
│   │
│   └── code/
│       ├── invoke (POST)
│       └── stream (POST)
│
├── /api/webhooks/
│   ├── copilot-studio (POST)
│   ├── azure-ai (POST)
│   └── aws-lex (POST)
│
├── /api/conversation/
│   ├── start (POST)
│   ├── chat (POST)
│   ├── {session_id} (GET)
│   └── {session_id} (DELETE)
│
└── /[LangServe Routes]/ (Legacy)
    ├── /chat/invoke
    ├── /rag/invoke
    └── /agent/invoke
```

### Request/Response Patterns

#### Standard Agent Invocation

**Request**:
```json
POST /api/enterprise/research/invoke
{
  "query": "What is LangChain?",
  "session_id": "optional-session-123"
}
```

**Response**:
```json
{
  "success": true,
  "response": "LangChain is a framework...",
  "session_id": "research-session-456",
  "agent_type": "research",
  "tool_calls": null,
  "error": null
}
```

#### Webhook Integration Pattern

**Request**:
```json
POST /api/webhooks/copilot-studio
{
  "query": "Generate a report",
  "agent_type": "document",
  "session_id": "copilot-session-789",
  "user_id": "user-123",
  "conversation_id": "conv-456",
  "metadata": {}
}
```

**Response**:
```json
{
  "success": true,
  "response": "Generated report content...",
  "session_id": "copilot-session-789",
  "agent_type": "document",
  "source": "copilot-studio",
  "metadata": {
    "user_id": "user-123",
    "conversation_id": "conv-456"
  },
  "error": null
}
```

### API Versioning Strategy

**Current State**: Unversioned endpoints

**Future Extensibility**:
```python
# Add version prefix without breaking existing clients
app.include_router(v1_router, prefix="/api/v1")
app.include_router(v2_router, prefix="/api/v2")

# Maintain /api/enterprise/* as latest stable
```

---

## Security Architecture

### Authentication & Authorization

```
┌─────────────────────────────────────────────────────────┐
│               Request Authentication Flow               │
│                                                         │
│  1. HTTP Request                                        │
│     │                                                   │
│     ▼                                                   │
│  2. APIKeyMiddleware.dispatch()                         │
│     │                                                   │
│     ├─> API_KEY_ENABLED?                               │
│     │   ├─> No: Skip auth                              │
│     │   └─> Yes: Continue                              │
│     │                                                   │
│     ├─> Path in PUBLIC_PATHS?                          │
│     │   ├─> Yes: Skip auth                             │
│     │   └─> No: Continue                               │
│     │                                                   │
│     ├─> Check X-API-Key header                         │
│     │   ├─> Valid: Allow                               │
│     │   └─> Invalid: Return 401                        │
│     │                                                   │
│     ▼                                                   │
│  3. Route Handler                                       │
└─────────────────────────────────────────────────────────┘
```

### Security Layers

1. **Network Layer**: HTTPS via ngrok/load balancer
2. **Application Layer**: API Key Middleware
3. **Data Layer**: Environment variables for secrets
4. **Validation Layer**: Pydantic input validation

### Security Configuration

```python
# .env configuration
API_KEY_ENABLED=true
API_KEY=<generated-secure-key>
CORS_ORIGINS=https://trusted-domain.com

# Public endpoints (no auth required)
PUBLIC_PATHS = {
    "/", "/docs", "/redoc", "/openapi.json",
    "/health", "/ready"
}
```

### Secrets Management

**Pattern**: Zero hardcoded secrets

```python
# Always use environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not configured")

# Never commit .env to git
# .gitignore includes:
.env
.env.*
!.env.example
```

---

## Agent Implementation Patterns

### Standard Agent Implementation Template

```python
"""Agent module docstring with purpose and features."""

from typing import Annotated, Any
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from app.agents.base.agent_base import BaseAgent, AgentConfig
from app.agents.base.tools import tool_error_handler

# 1. Define State Schema
class MyAgentState(BaseModel):
    messages: Annotated[list, add_messages] = Field(...)
    session_id: str | None = None
    # Domain-specific fields
    custom_field: str = ""

# 2. Define Agent-Specific Tools
@tool
@tool_error_handler
def my_custom_tool(param: str) -> str:
    """Tool description for LLM."""
    # Implementation
    return result

# 3. Implement Agent Class
class MyAgent(BaseAgent):
    """Agent description."""

    def __init__(self, config: AgentConfig | None = None):
        super().__init__(config)
        self.register_tools([my_custom_tool])

    def _get_system_prompt(self) -> str:
        return """System prompt defining agent behavior."""

    def _build_graph(self) -> StateGraph:
        """Build workflow graph."""

        def call_model(state: MyAgentState) -> dict:
            system = SystemMessage(content=self._get_system_prompt())
            messages = [system] + list(state.messages)
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def should_continue(state: MyAgentState) -> str:
            messages = list(state.messages)
            if not messages:
                return "end"
            last = messages[-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return "end"

        graph = StateGraph(MyAgentState)
        graph.add_node("agent", call_model)
        graph.add_node("tools", ToolNode(self._tools))
        graph.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", "end": END}
        )
        graph.add_edge("tools", "agent")
        graph.add_edge(START, "agent")

        return graph
```

### Agent Workflow Pattern

```
START
  │
  ▼
┌─────────────┐
│ agent_node  │  # LLM generates response
│             │  # May include tool calls
└─────┬───────┘
      │
      ▼
┌──────────────────┐
│ should_continue  │  # Routing decision
└─────┬────────────┘
      │
      ├─> tool_calls?
      │   ├─> Yes ──────┐
      │   │             ▼
      │   │      ┌─────────────┐
      │   │      │ tools_node  │  # Execute tools
      │   │      └─────┬───────┘
      │   │            │
      │   │            └─────> Loop back to agent_node
      │   │
      │   └─> No ─────> END
```

### Human-in-the-Loop (HITL) Pattern

Used in ContentAgent and HITLSupportAgent:

```python
from langgraph.types import interrupt

def _approval_node(self, state: AgentState) -> dict:
    """Request human approval."""

    # Extract content needing approval
    draft = extract_draft_from_state(state)

    # Interrupt execution for human input
    feedback = interrupt({
        "type": "approval_request",
        "draft": draft,
        "options": ["approve", "reject", "revise"]
    })

    # Process feedback
    if feedback["decision"] == "approve":
        return {"status": "approved"}
    elif feedback["decision"] == "revise":
        return {"feedback": feedback["notes"]}
    else:
        return {"status": "rejected"}
```

---

## Extension Points

### Adding a New Agent

**Step-by-Step Process**:

1. **Create Module Structure**
```bash
mkdir app/agents/new_agent
touch app/agents/new_agent/__init__.py
touch app/agents/new_agent/new_agent.py
```

2. **Implement Agent** (follow template pattern above)

3. **Export from Module**
```python
# app/agents/new_agent/__init__.py
from app.agents.new_agent.new_agent import NewAgent, NewAgentState

__all__ = ["NewAgent", "NewAgentState"]
```

4. **Register in Server**
```python
# app/server.py
from app.agents.new_agent import NewAgent

def load_enterprise_agents():
    global new_agent
    new_agent = NewAgent()
    status["new_agent"] = True
```

5. **Add API Endpoint**
```python
@app.post("/api/enterprise/new-agent/invoke")
async def new_agent_invoke(request: EnterpriseAgentRequest):
    if new_agent is None:
        raise HTTPException(503, "Agent not available")
    result = new_agent.invoke(message=request.message)
    return EnterpriseAgentResponse(...)
```

6. **Add Documentation** in ARCHITECTURE.md, API reference, etc.

### Adding Custom Tools

**Pattern**:
```python
from langchain_core.tools import tool
from app.agents.base.tools import tool_error_handler

@tool
@tool_error_handler
def my_custom_tool(
    param1: str,
    param2: int = 10,
    param3: Literal["option1", "option2"] = "option1"
) -> str:
    """Detailed description for LLM.

    Args:
        param1: Parameter description
        param2: Optional parameter with default
        param3: Enum-style parameter

    Returns:
        Description of return value
    """
    # Implementation
    result = perform_operation(param1, param2, param3)
    return result

# Register in agent
class MyAgent(BaseAgent):
    def __init__(self, config: AgentConfig | None = None):
        super().__init__(config)
        self.register_tools([my_custom_tool])
```

### Adding Webhook Integrations

**Pattern**:
```python
class NewPlatformRequest(BaseModel):
    query: str
    agent_type: str
    session_id: str | None = None
    # Platform-specific fields
    platform_id: str
    custom_metadata: dict[str, Any] = {}

@app.post("/api/webhooks/new-platform")
async def new_platform_webhook(
    request: NewPlatformRequest
) -> ThirdPartyResponse:
    # Validate request
    if request.agent_type not in AGENT_TYPES:
        raise HTTPException(400, "Invalid agent_type")

    # Generate session ID
    session_id = request.session_id or f"new-platform-{request.platform_id}"

    # Invoke agent
    success, response = _invoke_enterprise_agent(
        request.agent_type,
        request.query
    )

    # Return standardized response
    return ThirdPartyResponse(
        success=success,
        response=response,
        session_id=session_id,
        agent_type=request.agent_type,
        source="new-platform",
        metadata={"platform_id": request.platform_id},
        error=None if success else response
    )
```

### Configuration Extension

**Adding New LLM Provider**:

```python
# In BaseAgent._init_llm()
elif provider == "new_provider":
    from langchain_new_provider import ChatNewProvider
    self._llm = ChatNewProvider(
        api_key=os.getenv("NEW_PROVIDER_API_KEY"),
        model=self.config.model_name or "default-model",
        temperature=self.config.temperature,
        max_tokens=self.config.max_tokens,
    )
```

---

## Deployment Architecture

### Container Architecture

```
┌─────────────────────────────────────────────────────┐
│              Docker Container                       │
│                                                     │
│  ┌───────────────────────────────────────────────┐ │
│  │         Application Layer                     │ │
│  │  - FastAPI Server (uvicorn)                   │ │
│  │  - Port 8000                                  │ │
│  └───────────────────────────────────────────────┘ │
│                       │                            │
│  ┌────────────────────▼──────────────────────────┐ │
│  │         Runtime Layer                         │ │
│  │  - Python 3.11 (slim)                         │ │
│  │  - Virtual Environment (/opt/venv)            │ │
│  └───────────────────────────────────────────────┘ │
│                       │                            │
│  ┌────────────────────▼──────────────────────────┐ │
│  │         Dependencies                          │ │
│  │  - langchain, langgraph                       │ │
│  │  - fastapi, uvicorn                           │ │
│  │  - Domain packages (pandas, tavily, etc.)     │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│  Security:                                          │
│  - Non-root user (appuser)                         │
│  - Minimal base image                              │
│  - No build tools in production image              │
└─────────────────────────────────────────────────────┘
```

### Multi-Stage Docker Build

```dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder
RUN pip install uv
WORKDIR /build
COPY pyproject.toml README.md ./
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install --no-cache .

# Stage 2: Production
FROM python:3.11-slim as production
RUN useradd --create-home --shell /bin/bash appuser
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY --chown=appuser:appuser ./app ./app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Deployment Patterns

#### Local Development
```bash
python -m uvicorn app.server:app --reload
```

#### Docker Deployment
```bash
docker compose up -d
```

#### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langchain-platform
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: app
        image: langchain-platform:latest
        ports:
        - containerPort: 8000
        envFrom:
        - secretRef:
            name: langchain-secrets
```

---

## Technology Stack

### Core Dependencies

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Runtime** | Python | 3.10+ | Programming language |
| **Web Framework** | FastAPI | 0.115+ | REST API server |
| **ASGI Server** | Uvicorn | 0.32+ | Production server |
| **LLM Framework** | LangChain | 0.3+ | Chain abstractions |
| **Agent Framework** | LangGraph | 0.2+ | State machine agents |
| **API Serving** | LangServe | 0.3+ | LangChain HTTP endpoints |
| **Tracing** | LangSmith | 0.1+ | Observability |
| **Validation** | Pydantic | 2.0+ | Data validation |
| **Configuration** | python-dotenv | 1.0+ | Environment management |

### LLM Providers

| Provider | Package | Models |
|----------|---------|--------|
| OpenAI | langchain-openai | gpt-4o, gpt-4o-mini |
| Anthropic | langchain-anthropic | claude-3-5-sonnet |

### Domain-Specific Dependencies

| Domain | Packages | Purpose |
|--------|----------|---------|
| **Data Analysis** | pandas, openpyxl | Excel/CSV processing |
| **Research** | tavily-python | Web search |
| **Documents** | pypdf, python-docx | Document processing |
| **RAG** | chromadb, faiss-cpu | Vector storage |
| **Multilingual** | langdetect | Language detection |

### Development Dependencies

| Tool | Purpose |
|------|---------|
| pytest | Testing framework |
| ruff | Linting and formatting |
| mypy | Type checking |
| httpx | Test client |

---

## Architectural Decisions Records (ADRs)

### ADR-001: LangGraph over Legacy Agents

**Context**: Need stateful, multi-step agent workflows with clear control flow.

**Decision**: Use LangGraph StateGraph instead of langchain.agents.

**Rationale**:
- Explicit state management with Pydantic
- Built-in checkpointing support
- Clear workflow visualization
- Better debugging capabilities
- Type-safe state transitions

**Consequences**:
- More verbose agent definitions
- Learning curve for LangGraph concepts
- Better long-term maintainability

### ADR-002: BaseAgent Abstract Pattern

**Context**: Need consistency across 7+ enterprise agents.

**Decision**: Implement abstract BaseAgent class with template methods.

**Rationale**:
- Code reuse (LLM init, compilation, invocation)
- Enforces consistent interface
- Simplifies agent creation
- Centralizes common functionality

**Consequences**:
- All agents must inherit from BaseAgent
- Less flexibility in agent architecture
- Easier to maintain and test

### ADR-003: Pydantic for State and Validation

**Context**: Need type-safe state and API validation.

**Decision**: Use Pydantic BaseModel for all state and request/response models.

**Rationale**:
- Automatic validation
- Type safety with IDE support
- Self-documenting schemas
- OpenAPI generation
- Runtime type checking

**Consequences**:
- Dependency on Pydantic 2.0+
- Learning curve for Pydantic features
- Excellent type safety and validation

### ADR-004: Provider Auto-Detection

**Context**: Support multiple LLM providers without code changes.

**Decision**: Implement auto-detection based on environment variables.

**Rationale**:
- Single codebase for all providers
- Easy switching between providers
- No configuration needed for simple cases
- Prefer OpenAI for broader compatibility

**Consequences**:
- Relies on environment variables
- May be unexpected for users
- Flexible deployment

### ADR-005: MemorySaver for Development

**Context**: Need checkpointing for conversation state.

**Decision**: Use MemorySaver (in-memory) for development, extensible to Redis/PostgreSQL.

**Rationale**:
- Simple development setup
- No external dependencies
- Easy to replace for production
- Clear upgrade path

**Consequences**:
- State lost on restart
- Not suitable for production at scale
- Simple development experience

---

## Blueprint for New Development

### Starting a New Feature

1. **Analyze Requirements**
   - Identify if it's a new agent, tool, or API endpoint
   - Determine state requirements
   - Identify external dependencies

2. **Choose Pattern**
   - New Agent → Use BaseAgent template
   - New Tool → Use @tool decorator pattern
   - New Endpoint → Follow FastAPI patterns
   - New Integration → Use webhook pattern

3. **Implementation Checklist**
   - [ ] Define Pydantic models (state/request/response)
   - [ ] Implement core logic
   - [ ] Add error handling
   - [ ] Add type hints
   - [ ] Write docstrings
   - [ ] Add to __init__.py exports
   - [ ] Register in server.py
   - [ ] Write tests
   - [ ] Update documentation

### Common Implementation Patterns

**Adding New Endpoint**:
```python
from fastapi import HTTPException
from pydantic import BaseModel

class MyRequest(BaseModel):
    field1: str
    field2: int = 0

class MyResponse(BaseModel):
    success: bool
    data: dict[str, Any]

@app.post("/api/my-endpoint")
async def my_endpoint(request: MyRequest) -> MyResponse:
    try:
        result = process_request(request)
        return MyResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(500, str(e))
```

**Adding New State Field**:
```python
class MyAgentState(BaseAgentState):
    # Inherit base fields
    # Add domain-specific fields
    custom_data: list[str] = Field(default_factory=list)
    processing_status: Literal["pending", "complete"] = "pending"
```

### Testing Strategy

```python
# tests/test_my_agent.py
import pytest
from fastapi.testclient import TestClient
from app.server import app

client = TestClient(app)

def test_my_agent_invoke():
    response = client.post(
        "/api/enterprise/my-agent/invoke",
        json={"query": "test query"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "response" in data
```

### Documentation Checklist

- [ ] Update this Architecture Blueprint
- [ ] Add to API reference
- [ ] Update SETUP.md if new dependencies
- [ ] Update DEPLOYMENT.md if new configuration
- [ ] Add examples to CLAUDE.md
- [ ] Update OpenAPI docstrings

---

## Architectural Health Metrics

### Code Quality Indicators

| Metric | Target | Current |
|--------|--------|---------|
| Type coverage | >90% | ~95% (all functions typed) |
| Docstring coverage | >80% | ~90% (all public APIs) |
| Test coverage | >80% | Variable by module |
| Cyclomatic complexity | <10 per function | Generally <10 |
| Module coupling | Low | Decoupled via BaseAgent |

### Architecture Principles Adherence

- **Single Responsibility**: Each agent has one primary function ✓
- **Open/Closed**: Extensible via inheritance ✓
- **Liskov Substitution**: All agents interchangeable ✓
- **Interface Segregation**: Minimal required methods ✓
- **Dependency Inversion**: Depends on abstractions (BaseAgent) ✓

---

## Future Architecture Considerations

### Scalability Enhancements

1. **Persistent Checkpointer**
   - Replace MemorySaver with Redis/PostgreSQL
   - Enables horizontal scaling
   - Conversation persistence across restarts

2. **Async Processing**
   - Background task queue (Celery/RQ)
   - Long-running agent executions
   - Webhook callbacks

3. **Caching Layer**
   - Redis for LLM response caching
   - Reduce costs and latency
   - Cache invalidation strategy

### Monitoring Enhancements

1. **Metrics Collection**
   - Prometheus metrics
   - Custom agent metrics
   - Performance dashboards

2. **Alerting**
   - Error rate thresholds
   - Response time SLAs
   - Availability monitoring

### Feature Additions

1. **Agent Composition**
   - Multi-agent workflows
   - Agent-to-agent communication
   - Hierarchical agent structures

2. **Advanced State Management**
   - State branching (A/B testing)
   - Time-travel debugging
   - State replay

---

## Conclusion

This architecture blueprint provides a comprehensive guide to the LangChain Enterprise Agents Platform. The platform demonstrates enterprise-grade software architecture principles:

- **Modularity**: Clear separation of concerns
- **Extensibility**: Easy to add new agents and tools
- **Maintainability**: Consistent patterns and abstractions
- **Observability**: Full tracing with LangSmith
- **Security**: API authentication and secrets management
- **Type Safety**: Pydantic models throughout

The architecture is designed for production use while maintaining flexibility for future enhancements. New developers can follow the patterns established here to extend the platform with confidence.

---

**Document Maintenance**

- Review quarterly for architectural drift
- Update after major feature additions
- Keep synchronized with implementation
- Solicit feedback from development team

**Last Reviewed**: December 19, 2025
**Next Review**: March 19, 2026
