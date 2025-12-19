# Architecture Documentation

> **Last Updated**: 2025-12-19
> **Version**: 2.0.0
> **Status**: Production-Ready

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagrams](#architecture-diagrams)
3. [Component Architecture](#component-architecture)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Design Patterns](#design-patterns)
7. [Scalability Considerations](#scalability-considerations)
8. [Security Architecture](#security-architecture)
9. [Integration Architecture](#integration-architecture)

---

## System Overview

### Purpose

The LangChain Enterprise Agents Platform is a **production-ready deployment platform** that serves AI agents as REST APIs. It provides:

- **7 Enterprise Agents** for IT operations, content creation, and development tasks
- **Multi-provider LLM Support** (OpenAI, Anthropic)
- **Human-in-the-Loop (HITL)** workflows for sensitive operations
- **Full Observability** via LangSmith tracing
- **3rd Party Integration** webhooks for Copilot Studio, Azure AI, AWS Lex

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| LangGraph over legacy agents | Stateful, multi-step workflows with checkpointing |
| Pydantic state models | Type-safe state management, validation |
| BaseAgent pattern | Consistent interface, code reuse |
| Provider auto-detection | Flexibility without code changes |
| API key middleware | Zero-trust security for external access |
| Webhook-based integration | Loose coupling with 3rd party platforms |

---

## Architecture Diagrams

### High-Level System Architecture

```
                                    ┌─────────────────────────────────────┐
                                    │         External Clients             │
                                    │  ┌─────────┐ ┌─────────┐ ┌────────┐ │
                                    │  │ Web UI  │ │ CLI     │ │ Mobile │ │
                                    │  └────┬────┘ └────┬────┘ └───┬────┘ │
                                    └───────┼──────────┼───────────┼──────┘
                                            │          │           │
                                            ▼          ▼           ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                              API Gateway / ngrok                               │
│                         (API Key Authentication)                               │
└───────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                           FastAPI Application                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                        Middleware Layer                                   │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │  │
│  │  │ CORS Handler │  │ API Key Auth │  │ Rate Limiter │                   │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                   │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                       │                                        │
│  ┌────────────────────────────────────┴────────────────────────────────────┐  │
│  │                         API Layer                                        │  │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐   │  │
│  │  │ LangServe Routes │  │ Enterprise APIs  │  │ Webhook Endpoints    │   │  │
│  │  │ /chat /rag       │  │ /api/enterprise  │  │ /api/webhooks/*      │   │  │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                       │                                        │
│  ┌────────────────────────────────────┴────────────────────────────────────┐  │
│  │                      Agent Layer (LangGraph)                             │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐           │  │
│  │  │ Research   │ │ Content    │ │ Document   │ │ Code       │           │  │
│  │  │ Agent      │ │ Agent      │ │ Agent      │ │ Assistant  │           │  │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘           │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐                          │  │
│  │  │ Data       │ │ RAG        │ │ HITL IT    │                          │  │
│  │  │ Analyst    │ │ Agent      │ │ Support    │                          │  │
│  │  └────────────┘ └────────────┘ └────────────┘                          │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                       │                                        │
│  ┌────────────────────────────────────┴────────────────────────────────────┐  │
│  │                    LangSmith Tracing Layer                               │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────┘
                                            │
              ┌─────────────────────────────┼─────────────────────────────┐
              │                             │                             │
              ▼                             ▼                             ▼
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│   LLM Providers     │      │   External Tools    │      │   3rd Party         │
│  ┌───────────────┐  │      │  ┌───────────────┐  │      │   Platforms         │
│  │ OpenAI        │  │      │  │ Tavily Search │  │      │  ┌───────────────┐  │
│  │ Anthropic     │  │      │  │ Web Fetch     │  │      │  │ Copilot Studio│  │
│  └───────────────┘  │      │  └───────────────┘  │      │  │ Azure AI      │  │
└─────────────────────┘      └─────────────────────┘      │  │ AWS Lex       │  │
                                                          │  └───────────────┘  │
                                                          └─────────────────────┘
```

### Agent Architecture Pattern

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           BaseAgent (Abstract)                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Properties                                                            │  │
│  │  - config: AgentConfig                                               │  │
│  │  - llm: BaseChatModel                                                │  │
│  │  - tools: list[BaseTool]                                             │  │
│  │  - graph: CompiledGraph                                              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Abstract Methods                                                      │  │
│  │  - _build_graph() -> StateGraph                                      │  │
│  │  - _get_system_prompt() -> str                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Concrete Methods                                                      │  │
│  │  - invoke(message, session_id, **kwargs) -> dict                     │  │
│  │  - astream(message, session_id, **kwargs) -> AsyncIterator           │  │
│  │  - get_state(session_id) -> dict                                     │  │
│  │  - get_last_response(result) -> str                                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ extends
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
         ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
         │ Research     │ │ Content      │ │ Document     │
         │ Agent        │ │ Agent (HITL) │ │ Agent        │
         └──────────────┘ └──────────────┘ └──────────────┘
```

### LangGraph State Flow

```
                            ┌─────────────────┐
                            │      START      │
                            └────────┬────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │   agent_node    │◄─────────────┐
                            │  (LLM + Tools)  │              │
                            └────────┬────────┘              │
                                     │                       │
                                     ▼                       │
                            ┌─────────────────┐              │
                            │ should_continue │              │
                            │   (Router)      │              │
                            └────────┬────────┘              │
                                     │                       │
                     ┌───────────────┴───────────────┐       │
                     │                               │       │
              tool_calls?                      no tool_calls │
                     │                               │       │
                     ▼                               ▼       │
            ┌─────────────────┐             ┌──────────────┐ │
            │   tools_node    │             │     END      │ │
            │ (Execute Tools) │             └──────────────┘ │
            └────────┬────────┘                              │
                     │                                       │
                     └───────────────────────────────────────┘
```

---

## Component Architecture

### Core Components

| Component | Location | Responsibility |
|-----------|----------|----------------|
| **server.py** | `app/server.py` | FastAPI app, routing, middleware |
| **BaseAgent** | `app/agents/base/agent_base.py` | Agent abstraction, LLM init |
| **Tools** | `app/agents/base/tools.py` | Shared tool utilities |
| **Enterprise Agents** | `app/agents/*/` | Domain-specific agents |
| **Evaluators** | `app/agents/evals/` | Quality evaluation framework |

### Enterprise Agents

| Agent | Module | Features |
|-------|--------|----------|
| **ResearchAgent** | `agents/research/` | Web search, source synthesis |
| **ContentAgent** | `agents/content/` | Multi-platform content, HITL approval |
| **DataAnalystAgent** | `agents/data_analyst/` | Excel/CSV analysis, visualization |
| **DocumentAgent** | `agents/documents/` | SOP, WLI, Policy generation |
| **MultilingualRAGAgent** | `agents/rag/` | 10+ languages, document Q&A |
| **HITLSupportAgent** | `agents/it_support/` | IT tickets, approval gates |
| **CodeAssistantAgent** | `agents/code_assistant/` | Code review, modernization |

---

## Data Flow

### Request Flow (Standard)

```
1. HTTP Request → FastAPI Router
2. API Key Middleware validates credentials
3. Router dispatches to endpoint handler
4. Handler invokes Agent.invoke(message)
5. LangGraph executes state machine:
   a. agent_node: LLM generates response
   b. should_continue: Check for tool calls
   c. tools_node: Execute tools if needed
   d. Loop until no more tool calls
6. Response extracted from state
7. HTTP Response returned
```

### State Management

```python
class BaseAgentState(BaseModel):
    """Pydantic model for type-safe state."""

    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = {}
```

### Checkpointing

- **MemorySaver**: In-memory checkpoint for development
- **Session ID**: Thread identifier for conversation continuity
- **State Persistence**: Full message history preserved

---

## Technology Stack

### Runtime

| Layer | Technology | Version |
|-------|------------|---------|
| Language | Python | 3.10+ |
| Web Framework | FastAPI | 0.115+ |
| LLM Framework | LangChain | 1.2+ |
| Agent Framework | LangGraph | 1.0+ |
| API Serving | LangServe | 0.3+ |
| Tracing | LangSmith | 0.5+ |

### LLM Providers

| Provider | Models | Use Case |
|----------|--------|----------|
| OpenAI | gpt-4o-mini, gpt-4o | Primary provider |
| Anthropic | claude-3-5-sonnet | Alternative provider |

### Infrastructure

| Component | Technology |
|-----------|------------|
| Containerization | Docker |
| Orchestration | Docker Compose |
| Tunnel/Exposure | ngrok |
| CI/CD | GitHub Actions |

---

## Design Patterns

### 1. Abstract Factory (Agent Creation)

```python
class BaseAgent(ABC):
    @abstractmethod
    def _build_graph(self) -> StateGraph:
        """Factory method for graph creation."""
        pass
```

### 2. Template Method (Agent Invocation)

```python
def invoke(self, message: str, **kwargs) -> dict:
    if self._compiled_graph is None:
        self.compile()  # Template step 1
    input_state = self._build_input(message, **kwargs)  # Step 2
    result = self._compiled_graph.invoke(input_state)  # Step 3
    return result
```

### 3. Strategy Pattern (LLM Provider)

```python
def _init_llm(self) -> None:
    if provider == "openai":
        self._llm = ChatOpenAI(...)
    elif provider == "anthropic":
        self._llm = ChatAnthropic(...)
```

### 4. State Pattern (LangGraph)

- Graph nodes represent states
- Edges represent transitions
- Conditional edges for branching logic

### 5. Decorator Pattern (Tracing)

```python
@traceable(name="agent_invoke")
def invoke(self, message: str) -> dict:
    ...
```

---

## Scalability Considerations

### Horizontal Scaling

- **Stateless Design**: No server-side session state
- **Checkpointing**: MemorySaver can be replaced with Redis/PostgreSQL
- **Load Balancing**: Standard HTTP load balancing supported

### Performance Optimizations

| Optimization | Implementation |
|--------------|----------------|
| Lazy Loading | Agents load only when API keys available |
| Connection Pooling | LLM clients manage connections |
| Async Support | `astream()` for non-blocking operations |
| Caching | Optional Redis integration |

### Resource Limits

```yaml
# docker-compose.yml
deploy:
  resources:
    limits:
      cpus: "2"
      memory: 2G
    reservations:
      cpus: "0.5"
      memory: 512M
```

---

## Security Architecture

### Authentication

```
┌─────────────────────────────────────────────────────────────────┐
│                    API Key Middleware                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 1. Check API_KEY_ENABLED flag                             │  │
│  │ 2. Skip public paths (/health, /docs)                     │  │
│  │ 3. Validate X-API-Key header against API_KEY env var      │  │
│  │ 4. Return 401 if invalid/missing                          │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Secrets Management

- Environment variables for all secrets
- `.env` files excluded from git
- No hardcoded credentials

### Input Validation

- Pydantic models for all request/response
- Type checking at API boundary
- Sanitized outputs in tools

---

## Integration Architecture

### Webhook Pattern

```
┌──────────────────┐          ┌──────────────────┐
│   3rd Party      │          │   LangChain      │
│   Platform       │          │   Platform       │
│                  │          │                  │
│  ┌────────────┐  │  HTTP    │  ┌────────────┐  │
│  │ Copilot    │──┼─────────►│  │ /webhooks  │  │
│  │ Studio     │  │  POST    │  │ /copilot-  │  │
│  └────────────┘  │          │  │  studio    │  │
│                  │          │  └────────────┘  │
│  ┌────────────┐  │          │        │        │
│  │ Azure AI   │──┼─────────►│        ▼        │
│  └────────────┘  │          │  ┌────────────┐  │
│                  │          │  │ Enterprise │  │
│  ┌────────────┐  │          │  │   Agent    │  │
│  │ AWS Lex    │──┼─────────►│  └────────────┘  │
│  └────────────┘  │          │        │        │
│                  │          │        ▼        │
│                  │◄─────────┼──── Response    │
└──────────────────┘  JSON    └──────────────────┘
```

### Request/Response Format

```json
// Webhook Request
{
  "query": "User question",
  "agent_type": "research",
  "session_id": "optional-session-id",
  "metadata": {}
}

// Webhook Response
{
  "success": true,
  "response": "Agent answer",
  "session_id": "generated-or-provided",
  "agent_type": "research",
  "source": "copilot-studio",
  "error": null
}
```

---

## Architecture Decision Records (ADRs)

### ADR-001: LangGraph over Legacy Agents

**Context**: Need stateful, multi-step agent workflows.

**Decision**: Use LangGraph StateGraph instead of langchain.agents.

**Rationale**:
- Built-in checkpointing support
- Type-safe state with Pydantic
- Explicit control flow
- Better debugging

### ADR-002: Provider Auto-Detection

**Context**: Support multiple LLM providers without code changes.

**Decision**: Auto-detect available provider from environment.

**Rationale**:
- Prefer OpenAI for broader compatibility
- Fall back to Anthropic if OpenAI unavailable
- Single codebase for all providers

### ADR-003: Webhook-Based Integration

**Context**: Need integration with Copilot Studio, Azure AI, AWS Lex.

**Decision**: Implement webhook endpoints with standardized request/response.

**Rationale**:
- Loose coupling with external platforms
- Platform-agnostic design
- Easy to add new integrations

---

## Related Documentation

- [DEPLOYMENT.md](./DEPLOYMENT.md) - Deployment guide
- [SECURITY.md](./SECURITY.md) - Security guidelines
- [OPERATIONS.md](./OPERATIONS.md) - Operations runbook
- [API Reference](./api/README.md) - API documentation
