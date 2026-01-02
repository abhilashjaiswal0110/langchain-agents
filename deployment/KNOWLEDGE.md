# LangChain Platform - Knowledge Base

> **Purpose**: This document serves as the authoritative knowledge source for AI agents working on this repository. It contains architectural decisions, implementation patterns, and guidelines that must be followed when making changes or enhancements.

**Last Updated**: 2026-01-02 (v3.9 - Teams & Slack Integrations)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Directory Structure](#directory-structure)
4. [Key Components](#key-components)
5. [IT Support Agents](#it-support-agents)
6. [Configuration](#configuration)
7. [API Endpoints](#api-endpoints)
8. [Web UI & CLI](#web-ui--cli)
9. [External Integrations](#external-integrations)
10. [Enterprise Agents](#enterprise-agents)
11. [Governance Framework](#governance-framework)
12. [MCP Integration](#mcp-integration)
13. [DeepSearch Research](#deepsearch-research)
14. [Dependencies](#dependencies)
15. [Development Patterns](#development-patterns)
16. [Testing Strategy](#testing-strategy)
17. [Deployment](#deployment)
18. [Common Tasks](#common-tasks)
19. [Troubleshooting](#troubleshooting)
20. [Change Log](#change-log)

---

## Project Overview

### What is this project?

A **production-ready deployment platform** that serves LangChain chains and LangGraph agents as REST APIs. It provides:

- FastAPI server with LangServe integration
- Multiple AI endpoints (chat, RAG, agents)
- **IT Support Agents** (IT Helpdesk, ServiceNow) with conversation memory
- **Web UI and CLI** for demos and testing
- **External Integration Webhooks** for Copilot Studio, Azure AI, AWS AI
- Document RAG with PDF/Word/TXT support
- LangSmith tracing for observability
- Docker containerization for deployment
- Kubernetes-ready health checks

### Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Web Framework | FastAPI | >=0.115.0 |
| LLM Framework | LangChain | >=0.3.0 |
| Agent Framework | LangGraph | >=0.2.0 |
| API Serving | LangServe | >=0.3.0 |
| Tracing | LangSmith | >=0.1.0 |
| Primary LLM | OpenAI GPT-4o-mini | - |
| Alternative LLM | Anthropic Claude | - |
| Python | Python | >=3.10 |

### Key Design Decisions

1. **LangGraph over legacy agents**: Uses `langgraph.prebuilt.create_react_agent` instead of deprecated `langchain.agents.create_tool_calling_agent`
2. **Lazy loading**: Chains load only when API keys are available
3. **Provider agnostic**: Supports both OpenAI and Anthropic
4. **Tracing first**: LangSmith tracing enabled by default when configured
5. **Session-based conversations**: IT Support agents use MemorySaver for conversation continuity
6. **Webhook-based integration**: External platforms integrate via standardized webhook API
7. **Multi-agent architecture**: ConversationManager handles agent selection and session routing

---

## Architecture

### High-Level Architecture

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                           FastAPI Application                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                         API Layer                                         │  │
│  │  ┌────────────────────┐  ┌────────────────────┐  ┌───────────────────┐  │  │
│  │  │   LangServe Routes │  │ Conversation API   │  │ Webhook API       │  │  │
│  │  │  /chat /rag /agent │  │ /api/conversation  │  │ /api/webhook/chat │  │  │
│  │  └────────────────────┘  └────────────────────┘  └───────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                       │                                         │
│  ┌────────────────────────────────────┴─────────────────────────────────────┐ │
│  │                         Agent & Chain Layer                               │ │
│  │  ┌──────────────────────────────┐  ┌──────────────────────────────────┐  │ │
│  │  │       LangChain Chains       │  │        IT Support Agents         │  │ │
│  │  │  chat_chain │ rag_chain      │  │  ┌─────────────┬─────────────┐   │  │ │
│  │  │  agent_executor │ doc_rag    │  │  │ IT Helpdesk │ ServiceNow  │   │  │ │
│  │  └──────────────────────────────┘  │  │   Agent     │    Agent    │   │  │ │
│  │  ┌──────────────────────────────┐  │  └─────────────┴─────────────┘   │  │ │
│  │  │     LangGraph Agents         │  │  ┌─────────────────────────────┐ │  │ │
│  │  │  langgraph_agent             │  │  │  Conversation Manager       │ │  │ │
│  │  │  (OpenAI/Anthropic)          │  │  │  (Session + Memory)         │ │  │ │
│  │  └──────────────────────────────┘  │  └─────────────────────────────┘ │  │ │
│  │                                    └──────────────────────────────────┘  │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
│                                       │                                         │
│  ┌────────────────────────────────────┴─────────────────────────────────────┐ │
│  │                       LangSmith Tracing                                   │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────┘
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           │                            │                            │
           ▼                            ▼                            ▼
   ┌───────────────┐           ┌───────────────┐           ┌───────────────┐
   │   Web UI      │           │   CLI Chat    │           │   External    │
   │  /chat        │           │  cli_chat.py  │           │  Integrations │
   │  (Browser)    │           │  (Terminal)   │           │  (Webhooks)   │
   └───────────────┘           └───────────────┘           └───────────────┘
```

### Request Flow

**Standard Chains (via LangServe)**:
1. Request arrives at FastAPI endpoint
2. LangServe deserializes input
3. Chain/Agent processes request
4. LangSmith captures trace (if enabled)
5. Response returned to client

**IT Support Agents (via Conversation API)**:
1. Client calls `/api/conversation/start` with agent_type
2. ConversationManager creates session with MemorySaver
3. Agent processes messages via `/api/conversation/chat`
4. Conversation history persisted in session
5. LangSmith traces tool calls and responses

**External Integrations (via Webhook)**:
1. External platform sends `conversation.start` event
2. Platform receives session_id and welcome message
3. Subsequent `conversation.message` events with session_id
4. Platform calls `conversation.end` when done

---

## Directory Structure

```
deployment/
├── app/                          # Application source code
│   ├── __init__.py              # Package marker
│   ├── server.py                # FastAPI application entry point
│   ├── chains/                  # Chain implementations
│   │   ├── __init__.py          # Exports all chains
│   │   ├── chat.py              # Simple chat chain (OpenAI)
│   │   ├── rag.py               # RAG chain with vector store
│   │   ├── agent.py             # LangGraph React agent
│   │   ├── langgraph_agent.py   # LangGraph agent with custom tools
│   │   └── doc_rag.py           # Document RAG with file upload
│   ├── agents/                  # IT Support agents
│   │   ├── __init__.py          # Exports all agents
│   │   ├── it_helpdesk.py       # IT Helpdesk Agent
│   │   ├── servicenow_agent.py  # ServiceNow ITSM Agent
│   │   ├── conversation_manager.py  # Session management
│   │   └── research/            # DeepSearch research components (NEW)
│   │       ├── __init__.py      # Module exports
│   │       ├── research_agent.py    # Basic research agent
│   │       ├── planner.py       # Query decomposition
│   │       ├── source_manager.py    # Citation tracking
│   │       ├── search_providers.py  # Multi-provider search
│   │       └── deep_search_agent.py # Enhanced research agent
│   ├── governance/              # Governance framework
│   │   ├── __init__.py          # Module exports
│   │   ├── rbac.py              # Role-based access control
│   │   ├── audit_logger.py      # Compliance audit logging
│   │   ├── rate_limiter.py      # Token bucket rate limiting
│   │   ├── approval_workflow.py # Multi-level approval workflows
│   │   └── middleware.py        # FastAPI middleware integration
│   ├── mcp/                     # MCP integration
│   │   ├── __init__.py          # Module exports
│   │   ├── server.py            # FastMCP server with tools
│   │   ├── gateway.py           # Access control gateway
│   │   ├── servicenow_client.py # Real ServiceNow REST API
│   │   └── tools/               # Tool implementations
│   ├── memory/                  # Session persistence
│   │   ├── __init__.py          # Module exports
│   │   ├── base.py              # Base classes and types
│   │   ├── memory_store.py      # In-memory session store
│   │   ├── redis_store.py       # Redis session store
│   │   ├── sqlite_store.py      # SQLite session store
│   │   ├── conversation_memory.py # LangChain integration
│   │   └── config.py            # Configuration and factories
│   ├── integrations/            # External integrations (NEW)
│   │   ├── __init__.py          # Module exports
│   │   ├── teams_webhook.py     # Microsoft Teams webhook
│   │   ├── slack_webhook.py     # Slack webhook
│   │   └── routes.py            # FastAPI routes
│   └── static/                  # Static web files
│       └── chat.html            # Web UI for demos
├── tests/                       # Test suite
│   ├── __init__.py
│   └── test_server.py           # Server endpoint tests
├── cli_chat.py                  # CLI chat interface (NEW)
├── .env                         # Environment variables (NOT committed)
├── .env.example                 # Environment template (committed)
├── .gitignore                   # Git exclusions
├── Dockerfile                   # Production Docker image
├── docker-compose.yml           # Docker Compose configuration
├── Makefile                     # Development commands
├── pyproject.toml               # Python dependencies
├── README.md                    # User documentation
└── KNOWLEDGE.md                 # This file - AI agent knowledge base
```

---

## Key Components

### 1. Server (`app/server.py`)

**Purpose**: Main FastAPI application with LangServe routes

**Key Functions**:
- `setup_langsmith_tracing()`: Configures LangSmith if enabled
- `load_chains()`: Loads LangChain chains (requires OPENAI_API_KEY)
- `load_langgraph_agent()`: Loads LangGraph agent (OpenAI or Anthropic)
- `setup_langchain_routes()`: Registers LangServe endpoints

**Global State**:
```python
chains_loaded: bool        # True if LangChain chains loaded
langgraph_loaded: bool     # True if LangGraph agent loaded
tracing_enabled: bool      # True if LangSmith tracing active
```

### 2. Chat Chain (`app/chains/chat.py`)

**Purpose**: Simple conversational AI

**Implementation**:
```python
prompt | llm | StrOutputParser()
```

**Input Schema**: `{"input": "user message"}`

### 3. RAG Chain (`app/chains/rag.py`)

**Purpose**: Retrieval-Augmented Generation

**Components**:
- `InMemoryVectorStore` with OpenAI embeddings
- Pre-loaded sample documents about LangChain
- Retriever with k=3

**Input Schema**: `{"input": "question"}`

### 4. Agent (`app/chains/agent.py`)

**Purpose**: LangGraph React agent with tools

**Tools Available**:
- `get_current_time()`: Returns current datetime
- `calculate(expression)`: Evaluates math expressions
- `search_knowledge_base(query)`: Searches simulated knowledge base

**Implementation**: Uses `langgraph.prebuilt.create_react_agent`

### 5. LangGraph Agent (`app/chains/langgraph_agent.py`)

**Purpose**: Advanced LangGraph agent with StateGraph

**Features**:
- Supports OpenAI and Anthropic models
- Auto-selects available provider
- Custom tools: web_search, calculator, get_system_info
- Async support via `ainvoke`

**Model Selection Priority**:
1. Anthropic (if ANTHROPIC_API_KEY set)
2. OpenAI (if OPENAI_API_KEY set)

### 6. Document RAG (`app/chains/doc_rag.py`)

**Purpose**: Document-based RAG with file upload support

**Supported File Types**:
- PDF (`.pdf`)
- Word Documents (`.docx`, `.doc`)
- Plain Text (`.txt`)

**Components**:
- `DocumentRAGChain` class with upload and query methods
- FAISS vector store for embeddings
- `RecursiveCharacterTextSplitter` for chunking
- OpenAI embeddings and LLM

**Key Methods**:
- `load_from_bytes(content, filename)`: Load document from uploaded file
- `query(question, k=4)`: Query loaded documents
- `get_document_info()`: Get stats about loaded documents
- `clear_documents()`: Clear all documents from memory

**Configuration**:
```python
chunk_size: int = 1000      # Characters per chunk
chunk_overlap: int = 200    # Overlap between chunks
model: str = "gpt-4o-mini"  # LLM model
temperature: float = 0      # Response temperature
```

**LangSmith Tracing**:
- `@traceable(name="load_document", tags=["doc-rag", "ingestion"])`
- `@traceable(name="query_document", tags=["doc-rag", "query"])`

### 7. Memory Module (`app/memory/`)

**Purpose**: Persistent session storage with multiple backends for conversation history.

**Storage Backends**:
| Backend | Use Case | Features |
|---------|----------|----------|
| `InMemorySessionStore` | Development/testing | Fast, non-persistent, thread-safe |
| `RedisSessionStore` | Production/distributed | Scalable, TTL support, indices |
| `SQLiteSessionStore` | Single-instance production | Persistent, local storage, ACID |

**Core Components**:

1. **Base Types** (`base.py`):
   - `Message`: Role, content, timestamp, metadata
   - `Session`: ID, messages, context, metadata, expiration
   - `SessionMetadata`: User ID, agent type, tags, custom data
   - `BaseSessionStore`: Abstract interface for all backends

2. **Conversation Memory** (`conversation_memory.py`):
   - `ConversationMemory`: LangChain-integrated memory management
   - `ConversationSummary`: Session summary with message counts
   - `get_langchain_messages()`: Convert to LangChain message format
   - `get_chat_history_string()`: Formatted history for prompts

3. **Configuration** (`config.py`):
   - `MemoryBackend`: Enum for backend selection
   - `MemoryConfig`: Configuration from environment
   - `get_session_store()`: Factory for session stores
   - `get_checkpointer()`: Factory for LangGraph checkpointers

**Environment Variables**:
```bash
MEMORY_BACKEND=memory|redis|sqlite   # Storage backend (default: memory)
REDIS_URL=redis://localhost:6379     # Redis connection URL
SQLITE_PATH=data/sessions.db         # SQLite database path
SESSION_TTL_HOURS=24                 # Session TTL in hours
MAX_SESSIONS=10000                   # Max sessions (memory backend)
SESSION_KEY_PREFIX=session:          # Redis key prefix
```

**Usage Example**:
```python
from app.memory import (
    get_session_store,
    ConversationMemory,
    MemoryConfig,
)

# Using default configuration
store = get_session_store()

# Create conversation memory
memory = ConversationMemory(session_store=store)
session_id = memory.create_session("helpdesk", user_id="user-123")

# Add messages
memory.add_exchange(session_id, "Hello", "How can I help?")

# Get LangChain messages
lc_messages = memory.get_langchain_messages(session_id)
```

---

## IT Support Agents

### Overview

The IT Support Agents provide a demo-ready, production-capable multi-agent system for IT helpdesk and ServiceNow ITSM operations. They feature:

- **Conversation Memory**: LangGraph MemorySaver for session continuity
- **Tool Integration**: Simulated IT operations (can be connected to real systems)
- **Multi-Agent Support**: Switch between agents within a session
- **LangSmith Tracing**: Full observability of tool calls and responses

### 1. IT Helpdesk Agent (`app/agents/it_helpdesk.py`)

**Purpose**: General IT support agent for common helpdesk tasks

**Tools Available**:
| Tool | Description |
|------|-------------|
| `search_knowledge_base` | Search IT knowledge base for solutions |
| `create_support_ticket` | Create a new support ticket |
| `check_ticket_status` | Check status of existing ticket |
| `check_system_status` | Check status of IT systems |
| `initiate_password_reset` | Start password reset process |
| `request_software` | Request software installation |
| `escalate_to_human` | Escalate issue to human agent |

**Architecture**:
```python
class ITHelpdeskAgent:
    def __init__(self, model_provider: Literal["openai", "anthropic", "auto"] = "auto"):
        self.tools = [search_knowledge_base, create_support_ticket, ...]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.memory = MemorySaver()  # Conversation persistence
        self.graph = self._build_graph()  # LangGraph StateGraph
```

**LangGraph Flow**:
```
START → agent_node → [should_continue?]
                    ├── "continue" → tools_node → agent_node
                    └── "end" → END
```

### 2. ServiceNow Agent (`app/agents/servicenow_agent.py`)

**Purpose**: ServiceNow ITSM operations agent

**Tools Available**:
| Tool | Description |
|------|-------------|
| `search_incidents` | Search incidents by query |
| `get_incident_details` | Get detailed incident info |
| `create_incident` | Create new incident |
| `update_incident` | Update existing incident |
| `get_change_requests` | Get upcoming change requests |
| `search_cmdb` | Search Configuration Management DB |
| `get_my_tickets` | Get user's assigned tickets |

**Usage Example**:
```python
from app.agents.servicenow_agent import ServiceNowAgent

agent = ServiceNowAgent(model_provider="auto")
result = agent.chat(
    message="Show me high priority incidents for network",
    thread_id="session-123"
)
print(result["response"])
```

### 3. Conversation Manager (`app/agents/conversation_manager.py`)

**Purpose**: Unified session and conversation management across agents

**Key Features**:
- Session creation and tracking
- Agent selection and switching
- History management
- Command handling (`/help`, `/switch`, `/status`, `/history`, `/clear`)

**Class Structure**:
```python
class SessionStore:
    """In-memory session storage (use Redis/DB for production)."""
    def create_session(agent_type, user_id, metadata) -> str
    def get_session(session_id) -> dict | None
    def update_session(session_id, user_message, assistant_message)
    def get_history(session_id, limit) -> list[dict]

class ConversationManager:
    """Unified conversation manager for all IT Support agents."""
    AVAILABLE_AGENTS = {
        "it_helpdesk": "IT Helpdesk Agent - General IT support...",
        "servicenow": "ServiceNow Agent - Ticket management...",
    }

    def start_conversation(agent_type, user_id, metadata) -> dict
    def chat(session_id, message) -> dict
    async def achat(session_id, message) -> dict  # Async version
    def _handle_command(session_id, command) -> dict
```

### Adding a New IT Support Agent

1. Create agent file in `app/agents/`:
```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

@tool
def your_tool(param: str) -> str:
    """Tool description."""
    return result

class YourAgent:
    def __init__(self, model_provider="auto"):
        self.tools = [your_tool]
        self.memory = MemorySaver()
        self.graph = create_react_agent(
            self.llm,
            tools=self.tools,
            checkpointer=self.memory,
        )

    def chat(self, message: str, thread_id: str) -> dict:
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke({"messages": [...]}, config)
        return {"response": result["messages"][-1].content}
```

2. Register in `conversation_manager.py`:
```python
AVAILABLE_AGENTS = {
    ...,
    "your_agent": "Your Agent - Description",
}

def _load_agents(self):
    ...
    from app.agents.your_agent import YourAgent
    self._agents["your_agent"] = YourAgent()
```

3. Export in `app/agents/__init__.py`:
```python
from app.agents.your_agent import YourAgent
__all__ = [..., "YourAgent"]
```

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes* | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | No | - | Anthropic API key |
| `LANGCHAIN_TRACING_V2` | No | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | - | LangSmith API key |
| `LANGCHAIN_PROJECT` | No | `langchain-platform` | LangSmith project name |
| `LANGCHAIN_ENDPOINT` | No | `https://api.smith.langchain.com` | LangSmith endpoint |
| `TAVILY_API_KEY` | No | - | Tavily search API key |
| `PORT` | No | `8000` | Server port |

*At least one LLM provider key required

### Loading Order

1. `.env` file loaded via `python-dotenv`
2. Environment variables override `.env`
3. Tracing configured before chains load
4. Chains load based on available API keys

---

## API Endpoints

### Health & Status

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Redirects to `/docs` |
| `/docs` | GET | Swagger UI documentation |
| `/health` | GET | Health check with component status |
| `/ready` | GET | Kubernetes readiness probe |

### LangChain Endpoints (via LangServe)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat/invoke` | POST | Chat completion |
| `/chat/stream` | POST | Streaming chat |
| `/chat/batch` | POST | Batch chat requests |
| `/rag/invoke` | POST | RAG query |
| `/rag/stream` | POST | Streaming RAG |
| `/agent/invoke` | POST | Agent execution |
| `/agent/stream` | POST | Streaming agent |

### LangGraph Endpoint (Custom)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/langgraph/invoke` | POST | LangGraph agent |

### Document RAG Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/doc-rag/upload` | POST | Upload document (PDF, Word, TXT) |
| `/doc-rag/query` | POST | Query uploaded documents |
| `/doc-rag/info` | GET | Get loaded document info |
| `/doc-rag/clear` | DELETE | Clear all documents |

### Conversation API Endpoints (IT Support)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/conversation/start` | POST | Start new conversation with agent |
| `/api/conversation/chat` | POST | Send message in conversation |
| `/api/conversation/{session_id}` | GET | Get session information |
| `/api/conversation/{session_id}` | DELETE | End conversation |
| `/api/agents` | GET | List available agents |

### Webhook API (External Integrations)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/webhook/chat` | POST | Webhook for external platforms |

### User Interface Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | GET | Web UI for browser demos |

### Request/Response Formats

**LangServe Format**:
```json
// Request
{"input": {"input": "your message"}}

// Response
{"output": "response", "metadata": {"run_id": "..."}}
```

**LangGraph Format**:
```json
// Request
{"input": "your message"}

// Response
{"output": "response"}
```

**Document RAG Format**:
```json
// Upload: POST /doc-rag/upload with multipart/form-data (file field)
// Response
{"status": "success", "file_name": "...", "chunks_created": 2}

// Query Request
{"question": "What is this document about?", "k": 4}

// Query Response
{
  "status": "success",
  "answer": "The document is about...",
  "sources": [{"source": "file.pdf", "chunk_index": 0, "preview": "..."}],
  "num_sources": 4
}
```

**Conversation API Format**:
```json
// Start Conversation Request
{
  "agent_type": "it_helpdesk",  // or "servicenow"
  "user_id": "user-123",
  "metadata": {"source": "web"}
}

// Start Conversation Response
{
  "session_id": "uuid-...",
  "agent_type": "it_helpdesk",
  "welcome_message": "Welcome to IT Support!...",
  "available_commands": ["/help", "/switch", ...]
}

// Chat Request
{
  "session_id": "uuid-...",
  "message": "I need to reset my password"
}

// Chat Response
{
  "session_id": "uuid-...",
  "response": "I can help you with that...",
  "agent_type": "it_helpdesk",
  "tool_calls": []
}
```

**Webhook Format**:
```json
// conversation.start event
{
  "event_type": "conversation.start",
  "agent_type": "it_helpdesk",
  "user_id": "external-user-123",
  "metadata": {"source": "copilot-studio", "channel": "teams"}
}

// conversation.message event
{
  "event_type": "conversation.message",
  "session_id": "uuid-...",
  "message": "Help with VPN"
}

// conversation.end event
{
  "event_type": "conversation.end",
  "session_id": "uuid-..."
}

// Response format
{
  "success": true,
  "message": "Agent response...",
  "session_id": "uuid-...",
  "data": {"tool_calls": []}
}
```

---

## Web UI & CLI

### Web UI (`app/static/chat.html`)

A browser-based chat interface for stakeholder demos:

**Access**: `http://localhost:8000/chat`

**Features**:
- Agent selection dropdown (IT Helpdesk, ServiceNow)
- Quick action buttons for common requests
- Real-time chat with conversation history
- System status display
- Session information panel
- Mobile-responsive design

**Usage**:
1. Open `http://localhost:8000/chat` in browser
2. Select an agent (IT Helpdesk or ServiceNow)
3. Click "Start Chat" to begin session
4. Type messages or use quick actions
5. Use commands: `/help`, `/status`, `/switch`, `/clear`

### CLI Chat (`cli_chat.py`)

A terminal-based chat interface using Rich library:

**Usage**:
```bash
cd deployment
python cli_chat.py
```

**Features**:
- Rich terminal UI with colors and panels
- Agent selection menu
- Command history
- System status checks
- Session management

**Commands**:
| Command | Description |
|---------|-------------|
| `/help` | Show help message |
| `/status` | Check system status |
| `/switch <agent>` | Switch to different agent |
| `/history` | View conversation history |
| `/clear` | Clear conversation |
| `/quit` or `/exit` | Exit chat |

---

## External Integrations

### Overview

The platform provides webhook-based integration for external AI platforms:

- **Microsoft Copilot Studio**: Via HTTP actions
- **Azure AI Agent**: Via webhook connectors
- **AWS AI Agent**: Via Lambda integration
- **Custom Platforms**: Any HTTP-capable system

### Webhook Integration Pattern

```
External Platform         LangChain Platform
      │                          │
      │  POST /api/webhook/chat  │
      │  event_type: "start"     │
      │ ─────────────────────────>
      │                          │
      │  session_id, welcome_msg │
      │ <─────────────────────────
      │                          │
      │  POST /api/webhook/chat  │
      │  event_type: "message"   │
      │  session_id, message     │
      │ ─────────────────────────>
      │                          │
      │  response, tool_calls    │
      │ <─────────────────────────
      │                          │
      │  (repeat as needed)      │
      │                          │
      │  POST /api/webhook/chat  │
      │  event_type: "end"       │
      │ ─────────────────────────>
      │                          │
```

### Integration Examples

**Microsoft Copilot Studio**:
1. Create HTTP action in Copilot Studio
2. Configure POST to `/api/webhook/chat`
3. Map Copilot variables to webhook payload
4. Parse response back to Copilot

**Azure Logic Apps / Power Automate**:
1. Add HTTP connector
2. Configure webhook endpoint
3. Use JSON expressions for payload
4. Extract response fields

**AWS Lambda**:
```python
import requests

def lambda_handler(event, context):
    response = requests.post(
        "https://your-platform/api/webhook/chat",
        json={
            "event_type": "conversation.message",
            "session_id": event["session_id"],
            "message": event["user_message"],
        }
    )
    return response.json()
```

---

## Enterprise Agents

### Overview

In addition to the IT Support Agents, the platform includes 7 enterprise-grade agents for various business use cases:

| Agent | Purpose | HITL | Key Features |
|-------|---------|------|--------------|
| ResearchAgent | Web research | No | Tavily search, source citations |
| ContentCreationAgent | Marketing content | Yes | Draft approval, revision cycles |
| DataAnalystAgent | Data analysis | No | CSV/Excel processing, visualizations |
| DocumentGenerationAgent | Doc creation | No | SOP, WLI, Policy templates |
| MultilingualRAGAgent | Multi-language Q&A | No | 10+ languages, translation |
| HITLITSupportAgent | IT tickets | Yes | Priority routing, approvals |
| CodeAssistantAgent | Code help | No | Review, generation, debugging |

### BaseAgent Pattern

All enterprise agents extend `BaseAgent`:

```python
from app.agents.base import BaseAgent, AgentState

class MyAgent(BaseAgent):
    name = "my_agent"
    description = "Agent description"

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("process", self._process_node)
        workflow.set_entry_point("process")
        workflow.add_edge("process", END)
        return workflow.compile()

    def _process_node(self, state: AgentState) -> dict:
        # Process logic
        return {"messages": [...], "metadata": {...}}
```

### Evaluation Framework

**3 Evaluators:**
1. **ResponseQualityEvaluator** - Coherence, relevance, completeness
2. **TaskCompletionEvaluator** - Task success rate
3. **FactualAccuracyEvaluator** - Factual correctness

**Usage:**
```python
from app.agents.evals import (
    evaluate_agent_response,
    create_evaluation_summary,
    get_dataset
)

# Single evaluation
result = evaluate_agent_response(
    response="Agent output",
    query="User query",
    expected_output="Expected result"
)

# Batch evaluation
dataset = get_dataset("research")
results = []
for case in dataset.test_cases:
    result = evaluate_agent_response(response, case.input, case.expected_output)
    results.append(result)
summary = create_evaluation_summary(results)
```

### Human-in-the-Loop Pattern

Used in ContentCreationAgent and HITLITSupportAgent:

```python
from langgraph.types import interrupt

def _approval_node(self, state: AgentState) -> dict:
    # Request human approval
    approval = interrupt({
        "type": "approval_request",
        "content": state["draft_content"],
        "options": ["approve", "reject", "revise"]
    })

    if approval["decision"] == "approve":
        return {"status": "approved"}
    elif approval["decision"] == "revise":
        return {"feedback": approval["feedback"]}
    else:
        return {"status": "rejected"}
```

### Enterprise Agent API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/enterprise/agents` | GET | List all available agents |
| `/api/enterprise/{agent}/invoke` | POST | Invoke agent |
| `/api/enterprise/{agent}/stream` | POST | Stream agent response |

Where `{agent}` is: `research`, `content`, `data-analyst`, `document`, `multilingual-rag`, `hitl-support`, `code-assistant`

### Enterprise Webhook Endpoints

| Endpoint | Method | Platform |
|----------|--------|----------|
| `/api/webhooks/copilot-studio` | POST | Microsoft Copilot Studio |
| `/api/webhooks/azure-ai` | POST | Azure AI Agent |
| `/api/webhooks/aws-lex` | POST | AWS Lex |

---

## Governance Framework

### Overview

The governance framework provides enterprise-grade security, compliance, and operational controls for agent deployments:

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| RBAC | Access control | 5 roles, permissions, API key mapping |
| Audit Logger | Compliance | JSON Lines, privacy hashing, async |
| Rate Limiter | Protection | Token bucket, Redis support, per-user/agent |
| Approval Workflow | HITL | Multi-level (L1-L3), callbacks, expiry |
| PII Detector | Privacy | Regex + Presidio, masking, blocking |
| Cost Tracker | Budgeting | Token usage, pricing, budget alerts |
| Anomaly Detector | Security | Rate/error/content anomalies, auto-block |
| Middleware | Integration | FastAPI middleware stack |

### Role-Based Access Control (RBAC)

**Roles:**
| Role | Permissions | Use Case |
|------|-------------|----------|
| ADMIN | All (`*`) | System administrators |
| OPERATOR | Invoke, approve L1/L2, audit read | IT operators |
| USER | Invoke, read, list agents | Regular users |
| VIEWER | Read, list agents, audit read | Auditors |
| SERVICE | Invoke, read, list agents | API integrations |

**Usage:**
```python
from app.governance import (
    get_rbac_manager, check_permission, require_permission,
    Permission, Role, UserContext
)

# Check permission
if check_permission("sk-admin-token", Permission.SYSTEM_ADMIN):
    # Admin action

# Require permission (raises PermissionDeniedError)
ctx = require_permission("sk-user-token", Permission.AGENT_INVOKE)

# API key patterns for auto-role detection:
# sk-admin-*   -> ADMIN
# sk-operator-* -> OPERATOR
# sk-service-* -> SERVICE
```

### Audit Logging

**Format:** JSON Lines for easy parsing and compliance tools

```python
from app.governance import audit_agent_response, get_audit_logger

# Log agent response
entry = audit_agent_response(
    user_id="user123",
    agent_type="helpdesk",
    input_text="Help me reset password",
    output_text="I can help with that...",
    duration_ms=1500,
)

# Query logs
logger = get_audit_logger()
entries = logger.query(user_id="user123", action=AuditAction.AGENT_INVOKE)

# Export logs
logger.export("/path/to/export.jsonl")
```

**Privacy:** Input/output are SHA-256 hashed by default. Set `log_inputs=True` / `log_outputs=True` in config to log full content.

### Rate Limiting

**Token Bucket Algorithm** with support for:
- Per-user limits (default: 100/min)
- Per-agent limits (default: 30/min)
- Global limits (default: 1000/min)
- Burst allowance (1.5x multiplier)

```python
from app.governance import check_rate_limit, require_rate_limit

# Check rate limit
result = await check_rate_limit(user_id="user123", agent_type="research")
if not result.allowed:
    print(f"Retry after {result.retry_after} seconds")

# Require rate limit (raises RateLimitExceededError)
await require_rate_limit("user123", "research")
```

**Backends:**
- In-memory (default, single instance)
- Redis (distributed, production)

### Approval Workflows

**Levels:**
| Level | Actions | Approvers |
|-------|---------|-----------|
| L1 | Create incident, share docs | Operators, Admins |
| L2 | Close incident, password reset, create change | Operators (L2), Admins |
| L3 | System restart, access revoke, config change | Admins only |

```python
from app.governance import (
    request_approval, get_approval_manager,
    ActionType, ApprovalLevel
)

# Request approval
request = await request_approval(
    action_type=ActionType.PASSWORD_RESET,
    requester_id="user123",
    agent_type="helpdesk",
    action_details={"username": "jsmith"},
)

# Approve/reject
manager = get_approval_manager()
admin_ctx = UserContext(user_id="admin", role=Role.ADMIN)
response = manager.approve(request.id, admin_ctx, reason="Verified identity")
```

### PII Detection

**Purpose:** Detect and mask Personally Identifiable Information in agent inputs/outputs

**Supported PII Types:**
| Type | Examples | Severity |
|------|----------|----------|
| Email | user@example.com | HIGH |
| Phone | 555-123-4567 | HIGH |
| Credit Card | 4111111111111111 | CRITICAL |
| SSN | 123-45-6789 | CRITICAL |
| API Key | sk-xxxx... | CRITICAL |
| IP Address | 192.168.1.1 | MEDIUM |
| Password | password: xxx | CRITICAL |

**Usage:**
```python
from app.governance import (
    PIIDetector, detect_pii, mask_pii, check_for_pii,
    PIIType, PIISeverity, PIIConfig,
)

# Simple detection
matches = detect_pii("Contact john@email.com or 555-123-4567")

# Masking
masked = mask_pii("My email is john@email.com")
# Returns: "My email is [EMAIL_REDACTED]"

# Check for critical PII (blocks credit cards, SSN, API keys)
if check_for_pii("Card: 4111111111111111"):
    raise Exception("Cannot process sensitive data")

# Custom detector configuration
config = PIIConfig(
    enabled=True,
    use_presidio=True,  # Use Presidio if available
    block_on_pii=False,
    allowed_pii_types={PIIType.EMAIL},  # Allow emails through
)
detector = PIIDetector(config)

# Add custom pattern
detector.add_custom_pattern("employee_id", r"EMP-\d{6}", PIISeverity.MEDIUM)
```

**Presidio Integration:** When `presidio-analyzer` is installed, the detector uses Presidio for enhanced NER-based detection of names, addresses, and other entities.

### Cost Tracking

**Purpose:** Track token usage and costs for budget management and analytics

**Supported Models:**
| Provider | Model | Input $/1K | Output $/1K |
|----------|-------|------------|-------------|
| OpenAI | gpt-4o | 0.005 | 0.015 |
| OpenAI | gpt-4o-mini | 0.00015 | 0.0006 |
| Anthropic | claude-3-5-sonnet | 0.003 | 0.015 |
| Anthropic | claude-3-opus | 0.015 | 0.075 |
| Anthropic | claude-3-haiku | 0.00025 | 0.00125 |

**Usage:**
```python
from app.governance import (
    CostTracker, track_usage, get_usage_summary,
    CostConfig, BudgetConfig, ModelPricing,
)

# Track usage
usage = track_usage(
    model="gpt-4o-mini",
    input_tokens=1000,
    output_tokens=500,
    user_id="user123",
    agent_type="research",
)
print(f"Cost: ${usage.cost:.6f}")

# Get usage summary
summary = get_usage_summary(user_id="user123")
print(f"Total cost: ${summary.total_cost:.4f}")
print(f"By model: {summary.by_model}")

# Budget configuration
config = CostConfig(
    budget=BudgetConfig(
        daily_limit=100.0,
        monthly_limit=2000.0,
        per_user_daily=10.0,
        alert_threshold=0.8,  # Alert at 80%
    ),
    alert_callback=lambda t, c, l: print(f"Budget alert: {t}"),
)
tracker = CostTracker(config)

# Check budget
if tracker.check_budget(user_id="user123", period="daily"):
    # Within budget
    pass

# Add custom model pricing
tracker.add_pricing(ModelPricing(
    model_name="custom-model",
    provider=ModelProvider.CUSTOM,
    input_price_per_1k=0.01,
    output_price_per_1k=0.02,
))
```

### Anomaly Detection

**Purpose:** Detect unusual patterns that may indicate security threats or abuse

**Anomaly Types:**
| Category | Type | Description |
|----------|------|-------------|
| Rate | HIGH_REQUEST_RATE | Too many requests in window |
| Rate | BURST_ACTIVITY | Spike in requests per second |
| Rate | OFF_HOURS_ACTIVITY | Unusual activity timing |
| Error | HIGH_ERROR_RATE | Excessive failures |
| Error | REPEATED_FAILURES | Consecutive failures |
| Error | AUTH_FAILURES | Multiple auth failures |
| Content | LARGE_INPUT | Oversized input |
| Content | PROMPT_INJECTION | Injection attempt patterns |
| Performance | HIGH_LATENCY | Slow responses |

**Usage:**
```python
from app.governance import (
    AnomalyDetector, record_event, check_for_anomalies,
    AnomalyConfig, RateConfig, AnomalySeverity,
)

# Record events
anomalies = record_event(
    user_id="user123",
    agent_type="research",
    event_type="request",
    success=True,
    metadata={
        "input_length": 500,
        "response_time_ms": 1200,
    },
)

# Check for recent anomalies
anomalies = check_for_anomalies(user_id="user123")
for anomaly in anomalies:
    print(f"{anomaly.anomaly_type}: {anomaly.description}")

# Get user risk score
detector = get_anomaly_detector()
risk = detector.get_user_risk_score("user123")  # 0.0 to 1.0

# Add custom detection rule
detector.add_rule(
    "slow_response",
    lambda e: e.metadata.get("response_time_ms", 0) > 5000,
    AnomalySeverity.MEDIUM,
    "Response time exceeds 5 seconds",
)

# Configuration with auto-blocking
config = AnomalyConfig(
    rate_config=RateConfig(
        max_requests_per_window=100,
        burst_threshold=10,
    ),
    auto_block=True,  # Auto-block on critical anomalies
    alert_callback=lambda a: print(f"Anomaly: {a.anomaly_type}"),
)
```

### FastAPI Middleware Integration

```python
from app.governance import setup_governance_middleware

# Add full governance stack with Phase 3 components
setup_governance_middleware(
    app,
    enable_rbac=True,
    enable_rate_limit=True,
    enable_audit=True,
    enable_pii=True,           # NEW: PII detection
    enable_anomaly=True,       # NEW: Anomaly detection
    block_on_pii=False,        # Block requests with critical PII
    block_on_anomaly=False,    # Block on critical anomalies
    exclude_paths=["/health", "/ready", "/docs"],
)

# Use in routes
from app.governance import require_admin, require_agent_invoke

@app.get("/admin/settings")
async def admin_settings(user: UserContext = Depends(require_admin)):
    return {"settings": "..."}

@app.post("/agent/invoke")
async def invoke(user: UserContext = Depends(require_agent_invoke)):
    return {"result": "..."}
```

### Environment Variables

```bash
# RBAC
RBAC_ENABLED=true
RBAC_DEFAULT_ROLE=viewer
RBAC_STRICT_MODE=false
RBAC_ADMIN_API_KEYS=sk-admin-key1,sk-admin-key2

# Audit
AUDIT_ENABLED=true
AUDIT_LOG_PATH=./logs/audit.jsonl
AUDIT_LOG_INPUTS=false
AUDIT_LOG_OUTPUTS=false

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_BACKEND=memory  # or "redis"
RATE_LIMIT_REDIS_URL=redis://localhost:6379
RATE_LIMIT_DEFAULT=100

# Approval Workflow
APPROVAL_WORKFLOW_ENABLED=true
APPROVAL_AUTO_APPROVE_L1=false
APPROVAL_EXPIRY_HOURS=24

# PII Detection (Phase 3)
PII_DETECTION_ENABLED=true
PII_USE_PRESIDIO=true
PII_BLOCK_ON_CRITICAL=false

# Cost Tracking (Phase 3)
COST_TRACKING_ENABLED=true
COST_DAILY_LIMIT=100.0
COST_MONTHLY_LIMIT=2000.0
COST_PER_USER_DAILY=10.0
COST_ALERT_THRESHOLD=0.8

# Anomaly Detection (Phase 3)
ANOMALY_DETECTION_ENABLED=true
ANOMALY_MAX_REQUESTS_PER_MINUTE=100
ANOMALY_BURST_THRESHOLD=10
ANOMALY_AUTO_BLOCK=false
```

---

## MCP Integration

### Overview

The MCP (Model Context Protocol) integration exposes enterprise agents as tools that can be used by MCP-compatible clients like Claude Desktop, VS Code extensions, or other AI applications.

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| FastMCP Server | Tool exposure | 12 tools across 5 categories |
| Gateway | Access control | Auth, rate limiting, audit |
| ServiceNow Client | ITSM integration | Live + simulation modes |

### Available Tools

| Category | Tool | Description |
|----------|------|-------------|
| **Research** | `research_topic` | Web research with citations |
| | `quick_search` | Fast simple searches |
| **ServiceNow** | `create_incident` | Create new incidents |
| | `search_incidents` | Search existing incidents |
| | `get_incident` | Get incident details |
| | `update_incident` | Update incident (notes, state) |
| | `query_cmdb` | Query configuration items |
| **Documents** | `generate_document` | Generate SOP/policy/WLI |
| | `generate_sop` | Generate Standard Operating Procedure |
| **IT Support** | `it_support_query` | General IT questions |
| | `troubleshoot_issue` | Troubleshooting guidance |
| **Code** | `review_code` | Code review |
| | `explain_code` | Code explanation |

### Running the MCP Server

```bash
# Standalone (stdio transport for Claude Desktop)
python -m app.mcp.server

# With HTTP transport
MCP_TRANSPORT=http python -m app.mcp.server
```

### ServiceNow Client

Supports both live API calls and simulation mode for development:

```python
from app.mcp import get_servicenow_client

client = get_servicenow_client()

# Create incident (simulation mode by default)
result = await client.create_incident(
    short_description="Password reset needed",
    description="User forgot password",
    priority="3",
)

# Search incidents
incidents = await client.search_incidents("password", limit=10)

# Query CMDB
servers = await client.query_cmdb("cmdb_ci_server", "web-server")
```

### MCP Gateway

Rate limiting and access control for MCP tools:

```python
from app.mcp import get_mcp_gateway, MCPClientInfo

gateway = get_mcp_gateway()

# Register authenticated client
gateway.register_token("my-token", MCPClientInfo(
    client_id="client1",
    user_id="user123",
    role="admin",
))

# Block specific tools
gateway.config.blocked_tools = ["dangerous_tool"]

# Get statistics
stats = gateway.get_stats()
```

### Environment Variables

```bash
# MCP Server
MCP_TRANSPORT=stdio  # or "http"
MCP_GATEWAY_ENABLED=true
MCP_REQUIRE_AUTH=false
MCP_RATE_LIMIT_ENABLED=true
MCP_RATE_LIMIT_PER_MINUTE=60

# ServiceNow
SERVICENOW_INSTANCE=dev12345
SERVICENOW_USERNAME=admin
SERVICENOW_PASSWORD=secret
SERVICENOW_MODE=simulation  # or "live"
```

---

## DeepSearch Research

### Overview

DeepSearch provides advanced research capabilities beyond the basic Research Agent:

- **Query Decomposition**: Breaks complex queries into focused sub-queries
- **Multi-Provider Search**: Supports Tavily, DuckDuckGo, and simulated search
- **Source Credibility Scoring**: Automatic credibility assessment for sources
- **Citation Management**: Tracks and exports citations in multiple formats
- **Structured Research Reports**: Generates comprehensive research reports

### Components

| Component | Purpose |
|-----------|---------|
| `ResearchPlanner` | Decomposes queries into sub-queries with execution strategy |
| `SourceManager` | Tracks sources with credibility scoring and citation formatting |
| `SearchProviderManager` | Unified interface to multiple search providers |
| `DeepSearchAgent` | Orchestrates the full research workflow |

### Usage

```python
from app.agents.research import (
    DeepSearchAgent,
    ResearchDepth,
    ResearchReport,
)

# Create agent
agent = DeepSearchAgent()

# Perform research
report: ResearchReport = await agent.research(
    query="What are the best practices for AI agent development?",
    depth=ResearchDepth.COMPREHENSIVE,  # quick, standard, comprehensive
    max_sources=10,
)

# Access results
print(report.summary)
print(report.findings)
print(report.to_markdown())

# Get high-credibility sources
for source in report.get_high_credibility_sources():
    print(f"{source.title}: {source.credibility_score}")
```

### Query Planner

```python
from app.agents.research import ResearchPlanner, QueryIntent

planner = ResearchPlanner()

# Classify query intent
intent = planner.classify_intent("Compare React vs Vue")
# Returns: QueryIntent.COMPARISON

# Decompose into sub-queries
plan = planner.decompose("What is the future of AI agents?", depth="standard")

# Execute in order
for batch in plan.get_execution_order():
    for sub_query in batch:
        print(f"Priority {sub_query.priority}: {sub_query.query}")
```

### Source Management

```python
from app.agents.research import (
    SourceManager,
    CitationFormat,
    CredibilityLevel,
)

manager = SourceManager()

# Add sources with auto-credibility scoring
source = manager.add_source(
    url="https://arxiv.org/paper/123",
    title="Research Paper",
    content_summary="Summary...",
    author="Dr. Smith",
)

print(source.credibility)  # CredibilityLevel.HIGH
print(source.credibility_score)  # 0.85

# Export citations
citations = manager.export_citations(CitationFormat.APA)
```

### Search Providers

```python
from app.agents.research import (
    SearchProviderManager,
    SearchProviderType,
)

manager = SearchProviderManager()

# Search with automatic provider selection
response = await manager.search("LangChain agents", max_results=5)

# Or specify provider
response = await manager.search(
    "AI research",
    provider_type=SearchProviderType.TAVILY,
)

# Parallel search
responses = await manager.search_parallel(
    queries=["query1", "query2", "query3"],
    max_results_per_query=3,
)
```

### Environment Variables

```bash
# Search Providers
TAVILY_API_KEY=tvly-xxxxx      # For Tavily search (recommended)
# DuckDuckGo requires no API key

# Search defaults
SEARCH_DEFAULT_PROVIDER=tavily  # or "duckduckgo", "simulated"
SEARCH_MAX_RESULTS=10
```

### Credibility Scoring

Sources are scored based on multiple factors:

| Factor | Weight | Examples |
|--------|--------|----------|
| Domain reputation | 40% | .gov (0.95), .edu (0.9), arxiv.org (0.9) |
| Source type | 30% | Academic paper (0.9), Documentation (0.85), Blog (0.5) |
| Content quality | 20% | Has author, meaningful title, keywords |
| Recency | 10% | Within 30 days (1.0), within 1 year (0.6) |

### Execution Strategies

| Strategy | Description |
|----------|-------------|
| `PARALLEL` | Execute all sub-queries simultaneously |
| `SEQUENTIAL` | Execute sub-queries in priority order |
| `HIERARCHICAL` | Execute based on dependencies (results feed next query) |

---

## Dependencies

### Core Dependencies

```toml
langchain>=0.3.0
langchain-core>=0.3.0
langchain-openai>=0.2.0
langchain-anthropic>=0.3.0
langchain-text-splitters>=0.3.0
langgraph>=0.2.0
langserve[all]>=0.3.0
langsmith>=0.1.0
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
python-dotenv>=1.0.0
pydantic>=2.0.0
langchain-community>=0.3.0

# Document processing
pypdf>=4.0.0
python-docx>=1.1.0
python-multipart>=0.0.9
docx2txt>=0.8

# Vector store
chromadb>=0.5.0
faiss-cpu>=1.8.0

# CLI
rich>=13.0.0
```

### Development Dependencies

```toml
pytest>=8.0.0
pytest-asyncio>=0.24.0
httpx>=0.27.0
ruff>=0.8.0
mypy>=1.13.0
```

---

## Development Patterns

### Adding a New Chain

1. Create file in `app/chains/`
2. Implement chain using LangChain patterns
3. Export in `app/chains/__init__.py`
4. Add route in `app/server.py`:

```python
from app.chains.your_chain import your_chain

add_routes(
    app,
    your_chain,
    path="/your-chain",
    enabled_endpoints=["invoke", "stream"],
)
```

### Adding a New Tool

Add to `app/chains/agent.py` or `app/chains/langgraph_agent.py`:

```python
@tool
def your_tool(param: str) -> str:
    """Tool description for LLM.

    Args:
        param: Parameter description.

    Returns:
        Result description.
    """
    return result

# Add to tools list
tools = [..., your_tool]
```

### Error Handling Pattern

```python
try:
    result = risky_operation()
except Exception as e:
    logger.error("Operation failed", exc_info=True)
    return {"error": "Operation failed. Please try again."}
```

### Type Hints

All functions must include type hints:

```python
def process_input(data: str, count: int = 10) -> dict[str, Any]:
    """Process input data.

    Args:
        data: Input string to process.
        count: Number of items to return.

    Returns:
        Processed result dictionary.
    """
```

---

## Testing Strategy

### Test Categories

1. **Unit Tests** (`tests/unit/`): No network calls
2. **Integration Tests** (`tests/integration/`): With live APIs

### Running Tests

```bash
# All tests
make test

# Specific file
pytest tests/test_server.py -v

# With coverage
pytest --cov=app --cov-report=html
```

### Test Patterns

```python
def test_endpoint_name():
    """Test description."""
    # Arrange
    client = TestClient(app)

    # Act
    response = client.get("/endpoint")

    # Assert
    assert response.status_code == 200
```

---

## Deployment

### Local Development

```bash
cd deployment
cp .env.example .env
# Edit .env with your API keys
make run-reload
```

### Docker Deployment

```bash
cp .env.example .env
# Edit .env with your API keys
docker-compose up -d
```

### Kubernetes

Use health endpoints for probes:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 5
```

---

## Common Tasks

### Task: Update LLM Model

1. Edit the relevant chain file
2. Change the `model` parameter:
   ```python
   llm = ChatOpenAI(model="gpt-4o", temperature=0)
   ```
3. Restart server

### Task: Add New Endpoint

1. Create chain in `app/chains/`
2. Add to `app/chains/__init__.py`
3. Add route in `app/server.py`
4. Add tests in `tests/`
5. Update this KNOWLEDGE.md

### Task: Enable New LLM Provider

1. Add dependency to `pyproject.toml`
2. Add environment variable to `.env.example`
3. Update `load_chains()` or `load_langgraph_agent()`
4. Update documentation

### Task: Add Tracing Tags

```python
from langsmith import traceable

@traceable(name="custom_operation", tags=["production"])
def my_function():
    pass
```

---

## Troubleshooting

### CRITICAL FIXES (2025-12-19)

**Issue: "Failed to start session: Unknown error" (IT Support Agents)**

**Root Cause**: Global agent instantiation before environment variables loaded
- Files `it_helpdesk.py` and `servicenow_agent.py` created agents at module import
- Import happened before `.env` loaded in `server.py`
- Agents couldn't find `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`

**Fix Applied**:
- Removed global instantiations from both files (lines 641, 653)
- Agents now created lazily by `ConversationManager._load_agents()`

**Files Changed**:
- `app/agents/it_helpdesk.py`
- `app/agents/servicenow_agent.py`
- `app/agents/__init__.py`

---

**Issue: "No response received" (Enterprise Agents)**

**Root Cause**: API key middleware blocking local requests
- `API_KEY_ENABLED=true` blocked all `/api/*` endpoints
- Chat UI couldn't communicate with backend

**Fix Applied**:
- Set `API_KEY_ENABLED=false` in `.env` for local development

**File Changed**: `.env`

---

**Issue: "Invalid or missing API key" after changing .env**

**Cause**: Uvicorn hot reload doesn't reload environment variables

**Solution**:
1. **STOP server completely** (CTRL+C or kill process)
2. Restart: `python -m app.server`
3. Don't rely on hot reload for `.env` changes!

---

**Issue: GraphRecursionError - "Recursion limit of 25/50/100 reached"**

**Root Cause**: LangGraph agents making too many tool calls exceeding the default or configured recursion limit

**Common Scenarios**:
- Research Agent performing multiple web searches
- Complex multi-step workflows requiring many LLM calls
- Verbose system prompts causing over-iteration

**Solutions**:

1. **Increase Recursion Limit** (Proper Method):
```python
# In agent's compile() method:
def compile(self) -> None:
    self._graph = self._build_graph()
    self._compiled_graph = self._graph.compile(checkpointer=MemorySaver())
    # Store as instance variable
    self._recursion_limit = 200  # Increase for complex workflows

# In agent's invoke() method:
def invoke(self, message: str, session_id: str | None = None, **kwargs) -> dict:
    config = {
        "configurable": {"thread_id": session_id or "default"},
        "recursion_limit": getattr(self, '_recursion_limit', 100),  # Apply at invoke time
    }
    result = self._compiled_graph.invoke(input_state, config=config)
    return result
```

2. **Optimize System Prompt** (Reduce Iterations):
```python
# Add efficiency guidelines to system prompt
"""
IMPORTANT: Efficiency Guidelines
- Use MAXIMUM 2-3 tool calls per query
- After gathering information, provide your final answer immediately
- Do NOT keep searching for more information - be decisive
"""
```

3. **Clear Python Bytecode Cache** (If changes not applying):
```bash
# Kill all Python processes
powershell -Command "Get-Process python* | Stop-Process -Force"

# Clear cache
rm -rf app/**/__pycache__

# Start server without reload
python -c "import uvicorn; from app.server import app; uvicorn.run(app, host='0.0.0.0', port=8000, reload=False)"
```

**IMPORTANT**:
- ❌ DO NOT pass `recursion_limit` to `compile()` - it's not a valid parameter
- ✅ DO pass `recursion_limit` in config dict during `invoke()`
- ❌ DO NOT use `.with_config()` after compilation - can cause issues
- ✅ DO store limit as instance variable in `compile()`, apply in `invoke()`

**Reference**:
- Example implementation: [app/agents/research/research_agent.py](app/agents/research/research_agent.py) (lines 229-271, 322-357)
- LangGraph docs: https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT

---

### Testing

**Run automated tests**:
```bash
# PowerShell
.\test-agents.ps1

# Or manually
curl http://localhost:8000/health
```

**Detailed fix documentation**: See [../FIXES.md](../FIXES.md)

---

### Issue: Chains not loading

**Symptom**: `chains_loaded: false` in health check

**Solutions**:
1. Verify `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is set in `.env`
2. Check `.env` file is in deployment directory
3. Restart server after changing `.env`
4. No spaces or newlines in key values

### Issue: Import errors with agents

**Symptom**: `cannot import name 'create_tool_calling_agent'`

**Solution**: Use LangGraph's `create_react_agent` instead:
```python
from langgraph.prebuilt import create_react_agent
```

### Issue: Tracing not working

**Symptom**: No traces in LangSmith

**Solutions**:
1. Set `LANGCHAIN_TRACING_V2=true`
2. Verify `LANGCHAIN_API_KEY` is valid
3. Check `LANGCHAIN_PROJECT` name

### Issue: Docker container fails

**Symptom**: Container exits immediately

**Solutions**:
1. Check logs: `docker-compose logs -f`
2. Verify `.env` file exists
3. Ensure port 8000 is not in use

### Issue: Web UI not responding

**Solutions**:
1. Hard refresh browser (CTRL+SHIFT+R)
2. Clear browser cache
3. Check browser console (F12) for errors
4. Try incognito/private mode
5. Verify server: `curl http://localhost:8000/health`

---

## Change Log

### 2026-01-02 - Teams & Slack Integrations (v3.9)

**Added**:
- **Microsoft Teams Webhook Integration**: Full Bot Framework support
  - `app/integrations/teams_webhook.py` - Teams message handling
  - `TeamsAdaptiveCard`: Adaptive Card builder for rich messages
  - `TeamsMessageCard`: Legacy Message Card support
  - `TeamsActivity`: Incoming activity parsing (message, invoke, conversationUpdate)
  - `TeamsWebhookHandler`: Process incoming messages and route to agents
  - Support for card actions, mentions, and conversation contexts

- **Slack Webhook Integration**: Events API and interactivity support
  - `app/integrations/slack_webhook.py` - Slack event handling
  - `SlackBlockBuilder`: Block Kit message construction
  - `SlackMessage`: Rich message formatting with attachments
  - `SlackEvent`: Event parsing (message, app_mention, reaction)
  - `SlackWebhookHandler`: Process events and route to agents
  - `verify_slack_signature()`: HMAC signature verification for security
  - Slash command support with response formatting

- **Integration Routes**: FastAPI endpoints for external platforms
  - `app/integrations/routes.py` - Webhook endpoint routes
  - `POST /api/integrations/teams/webhook` - Teams Bot Framework endpoint
  - `POST /api/integrations/slack/events` - Slack Events API endpoint
  - `POST /api/integrations/slack/commands` - Slash command handler
  - `POST /api/integrations/slack/interactive` - Block Kit interactions

**Environment Variables**:
```bash
TEAMS_BOT_ID=your-bot-id
TEAMS_APP_ID=your-app-id
TEAMS_APP_PASSWORD=your-app-password
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret
SLACK_APP_TOKEN=xapp-your-app-token
```

**Testing**:
- 44 unit tests for Teams and Slack integrations (all passing)
- Tests cover message building, event parsing, signature verification, and webhook handling

**Files Added**:
- `deployment/app/integrations/__init__.py` - Module exports
- `deployment/app/integrations/teams_webhook.py` - ~300 lines
- `deployment/app/integrations/slack_webhook.py` - ~350 lines
- `deployment/app/integrations/routes.py` - ~200 lines
- `deployment/tests/test_integrations.py` - ~450 lines

---

### 2026-01-02 - Memory & Persistence Upgrade (v3.8)

**Added**:
- **Session Memory Module**: Multiple backend support for conversation persistence
  - `app/memory/base.py` - Base types: Message, Session, SessionMetadata, BaseSessionStore
  - `app/memory/memory_store.py` - In-memory session store (development/testing)
  - `app/memory/redis_store.py` - Redis session store (production/distributed)
  - `app/memory/sqlite_store.py` - SQLite session store (single-instance persistence)
  - `app/memory/conversation_memory.py` - LangChain-integrated conversation memory
  - `app/memory/config.py` - Configuration and factory functions

- **Storage Backends**:
  - `InMemorySessionStore`: Thread-safe in-memory storage with max sessions limit
  - `RedisSessionStore`: Redis-backed with TTL support and user/agent indices
  - `SQLiteSessionStore`: SQLite-backed with VACUUM and stats support

- **Conversation Features**:
  - `ConversationMemory`: High-level API for managing conversations
  - `ConversationSummary`: Session metadata and message counts
  - `get_langchain_messages()`: Convert to HumanMessage/AIMessage format
  - `get_chat_history_string()`: Formatted history for prompts

- **Configuration**:
  - `MemoryBackend` enum: MEMORY, REDIS, SQLITE
  - `MemoryConfig.from_env()`: Environment-based configuration
  - `get_session_store()`: Factory with singleton pattern
  - `get_checkpointer()`: LangGraph checkpointer factory

- **LangGraph Integration**:
  - `CheckpointerType` enum: MEMORY, REDIS, SQLITE, POSTGRES
  - Auto-matching checkpointer to session backend
  - Support for MemorySaver, SqliteSaver, PostgresSaver

**Environment Variables**:
```bash
MEMORY_BACKEND=memory|redis|sqlite
REDIS_URL=redis://localhost:6379
SQLITE_PATH=data/sessions.db
SESSION_TTL_HOURS=24
MAX_SESSIONS=10000
SESSION_KEY_PREFIX=session:
```

**Testing**:
- 59 unit tests for session memory module (all passing)
- Tests cover all backends, conversation memory, config, and factories

**Files Added**:
- `deployment/app/memory/__init__.py` - Module exports
- `deployment/app/memory/base.py` - ~400 lines
- `deployment/app/memory/memory_store.py` - ~305 lines
- `deployment/app/memory/redis_store.py` - ~300 lines
- `deployment/app/memory/sqlite_store.py` - ~575 lines
- `deployment/app/memory/conversation_memory.py` - ~400 lines
- `deployment/app/memory/config.py` - ~290 lines
- `deployment/tests/test_session_memory.py` - ~800 lines

---

### 2026-01-01 - Enhanced Governance & Security (v3.7)

**Added**:
- **PII Detection**: Privacy protection for agent inputs/outputs
  - `app/governance/pii_detector.py` - Regex + Presidio-based PII detection
  - Supports: email, phone, credit cards, SSN, API keys, passwords, IP addresses
  - Masking with configurable redaction format
  - Severity levels: LOW, MEDIUM, HIGH, CRITICAL

- **Cost Tracking**: Token usage and budget management
  - `app/governance/cost_tracker.py` - Multi-model pricing and tracking
  - Pre-configured pricing for OpenAI and Anthropic models
  - Daily/monthly budget limits with alerts
  - Usage summaries by user, agent, and model

- **Anomaly Detection**: Security threat and abuse detection
  - `app/governance/anomaly_detector.py` - Pattern-based anomaly detection
  - Rate anomalies: high request rate, burst activity, off-hours
  - Error anomalies: high error rate, consecutive failures, auth failures
  - Content anomalies: large input/output, prompt injection detection
  - User risk scoring and auto-blocking

- **Middleware Integration**: Extended governance middleware
  - `PIIMiddleware` - Request PII scanning with optional blocking
  - `AnomalyMiddleware` - Event recording and anomaly detection
  - Updated `setup_governance_middleware()` with new options

**Testing**:
- 59 unit tests for Phase 3 components (all passing)
- Tests cover PII detection, cost tracking, anomaly detection, and middleware integration

**Files Added**:
- `deployment/app/governance/pii_detector.py` - ~550 lines
- `deployment/app/governance/cost_tracker.py` - ~500 lines
- `deployment/app/governance/anomaly_detector.py` - ~650 lines
- `deployment/tests/test_governance_phase3.py` - ~650 lines

**Files Modified**:
- `deployment/app/governance/__init__.py` - Added new exports
- `deployment/app/governance/middleware.py` - Added PII and Anomaly middleware

---

### 2026-01-01 - DeepSearch Enhancement (v3.6)

**Added**:
- **DeepSearch Research System**: Advanced multi-step research capabilities
  - `app/agents/research/planner.py` - Query decomposition with execution strategies
  - `app/agents/research/source_manager.py` - Citation tracking with credibility scoring
  - `app/agents/research/search_providers.py` - Multi-provider search abstraction
  - `app/agents/research/deep_search_agent.py` - Enhanced research agent orchestration

**Features**:
- **Query Planning**: LLM-powered query decomposition into focused sub-queries
- **Execution Strategies**: Parallel, sequential, and hierarchical execution modes
- **Source Credibility**: Automatic scoring based on domain, type, content, and recency
- **Citation Formats**: Support for APA, MLA, IEEE, Chicago, Markdown, and plain text
- **Search Providers**: Tavily (AI-optimized), DuckDuckGo (no API key), Simulated (testing)
- **Research Reports**: Structured reports with findings, sources, and markdown export

**Key Classes**:
- `ResearchPlanner` - Decomposes complex queries into sub-queries
- `SourceManager` - Manages sources with credibility scoring
- `SearchProviderManager` - Unified interface to multiple search backends
- `DeepSearchAgent` - Full research workflow orchestration

**Testing**:
- 61 unit tests for DeepSearch module (all passing)
- Tests cover planner, source manager, search providers, and agent integration
- Integration tests with mocked LLM responses

**Files Added**:
- `deployment/app/agents/research/planner.py` - ~350 lines
- `deployment/app/agents/research/source_manager.py` - ~550 lines
- `deployment/app/agents/research/search_providers.py` - ~500 lines
- `deployment/app/agents/research/deep_search_agent.py` - ~450 lines
- `deployment/tests/test_deep_search.py` - ~700 lines

---

### 2025-12-31 - MCP Integration (v3.5)

**Added**:
- **MCP Server**: FastMCP server exposing enterprise agents as tools
  - `app/mcp/server.py` - 12 tools across 5 categories (research, servicenow, docs, IT support, code)
  - `app/mcp/gateway.py` - Access control with auth, rate limiting, audit logging
  - `app/mcp/servicenow_client.py` - Real ServiceNow REST API client with simulation mode

**Features**:
- **Research Tools**: `research_topic`, `quick_search` for web research
- **ServiceNow Tools**: Full ITSM integration (incidents, CMDB, changes)
- **Document Tools**: Generate SOPs, policies, WLIs
- **IT Support Tools**: General IT queries and troubleshooting
- **Code Tools**: Code review and explanation

**ServiceNow Client**:
- Dual mode: `simulation` (default) for development, `live` for production
- Full incident lifecycle: create, search, get, update
- CMDB queries for configuration items
- Change request creation

**Testing**:
- 48 unit tests for MCP module (all passing)
- Tests cover gateway, ServiceNow client, and integration scenarios

**Files Added**:
- `deployment/app/mcp/__init__.py` - Module exports
- `deployment/app/mcp/server.py` - ~450 lines
- `deployment/app/mcp/gateway.py` - ~300 lines
- `deployment/app/mcp/servicenow_client.py` - ~400 lines
- `deployment/tests/test_mcp.py` - ~500 lines

---

### 2025-12-31 - Governance Framework (v3.4)

**Added**:
- **Governance Framework**: Complete enterprise governance layer for agent deployments
  - `app/governance/rbac.py` - Role-Based Access Control with 5 roles (ADMIN, OPERATOR, USER, VIEWER, SERVICE)
  - `app/governance/audit_logger.py` - JSON Lines audit logging with privacy-preserving hashing
  - `app/governance/rate_limiter.py` - Token bucket rate limiting with Redis support
  - `app/governance/approval_workflow.py` - Multi-level (L1-L3) approval workflows for sensitive actions
  - `app/governance/middleware.py` - FastAPI middleware integration for RBAC, rate limiting, and audit

**Features**:
- **RBAC**: API key-based role detection (sk-admin-*, sk-operator-*, sk-service-*)
- **Audit**: Async logging, SHA-256 content hashing, log rotation, query/export
- **Rate Limiting**: Per-user, per-agent, and global limits with burst allowance
- **Approvals**: 24-hour expiry, callback notifications, requester cancellation
- **Middleware**: GovernanceExceptionMiddleware for unified error handling

**Testing**:
- 68 unit tests for governance framework (all passing)
- Tests cover RBAC, audit logging, rate limiting, and approval workflows
- Integration test validates full governance workflow

**Files Added**:
- `deployment/app/governance/__init__.py` - Module exports
- `deployment/app/governance/rbac.py` - 434 lines
- `deployment/app/governance/audit_logger.py` - 350 lines
- `deployment/app/governance/rate_limiter.py` - 450 lines
- `deployment/app/governance/approval_workflow.py` - 420 lines
- `deployment/app/governance/middleware.py` - 380 lines
- `deployment/tests/test_governance.py` - 670 lines

---

### 2025-12-29 - Enterprise Agent API Fixes (v3.3)

**Fixed**:
- **All Enterprise Agent Endpoints**: Fixed empty response issue for all enterprise agents
  - **Root Cause**: Server was using `result.get("output", "")` but LangGraph returns `{messages: [...]}`
  - **Solution**: Added `extract_agent_response()` helper to properly extract AI message content from LangGraph state
  - Modified: `app/server.py` - Added helper function after imports

- **Document Generator Agent**: Fixed method name and response extraction
  - Modified: `app/server.py` (line 1362) - Changed `generate()` to `create_document()`

- **Content Agent**: Fixed method name and parameter name
  - Modified: `app/server.py` (line 1362) - Changed `generate()` to `create_content()`
  - Modified: `app/server.py` (line 1366) - Changed `audience` to `target_audience`
  - **Note**: Content Agent has HITL (Human-In-The-Loop) workflow requiring human approval

- **Multilingual RAG Agent**: Fixed language parameter validation
  - Modified: `app/server.py` (line 1535) - Added `or "auto"` default for None values

**Added**:
- **File Upload UI for Enterprise Agents**: Added document upload capability
  - Modified: `app/static/chat.html` - Added upload button and handler
  - Supports: Multilingual RAG Agent and Data Analyst Agent
  - Endpoints: `/api/enterprise/rag/upload`, `/api/enterprise/data-analyst/upload`

**Testing Results**:
7 of 8 enterprise agents verified working:
- ✅ Research Agent - Working with recursion limit configured
- ✅ Document Generator Agent - Working with `create_document()` method
- ✅ HITL IT Support Agent - Working with proper response extraction
- ✅ Code Assistant Agent - Working with proper response extraction
- ✅ Multilingual RAG Agent - Working with file upload support
- ✅ Data Analyst Agent - Working with proper response extraction
- ⚠️ Content Agent - Uses HITL workflow (requires human approval via chat UI)

**Files Changed**:
- `deployment/app/server.py` - Added `extract_agent_response()`, fixed all enterprise agent endpoints
- `deployment/app/static/chat.html` - Added file upload UI for RAG and Data Analyst agents
- `deployment/app/agents/content/content_agent.py` - Added recursion limit configuration

---

### 2025-12-29 - Agent Fixes and Optimizations (v3.2)

**Fixed**:
- **Research Agent Recursion Limit**: Resolved `GraphRecursionError` by implementing proper recursion limit configuration
  - Modified: `app/agents/research/research_agent.py`
  - Override `compile()` method to store `_recursion_limit = 200` as instance variable
  - Override `invoke()` method to pass `recursion_limit` in config at runtime
  - **Root Cause**: LangGraph requires recursion_limit to be passed at `invoke()` time, not `compile()` time
  - **Technical Solution**:
    ```python
    # In compile():
    self._recursion_limit = 200

    # In invoke():
    config = {
        "configurable": {"thread_id": session_id or "default"},
        "recursion_limit": recursion_limit,
    }
    result = self._compiled_graph.invoke(input_state, config=config)
    ```

- **IT Support Agents LLM Provider**: Standardized to use OpenAI as primary provider
  - Modified: `app/agents/it_helpdesk.py` (lines 483-490)
  - Modified: `app/agents/servicenow_agent.py` (lines 548-555)
  - Changed provider preference order: OpenAI (primary) → Anthropic (fallback)
  - Ensures consistent model behavior across all IT support workflows

- **Document Generator Method Call**: Fixed incorrect method name
  - Modified: `app/server.py` (lines 1461-1475)
  - Changed from `document_agent.generate()` to `document_agent.create_document()`
  - Resolves `AttributeError: 'DocumentAgent' object has no attribute 'generate'`

**Optimized**:
- **Research Agent System Prompt**: Reduced tool iterations for faster responses
  - Added explicit efficiency guidelines: "Use MAXIMUM 2-3 web searches per query"
  - Changed from verbose research workflow to concise, decisive approach
  - **Impact**: Reduces recursion depth from 100+ steps to ~20-30 steps
  - Maintains quality while significantly improving performance

**Technical Discoveries**:
- **LangGraph Configuration Pattern**:
  - `recursion_limit` must be passed in the config parameter during `invoke()` calls
  - Using `.with_config()` after compilation creates new instance via `copy()` which can cause issues
  - Store limit as instance variable during `compile()`, apply during `invoke()`

- **Python Bytecode Caching Issue**:
  - Uvicorn hot reload does NOT clear `__pycache__` directories
  - Server can serve stale code even after file modifications
  - **Solution**: Kill all Python processes, clear cache manually, restart without reload

- **System Prompt Impact**:
  - Overly detailed prompts can cause excessive tool iterations
  - LLMs tend to "over-research" when given verbose instructions
  - Concise, directive prompts produce more efficient execution

**Testing Results**:
All 4 agents verified working via `test_agents_quick.py`:
- ✅ IT Helpdesk Agent - Using OpenAI model, session management working
- ✅ Research Agent - Recursion limit properly configured (200 steps)
- ✅ Document Generator Agent - Using `create_document()` method
- ✅ ServiceNow Agent - Using OpenAI model, ITSM operations functional

**Files Changed**:
- `deployment/app/agents/research/research_agent.py` (lines 229-271, 322-357)
- `deployment/app/agents/it_helpdesk.py` (lines 483-490) - previous session
- `deployment/app/agents/servicenow_agent.py` (lines 548-555) - previous session
- `deployment/app/server.py` (lines 1461-1475) - previous session

**Impact**:
- ✅ Research Agent can handle complex multi-step research workflows
- ✅ All IT support agents use consistent OpenAI models
- ✅ Document generation working correctly
- ✅ Platform stability improved with proper error handling
- ✅ Performance optimized through system prompt refinement

**Best Practices Established**:
1. **LangGraph Recursion Configuration**: Always pass `recursion_limit` in invoke config, not compile
2. **System Prompt Engineering**: Keep prompts concise and directive for efficiency
3. **Server Restart Protocol**: Full restart required for Python code changes (clear cache + no reload)
4. **LLM Provider Standardization**: Use single provider across agent types for consistency

**Reference Documentation**:
- [LangGraph Recursion Limit Docs](https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT)
- [LangGraph How-To: Loops](https://langchain-ai.github.io/langgraphjs/how-tos/recursion-limit/)

---

### 2025-12-19 - Critical Bug Fixes (v3.1)

**Fixed**:
- **IT Support Agents**: Removed global instantiation causing "Failed to start session" error
  - Modified: `app/agents/it_helpdesk.py`, `app/agents/servicenow_agent.py`
  - Agents now created lazily after environment variables load
- **Enterprise Agents**: Disabled API key protection for local development
  - Modified: `.env` - Set `API_KEY_ENABLED=false`
  - Resolves "No response received" error in chat UI
- **Environment Loading**: Fixed import order to ensure `.env` loads before agents

**Added**:
- Comprehensive fix documentation: [../FIXES.md](../FIXES.md)
- Automated test script: [../test-agents.ps1](../test-agents.ps1)
- Enhanced troubleshooting section with root cause analysis

**Files Changed**:
- `deployment/app/agents/it_helpdesk.py` (line 641)
- `deployment/app/agents/servicenow_agent.py` (line 653)
- `deployment/app/agents/__init__.py` (line 15)
- `deployment/.env` (lines 76-80)

**Impact**:
- ✅ All agents now load successfully on server startup
- ✅ IT Support conversation sessions start without errors
- ✅ Enterprise agents respond correctly via API and Web UI
- ✅ Local development no longer requires API key configuration

---

### 2025-12-19 - Documentation Consolidation (v3.0)

**Changed**:
- Merged root `CLAUDE.md` and `.claude/CLAUDE.md` into single unified document
- Consolidated development guidelines into `.claude/CLAUDE.md`
- Updated knowledge base version and references

**Documentation Structure**:
- `.claude/CLAUDE.md`: Enterprise development standards and project guidelines
- `deployment/KNOWLEDGE.md`: This file - detailed architecture and implementation guide
- `README.md`: Project overview and quick start

**Reference**:
- Development guidelines: [.claude/CLAUDE.md](../.claude/CLAUDE.md)

---

### 2025-12-15 - IT Support Agents (v2.0)

**Added**:
- IT Helpdesk Agent with LangGraph and conversation memory
  - Tools: search_knowledge_base, create_support_ticket, check_ticket_status, check_system_status, initiate_password_reset, request_software, escalate_to_human
- ServiceNow ITSM Agent
  - Tools: search_incidents, get_incident_details, create_incident, update_incident, get_change_requests, search_cmdb, get_my_tickets
- Conversation Manager for session-based multi-agent conversations
- Web UI (`/chat`) for browser-based demos
- CLI chat interface (`cli_chat.py`) with Rich terminal UI
- Webhook API (`/api/webhook/chat`) for external platform integration
- Conversation API endpoints (`/api/conversation/*`)
- Document RAG chain with PDF/Word/TXT support
- FAISS vector store for document embeddings

**Technical Decisions**:
- Used LangGraph StateGraph with MemorySaver for conversation persistence
- Implemented webhook-based integration pattern for external platforms
- Added session-based conversation management for multi-agent support
- CLI uses Rich library for enhanced terminal experience

**External Integration Support**:
- Microsoft Copilot Studio (via HTTP actions)
- Azure AI Agent (via webhook connectors)
- AWS AI Agent (via Lambda integration)
- Any HTTP-capable platform

---

### 2025-12-15 - Initial Release (v1.0)

**Added**:
- FastAPI server with LangServe integration
- Chat chain with OpenAI GPT-4o-mini
- RAG chain with in-memory vector store
- LangGraph React agent with tools
- LangGraph agent with Anthropic/OpenAI support
- LangSmith tracing configuration
- Docker and docker-compose setup
- Health and readiness endpoints
- Unit tests for server
- Comprehensive documentation

**Technical Decisions**:
- Chose `langgraph.prebuilt.create_react_agent` over deprecated `create_tool_calling_agent`
- Implemented lazy loading for chains based on API key availability
- Added provider auto-selection (Anthropic > OpenAI) for LangGraph agent

---

## Guidelines for AI Agents

### Essential Reading

Before making any changes, review these documents:
1. **This file** (`deployment/KNOWLEDGE.md`) - Architecture and implementation details
2. **Development Standards** ([.claude/CLAUDE.md](../.claude/CLAUDE.md)) - Code quality, security, and git workflow standards

### When Making Changes

1. **Read both documentation files** before making any changes
2. **Follow existing patterns** in the codebase
3. **Update KNOWLEDGE.md** when adding new features
4. **Run tests** before committing
5. **Use conventional commits** format

### Code Style

- Python 3.10+ features allowed
- Type hints required on all functions
- Google-style docstrings
- Ruff for linting and formatting
- No hardcoded secrets

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Security

- Never commit `.env` files
- Use environment variables for secrets
- Validate all user inputs
- Don't expose internal errors to users
