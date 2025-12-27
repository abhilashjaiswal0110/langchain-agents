# LangChain Platform - Knowledge Base

> **Purpose**: This document serves as the authoritative knowledge source for AI agents working on this repository. It contains architectural decisions, implementation patterns, and guidelines that must be followed when making changes or enhancements.

**Last Updated**: 2025-12-19 (v3.0 - Documentation Consolidation)

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
11. [Dependencies](#dependencies)
12. [Development Patterns](#development-patterns)
13. [Testing Strategy](#testing-strategy)
14. [Deployment](#deployment)
15. [Common Tasks](#common-tasks)
16. [Troubleshooting](#troubleshooting)
17. [Change Log](#change-log)

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
│   ├── agents/                  # IT Support agents (NEW)
│   │   ├── __init__.py          # Exports all agents
│   │   ├── it_helpdesk.py       # IT Helpdesk Agent
│   │   ├── servicenow_agent.py  # ServiceNow ITSM Agent
│   │   └── conversation_manager.py  # Session management
│   └── static/                  # Static web files (NEW)
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
