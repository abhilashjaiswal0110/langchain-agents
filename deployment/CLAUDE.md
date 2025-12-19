# LangChain Enterprise Agents Platform - Knowledge Base

## Quick Reference

```bash
# Start server
cd deployment && python -m app.server

# Run tests
cd deployment && python -m pytest tests/ -v --cov=app --cov-report=html

# Format code
cd deployment && black app/ tests/

# Lint code
cd deployment && ruff check app/ tests/
```

## Architecture Overview

```
deployment/
├── app/
│   ├── server.py              # FastAPI server with all endpoints
│   ├── agents/
│   │   ├── __init__.py        # Agent exports
│   │   ├── base.py            # BaseAgent abstract class
│   │   ├── research_agent.py  # Web research agent
│   │   ├── content_agent.py   # Content creation with HITL
│   │   ├── data_analyst.py    # Data analysis agent
│   │   ├── document_agent.py  # Document generation
│   │   ├── multilingual_rag.py # Multi-language RAG
│   │   ├── hitl_support.py    # IT support with approvals
│   │   ├── code_assistant.py  # Code review/generation
│   │   ├── documents/         # Document templates
│   │   │   └── templates/     # SOP, WLI, Policy templates
│   │   └── evals/
│   │       ├── __init__.py    # Eval exports
│   │       ├── evaluators.py  # 3 evaluators
│   │       └── datasets.py    # Test datasets
│   └── static/
│       └── chat.html          # Web UI
├── tests/
│   ├── conftest.py            # Pytest fixtures
│   ├── test_evaluators.py     # Evaluator tests
│   ├── test_datasets.py       # Dataset tests
│   └── test_enterprise_agents.py # API tests
└── .env.example               # Environment template
```

## 7 Enterprise Agents

| Agent | Purpose | HITL | Key Features |
|-------|---------|------|--------------|
| ResearchAgent | Web research | No | Tavily search, source citations |
| ContentCreationAgent | Marketing content | Yes | Draft approval, revision cycles |
| DataAnalystAgent | Data analysis | No | CSV/Excel processing, visualizations |
| DocumentGenerationAgent | Doc creation | No | SOP, WLI, Policy templates |
| MultilingualRAGAgent | Multi-language Q&A | No | 10+ languages, translation |
| HITLITSupportAgent | IT tickets | Yes | Priority routing, approvals |
| CodeAssistantAgent | Code help | No | Review, generation, debugging |

## API Endpoints

### Health & Discovery
- `GET /health` - Health check with agent status
- `GET /api/enterprise/agents` - List all available agents

### Agent Invocation
```
POST /api/enterprise/{agent}/invoke
POST /api/enterprise/{agent}/stream
```

Where `{agent}` is: research, content, data-analyst, document, multilingual-rag, hitl-support, code-assistant

### Webhook Integration (3rd Party)
```
POST /api/webhooks/copilot-studio  # Microsoft Copilot Studio
POST /api/webhooks/azure-ai        # Azure AI Agent
POST /api/webhooks/aws-lex         # AWS Lex
POST /api/webhook/chat             # Generic webhook (legacy)
```

## BaseAgent Pattern

All agents extend `BaseAgent`:

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

## Evaluation Framework

### 3 Evaluators
1. **ResponseQualityEvaluator** - Coherence, relevance, completeness
2. **TaskCompletionEvaluator** - Task success rate
3. **FactualAccuracyEvaluator** - Factual correctness

### Usage
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

## LangSmith Tracing

All agents use `@traceable` decorator:

```python
from langsmith import traceable

class MyAgent(BaseAgent):
    @traceable(name="my_agent_invoke")
    async def ainvoke(self, input_data: dict) -> dict:
        # Traced execution
        pass
```

Configure in `.env`:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_your_key
LANGCHAIN_PROJECT=langchain-platform
```

## Human-in-the-Loop Pattern

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

## Environment Variables

Required:
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` - LLM provider
- `LANGCHAIN_API_KEY` - LangSmith tracing

Optional:
- `TAVILY_API_KEY` - Web search (ResearchAgent)
- `ENTERPRISE_AGENT_PROVIDER` - Default: openai
- `ENTERPRISE_AGENT_MODEL` - Default: gpt-4o-mini

## Development Tasks

### Adding a New Agent

1. Create `app/agents/new_agent.py`
2. Extend `BaseAgent` class
3. Implement `_build_graph()` method
4. Add to `app/agents/__init__.py`
5. Add endpoint in `app/server.py`
6. Create test dataset in `app/agents/evals/datasets.py`
7. Add tests in `tests/test_enterprise_agents.py`
8. Update UI in `app/static/chat.html`

### Running Evaluations

```bash
# Run evaluation suite
python -m pytest tests/test_evaluators.py -v

# Run with coverage
python -m pytest tests/ --cov=app --cov-report=html
```

### Testing Webhooks

```bash
# Copilot Studio webhook
curl -X POST http://localhost:8000/api/webhooks/copilot-studio \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python?", "agent_type": "research", "user_id": "user-123"}'

# Azure AI webhook
curl -X POST http://localhost:8000/api/webhooks/azure-ai \
  -H "Content-Type: application/json" \
  -d '{"query": "Generate a report", "agent_type": "document", "deployment_id": "dep-123"}'

# AWS Lex webhook
curl -X POST http://localhost:8000/api/webhooks/aws-lex \
  -H "Content-Type: application/json" \
  -d '{"query": "Review this code", "agent_type": "code-assistant", "bot_id": "bot-123"}'
```

## Success Criteria Checklist

- [x] 7 enterprise agents implemented
- [x] LangSmith tracing enabled
- [x] Human-in-the-loop for Content and IT Support
- [x] 3 evaluators in evaluation suite
- [x] API endpoints for all agents
- [x] Documentation complete
- [x] No security vulnerabilities (env vars for secrets)
- [x] Tests passing (>80% coverage target)

## Common Issues

### Agent not loading
- Check API keys in `.env`
- Verify `ENTERPRISE_AGENT_PROVIDER` matches your API key

### Tracing not working
- Ensure `LANGCHAIN_TRACING_V2=true`
- Verify `LANGCHAIN_API_KEY` is valid

### HITL not responding
- Check interrupt handling in client
- Verify session ID is consistent across requests
