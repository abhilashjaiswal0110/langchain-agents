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
  <a href="https://twitter.com/langchainai" target="_blank"><img src="https://img.shields.io/twitter/url/https/twitter.com/langchainai.svg?style=social&label=Follow%20%40LangChainAI" alt="Twitter / X"></a>
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

- [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) – Build agents that can reliably handle complex tasks with LangGraph, our low-level agent orchestration framework. LangGraph offers customizable architecture, long-term memory, and human-in-the-loop workflows – and is trusted in production by companies like LinkedIn, Uber, Klarna, and GitLab.
- [Integrations](https://docs.langchain.com/oss/python/integrations/providers/overview) – List of LangChain integrations, including chat & embedding models, tools & toolkits, and more
- [LangSmith](https://www.langchain.com/langsmith) – Helpful for agent evals and observability. Debug poor-performing LLM app runs, evaluate agent trajectories, gain visibility in production, and improve performance over time.
- [LangSmith Deployment](https://docs.langchain.com/langsmith/deployments) – Deploy and scale agents effortlessly with a purpose-built deployment platform for long-running, stateful workflows. Discover, reuse, configure, and share agents across teams – and iterate quickly with visual prototyping in [LangSmith Studio](https://docs.langchain.com/langsmith/studio).
- [Deep Agents](https://github.com/langchain-ai/deepagents) *(new!)* – Build agents that can plan, use subagents, and leverage file systems for complex tasks

## Enterprise Agents Deployment

This repository includes a production-ready **Enterprise Agents Platform** in the [deployment/](deployment/) folder, built with LangChain and LangGraph for real-world AI agent applications.

### 🚀 Key Features

- **7 Production Agents**: Research, Content Generation (HITL), Data Analysis, Document Processing, Multilingual RAG, IT Support (HITL), Code Assistant
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
# Edit .env with your API keys (OPENAI_API_KEY, LANGSMITH_API_KEY, etc.)

# Run locally
python app/server.py

# Or use Docker
docker-compose up --build
```

### 🔗 Integration Examples

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

### 🎯 Use Cases

- **Enterprise Knowledge Management**: RAG-based document querying with multilingual support
- **Content Creation Workflows**: Human-in-the-loop content generation with approval gates
- **Data Analysis Automation**: Automated insights generation with visualization
- **IT Service Desk**: Intelligent ticket routing and resolution assistance
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
- [Code of Conduct](https://github.com/langchain-ai/langchain/blob/master/.github/CODE_OF_CONDUCT.md) – Our community guidelines and standards for participation.
