"""Enterprise IT Agents module.

This module provides a comprehensive suite of AI agents for IT operations:
- Research Agent: Information gathering and analysis
- Content Agent: Social media and blog content generation
- Data Analyst Agent: Excel/CSV data analysis
- Document Agent: SOP/WLI/Policy generation
- RAG Agent: Multilingual document Q&A
- IT Support Agent: Human-in-the-loop support
- Code Assistant: Application modernization

Following the 4-role Enterprise Development Standards from CLAUDE.md.
"""

# Existing agents
from app.agents.it_helpdesk import ITHelpdeskAgent
from app.agents.servicenow_agent import ServiceNowAgent
from app.agents.conversation_manager import ConversationManager

# Base framework
from app.agents.base import BaseAgent, AgentConfig

# New enterprise agents
from app.agents.research import ResearchAgent
from app.agents.content import ContentAgent
from app.agents.data_analyst import DataAnalystAgent
from app.agents.documents import DocumentAgent
from app.agents.rag import MultilingualRAGAgent
from app.agents.it_support import HITLSupportAgent
from app.agents.code_assistant import CodeAssistantAgent

# Tracing and evaluation
from app.agents.tracing import setup_tracing, get_tracing_status
from app.agents.evals import evaluate_agent_response

__all__ = [
    # Existing
    "ITHelpdeskAgent",
    "ServiceNowAgent",
    "ConversationManager",
    # Base
    "BaseAgent",
    "AgentConfig",
    # New agents
    "ResearchAgent",
    "ContentAgent",
    "DataAnalystAgent",
    "DocumentAgent",
    "MultilingualRAGAgent",
    "HITLSupportAgent",
    "CodeAssistantAgent",
    # Utilities
    "setup_tracing",
    "get_tracing_status",
    "evaluate_agent_response",
]
