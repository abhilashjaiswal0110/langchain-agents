"""Enterprise IT Agents module.

This module provides a comprehensive suite of AI agents for IT operations:
- Research Agent: Information gathering and analysis
- Content Agent: Social media and blog content generation
- Data Analyst Agent: Excel/CSV data analysis
- Document Agent: SOP/WLI/Policy generation
- RAG Agent: Multilingual document Q&A
- IT Support Agent: Human-in-the-loop support
- Code Assistant: Application modernization
- Employee Experience Agent: HR support, career development, and wellbeing

Following the 4-role Enterprise Development Standards from CLAUDE.md.
"""

# IT Support agents (classes only - no global instances)
# Base framework
from app.agents.base import AgentConfig, BaseAgent
from app.agents.code_assistant import CodeAssistantAgent
from app.agents.content import ContentAgent
from app.agents.conversation_manager import ConversationManager
from app.agents.data_analyst import DataAnalystAgent
from app.agents.document_intelligence import DocumentIntelligenceAgent
from app.agents.documents import DocumentAgent
from app.agents.employee_experience import EmployeeExperienceAgent
from app.agents.evals import evaluate_agent_response
from app.agents.it_helpdesk import ITHelpdeskAgent
from app.agents.it_support import HITLSupportAgent
from app.agents.rag import MultilingualRAGAgent

# New enterprise agents
from app.agents.research import ResearchAgent
from app.agents.servicenow_agent import ServiceNowAgent

# Tracing and evaluation
from app.agents.tracing import get_tracing_status, setup_tracing

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
    "DocumentIntelligenceAgent",
    "EmployeeExperienceAgent",
    # Utilities
    "setup_tracing",
    "get_tracing_status",
    "evaluate_agent_response",
]
