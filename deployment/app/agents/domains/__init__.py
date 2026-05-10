"""Domain-Specific Agents for Enterprise IT Support.

Provides specialized agents for different business domains:
- MarCom: Marketing & Communications
- HR: Human Resources
- L&D: Learning & Development
- Presales: Presales/Sales Support
- Datacenter: Datacenter Operations
- Cloud: Cloud Infrastructure
- Cybersecurity: Security Operations
- Data/AI: Data & AI Support

Each agent extends the base DomainAgent class and provides
domain-specific tools and knowledge.

Following Enterprise Development Standards:
- Software Architect: Domain-driven design
- Software Engineer: Consistent agent interface
"""

from app.agents.domains.base_domain_agent import (
    DomainAgent,
    DomainConfig,
    DomainType,
    create_domain_agent,
)
from app.agents.domains.marcom_agent import MarComAgent
from app.agents.domains.hr_agent import HRAgent
from app.agents.domains.lnd_agent import LnDAgent
from app.agents.domains.presales_agent import PresalesAgent
from app.agents.domains.datacenter_agent import DatacenterAgent
from app.agents.domains.cloud_agent import CloudAgent
from app.agents.domains.cybersecurity_agent import CybersecurityAgent
from app.agents.domains.data_ai_agent import DataAIAgent
from app.agents.domains.finance_agent import FinanceAgent

__all__ = [
    # Base class
    "DomainAgent",
    "DomainConfig",
    "DomainType",
    "create_domain_agent",
    # Domain agents
    "MarComAgent",
    "HRAgent",
    "LnDAgent",
    "PresalesAgent",
    "DatacenterAgent",
    "CloudAgent",
    "CybersecurityAgent",
    "DataAIAgent",
    "FinanceAgent",
]


def get_all_domain_agents() -> dict[str, DomainAgent]:
    """Get all domain agent instances.

    Returns:
        Dictionary of domain name to agent instance.
    """
    return {
        "marcom": MarComAgent(),
        "hr": HRAgent(),
        "lnd": LnDAgent(),
        "presales": PresalesAgent(),
        "datacenter": DatacenterAgent(),
        "cloud": CloudAgent(),
        "cybersecurity": CybersecurityAgent(),
        "data_ai": DataAIAgent(),
        "finance": FinanceAgent(),
    }
