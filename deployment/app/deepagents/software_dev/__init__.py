"""Software Development DeepAgent.

A comprehensive multi-agent system for end-to-end software development lifecycle (SDLC)
automation. This agent coordinates specialized subagents to handle:

- Requirements analysis and refinement
- Architecture and design
- Code generation and refactoring
- Code review and quality assurance
- Testing automation
- Security scanning and compliance
- CI/CD integration
- Debugging and optimization
- Documentation generation
"""

from app.deepagents.software_dev.state import SoftwareDevState
from app.deepagents.software_dev.software_dev_agent import (
    SoftwareDevDeepAgent,
    create_software_dev_agent,
)

__all__ = [
    "SoftwareDevState",
    "SoftwareDevDeepAgent",
    "create_software_dev_agent",
]
