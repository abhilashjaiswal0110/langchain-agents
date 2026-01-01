"""Cybersecurity Domain Agent.

Provides specialized support for:
- Security incidents
- Vulnerability management
- Access control
- Compliance
- Security awareness
"""

from langchain_core.tools import BaseTool, tool

from app.agents.domains.base_domain_agent import DomainAgent, DomainConfig, DomainType


@tool
def report_security_incident(incident_type: str, description: str) -> str:
    """Report a security incident.

    Args:
        incident_type: Type of incident (phishing, malware, breach, etc.).
        description: Description of the incident.
    """
    return f"""Security Incident Reported:
- Type: {incident_type.upper()}
- Description: {description}
- Incident ID: SEC-{hash(description) % 10000:04d}
- Priority: HIGH
- Status: Under Investigation
- Response Team: Notified

IMMEDIATE ACTIONS:
1. Do not click any suspicious links
2. Do not share credentials
3. Preserve any evidence
4. Security team will contact you shortly"""


@tool
def check_vulnerability_status(system: str) -> str:
    """Check vulnerability status for a system.

    Args:
        system: System or application name.
    """
    return f"""Vulnerability Status: {system}
- Last Scan: 2 days ago
- Critical: 0
- High: 2 (patches pending)
- Medium: 5
- Low: 12
- Compliance: 94%
Next scan: Scheduled in 5 days"""


@tool
def request_access(resource: str, justification: str) -> str:
    """Request access to a resource.

    Args:
        resource: Resource or system name.
        justification: Business justification.
    """
    return f"""Access Request Submitted:
- Resource: {resource}
- Justification: {justification}
- Request ID: ACC-{hash(resource) % 10000:04d}
- Status: Pending security review
- Expected Timeline: 1-2 business days
Approval workflow initiated."""


@tool
def check_compliance_status(framework: str = "all") -> str:
    """Check compliance status for security frameworks.

    Args:
        framework: Framework name (SOC2, ISO27001, GDPR, etc.).
    """
    frameworks = {
        "soc2": "SOC 2: 98% compliant - 2 findings in remediation",
        "iso27001": "ISO 27001: Certified (expires Dec 2025)",
        "gdpr": "GDPR: Compliant - last audit March 2024",
        "pci": "PCI DSS: Not applicable",
    }
    if framework.lower() == "all":
        return "\n".join(frameworks.values())
    return frameworks.get(framework.lower(), f"Status unknown for {framework}")


@tool
def security_training_status(employee_id: str = "self") -> str:
    """Check security awareness training status.

    Args:
        employee_id: Employee ID or 'self'.
    """
    return """Security Training Status:
- Security Awareness 2024: Completed
- Phishing Simulation: Passed (last: Oct 2024)
- Data Handling: Completed
- GDPR Training: Completed
- Next Required: Q1 2025 Refresh"""


class CybersecurityAgent(DomainAgent):
    """Cybersecurity specialist agent."""

    def get_config(self) -> DomainConfig:
        """Get Cybersecurity configuration."""
        return DomainConfig(
            domain=DomainType.CYBERSECURITY,
            name="Cybersecurity",
            description="Support for security incidents, vulnerabilities, access control, and compliance",
            expertise=[
                "security incidents",
                "vulnerability management",
                "access control",
                "compliance",
                "phishing",
                "malware",
                "encryption",
                "identity management",
            ],
            escalation_keywords=[
                "breach",
                "ransomware",
                "data leak",
                "compromised",
                "hack",
                "urgent security",
            ],
            requires_approval=[
                "privileged access",
                "exception requests",
                "security bypasses",
            ],
        )

    def get_tools(self) -> list[BaseTool]:
        """Get Cybersecurity tools."""
        return [
            report_security_incident,
            check_vulnerability_status,
            request_access,
            check_compliance_status,
            security_training_status,
        ]

    def get_system_prompt(self) -> str:
        """Get Cybersecurity system prompt."""
        return """You are the Cybersecurity specialist for the IT support team.

Your expertise includes:
- Security incident response
- Vulnerability management and patching
- Access control and identity management
- Compliance (SOC 2, ISO 27001, GDPR)
- Security awareness training
- Threat detection and prevention
- Encryption and data protection

When helping users:
1. ALWAYS escalate active security incidents
2. Never share sensitive security details in chat
3. Verify identity before granting any access
4. Document all security-related requests
5. Promote security awareness

Security is everyone's responsibility. Stay vigilant."""
