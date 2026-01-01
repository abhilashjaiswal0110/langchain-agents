"""Human Resources Domain Agent.

Provides specialized support for:
- Benefits information
- HR policies
- Payroll inquiries
- Onboarding/offboarding
- Performance management
- Employee relations
"""

from langchain_core.tools import BaseTool, tool

from app.agents.domains.base_domain_agent import DomainAgent, DomainConfig, DomainType


@tool
def get_benefits_info(benefit_type: str) -> str:
    """Get information about employee benefits.

    Args:
        benefit_type: Type of benefit (health, dental, vision, 401k, pto).
    """
    benefits = {
        "health": "Medical: PPO and HMO options. Coverage starts day 1. Family coverage available.",
        "dental": "Dental: Preventive 100%, Basic 80%, Major 50%. Annual max $2,000.",
        "vision": "Vision: Annual exam covered. $200 allowance for frames/lenses.",
        "401k": "401(k): Company matches 100% up to 4%, then 50% up to 6%. Vesting: 3 years.",
        "pto": "PTO: 15 days year 1, 20 days years 2-4, 25 days year 5+. Plus 10 holidays.",
    }
    return benefits.get(benefit_type.lower(), f"Contact HR for details on {benefit_type}")


@tool
def check_pto_balance(employee_id: str) -> str:
    """Check PTO balance for an employee.

    Args:
        employee_id: Employee ID or 'self' for current user.
    """
    return """PTO Balance:
- Vacation Days: 12 remaining
- Sick Days: 5 remaining
- Personal Days: 2 remaining
- Total Used YTD: 8 days
View full details at: hr.company.com/pto"""


@tool
def lookup_policy(policy_name: str) -> str:
    """Look up an HR policy.

    Args:
        policy_name: Name or topic of the policy.
    """
    return f"""Policy: {policy_name}
Summary: This policy outlines guidelines for {policy_name.lower()}.
Full policy available at: hr.company.com/policies
For specific questions, contact HR at hr@company.com"""


@tool
def submit_hr_request(request_type: str, details: str) -> str:
    """Submit an HR request.

    Args:
        request_type: Type of request (leave, transfer, accommodation, etc.).
        details: Details of the request.
    """
    return f"""HR Request Submitted:
- Type: {request_type}
- Details: {details}
- Request ID: HR-{hash(details) % 10000:04d}
- Expected Response: 2-3 business days
You'll receive an email confirmation shortly."""


class HRAgent(DomainAgent):
    """Human Resources specialist agent."""

    def get_config(self) -> DomainConfig:
        """Get HR configuration."""
        return DomainConfig(
            domain=DomainType.HR,
            name="Human Resources",
            description="Support for benefits, policies, payroll, and employee relations",
            expertise=[
                "employee benefits",
                "hr policies",
                "payroll",
                "time off",
                "onboarding",
                "performance reviews",
                "employee relations",
                "compliance",
            ],
            escalation_keywords=[
                "harassment",
                "discrimination",
                "termination",
                "legal",
                "complaint",
                "grievance",
                "confidential",
                "salary",
                "raise",
            ],
            requires_approval=[
                "salary changes",
                "promotions",
                "terminations",
                "policy exceptions",
            ],
        )

    def get_tools(self) -> list[BaseTool]:
        """Get HR tools."""
        return [
            get_benefits_info,
            check_pto_balance,
            lookup_policy,
            submit_hr_request,
        ]

    def get_system_prompt(self) -> str:
        """Get HR system prompt."""
        return """You are the Human Resources specialist for the IT support team.

Your expertise includes:
- Employee benefits (health, dental, vision, 401k, PTO)
- HR policies and procedures
- Payroll inquiries
- Onboarding and offboarding processes
- Performance management
- Employee relations
- Compliance and regulations

When helping users:
1. Protect employee confidentiality at all times
2. Direct salary/compensation questions to HR directly
3. Escalate any harassment, discrimination, or legal concerns
4. Provide policy information but not legal advice
5. Guide users to the appropriate HR forms and systems

Be empathetic and professional. Many HR matters are sensitive."""
