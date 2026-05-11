"""Finance domain agent for budget analysis, invoice processing, and expense management."""

from langchain_core.tools import BaseTool, tool

from app.agents.domains.base_domain_agent import DomainAgent, DomainConfig, DomainType


@tool
def analyze_budget(department: str, period: str) -> str:
    """Retrieve and analyze budget vs actuals for a department and period.

    Args:
        department: Department name (e.g., Engineering, Marketing).
        period: Fiscal period (e.g., Q1 2026, FY2025).
    """
    return (
        f"Budget Analysis — {department} | {period}\n"
        f"Budget: $500,000  |  Actuals: $423,800  |  Variance: +$76,200 (15.2% under)\n"
        f"Top spend categories: Salaries (68%), Software (14%), Travel (8%), Other (10%)\n"
        f"Status: On track. Full report at finance.company.com/budgets"
    )


@tool
def categorize_expense(description: str, amount: float) -> str:
    """Categorize an expense item into the correct GL account.

    Args:
        description: Expense description (e.g., 'AWS monthly invoice').
        amount: Expense amount in USD.
    """
    category_map = {
        "aws": "6120 – Cloud Infrastructure",
        "azure": "6120 – Cloud Infrastructure",
        "flight": "6210 – Travel & Accommodation",
        "hotel": "6210 – Travel & Accommodation",
        "software": "6130 – Software Licenses",
        "saas": "6130 – Software Licenses",
        "office": "6310 – Office Supplies",
    }
    category = "6900 – Miscellaneous"
    for keyword, gl in category_map.items():
        if keyword in description.lower():
            category = gl
            break
    return f"Expense: '{description}' (${amount:,.2f})\nGL Account: {category}\nSubmit at finance.company.com/expenses"


@tool
def get_invoice_status(invoice_id: str) -> str:
    """Check the payment status of a supplier invoice.

    Args:
        invoice_id: Invoice reference number.
    """
    return (
        f"Invoice {invoice_id}\n"
        f"Status: Approved — pending payment\n"
        f"Due Date: 30 days from receipt\n"
        f"Contact: ap@company.com for expedited processing"
    )


@tool
def submit_expense_report(employee_id: str, total_amount: float, description: str) -> str:
    """Submit an expense report for approval.

    Args:
        employee_id: Employee ID or 'self'.
        total_amount: Total expense amount in USD.
        description: Brief description of expenses.
    """
    report_id = f"EXP-{hash(description) % 100000:05d}"
    return (
        f"Expense Report Submitted\n"
        f"Report ID: {report_id}\n"
        f"Employee: {employee_id}  |  Amount: ${total_amount:,.2f}\n"
        f"Description: {description}\n"
        f"Approver notification sent. Expected review: 2 business days."
    )


class FinanceAgent(DomainAgent):
    """Finance and accounting support: budgets, invoices, expenses, financial reporting."""

    def get_config(self) -> DomainConfig:
        """Get Finance configuration."""
        return DomainConfig(
            domain=DomainType.FINANCE,
            name="Finance",
            description="Support for budgets, invoices, expense reports, and financial reporting",
            expertise=[
                "budget analysis",
                "expense management",
                "invoice processing",
                "accounts payable",
                "financial reporting",
                "GL coding",
                "cost centre allocation",
                "fiscal planning",
            ],
            escalation_keywords=[
                "fraud",
                "audit",
                "compliance",
                "tax",
                "legal",
                "overpayment",
                "discrepancy",
            ],
            requires_approval=[
                "budget exceptions",
                "invoice over threshold",
                "capital expenditure",
            ],
        )

    def get_tools(self) -> list[BaseTool]:
        """Get Finance tools."""
        return [
            analyze_budget,
            categorize_expense,
            get_invoice_status,
            submit_expense_report,
        ]

    def get_system_prompt(self) -> str:
        """Get Finance system prompt."""
        return """You are the Finance specialist for the enterprise platform.

Your expertise includes:
- Budget tracking and variance analysis
- Expense categorisation and GL coding
- Invoice status and accounts payable queries
- Expense report submission and approval workflows
- Financial policy guidance

When assisting users:
1. Always verify amounts and invoice numbers before confirming status
2. Escalate fraud, audit, or compliance concerns immediately
3. Direct tax and legal questions to the appropriate teams
4. Provide GL codes and cost centre guidance accurately
5. Explain financial policies clearly without giving specific tax advice

Be precise with numbers and reference IDs."""
