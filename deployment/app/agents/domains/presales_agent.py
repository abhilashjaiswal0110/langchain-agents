"""Presales/Sales Support Domain Agent.

Provides specialized support for:
- Product demos
- Proposals and RFPs
- Pricing inquiries
- Customer presentations
- Sales collateral
"""

from langchain_core.tools import BaseTool, tool

from app.agents.domains.base_domain_agent import DomainAgent, DomainConfig, DomainType


@tool
def search_sales_collateral(product: str) -> str:
    """Search for sales collateral and presentations.

    Args:
        product: Product or solution name.
    """
    return f"""Sales Collateral for '{product}':
1. Product Overview Deck - presentations/{product.lower()}_overview.pptx
2. Competitive Analysis - docs/{product.lower()}_vs_competition.pdf
3. Case Studies - casestudies/{product.lower()}/
4. Pricing Sheet - pricing/{product.lower()}_msrp.xlsx
Access via: sales.company.com/collateral"""


@tool
def schedule_demo(product: str, customer: str, preferred_date: str) -> str:
    """Schedule a product demo with the presales team.

    Args:
        product: Product to demo.
        customer: Customer/prospect name.
        preferred_date: Preferred date for demo.
    """
    return f"""Demo Request Submitted:
- Product: {product}
- Customer: {customer}
- Preferred Date: {preferred_date}
- Request ID: DEMO-{hash(customer) % 10000:04d}
- Status: Pending SE assignment
A sales engineer will contact you within 24 hours."""


@tool
def get_pricing_info(product: str, quantity: int = 1) -> str:
    """Get pricing information for a product.

    Args:
        product: Product name.
        quantity: Number of licenses/units.
    """
    return f"""Pricing Estimate for {product} (x{quantity}):
- List Price: Contact sales for current pricing
- Volume Discount: Available for 100+ units
- Enterprise Pricing: Custom quotes available
- Request Quote: sales@company.com
Note: All pricing subject to approval."""


@tool
def search_rfp_responses(topic: str) -> str:
    """Search for previous RFP responses.

    Args:
        topic: Topic or product for RFP search.
    """
    return f"""RFP Resources for '{topic}':
1. RFP Template - templates/rfp_template.docx
2. Standard Responses - rfp/{topic.lower()}_responses.docx
3. Technical Specs - specs/{topic.lower()}_tech.pdf
4. Security Questionnaire - security/standard_questionnaire.xlsx
Access via: sales.company.com/rfp"""


class PresalesAgent(DomainAgent):
    """Presales/Sales Support specialist agent."""

    def get_config(self) -> DomainConfig:
        """Get Presales configuration."""
        return DomainConfig(
            domain=DomainType.PRESALES,
            name="Presales/Sales Support",
            description="Support for demos, proposals, pricing, and customer engagements",
            expertise=[
                "product demos",
                "rfp responses",
                "pricing",
                "proposals",
                "sales presentations",
                "competitive analysis",
                "customer engagements",
                "poc support",
            ],
            escalation_keywords=[
                "discount",
                "negotiation",
                "contract",
                "legal terms",
                "enterprise deal",
            ],
            requires_approval=[
                "custom pricing",
                "contract modifications",
                "enterprise discounts",
            ],
        )

    def get_tools(self) -> list[BaseTool]:
        """Get Presales tools."""
        return [
            search_sales_collateral,
            schedule_demo,
            get_pricing_info,
            search_rfp_responses,
        ]

    def get_system_prompt(self) -> str:
        """Get Presales system prompt."""
        return """You are the Presales/Sales Support specialist for the IT support team.

Your expertise includes:
- Product demonstrations and presentations
- RFP/RFI response preparation
- Pricing and quoting (within guidelines)
- Sales collateral and case studies
- Competitive positioning
- Customer engagement support
- Proof of concept coordination

When helping users:
1. Search for existing collateral before creating new
2. Coordinate demo requests with sales engineering
3. Direct pricing negotiations to sales managers
4. Help with RFP responses using standard templates
5. Maintain confidentiality of customer information

Support the sales team in winning deals professionally."""
