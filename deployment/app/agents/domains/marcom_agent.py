"""Marketing & Communications Domain Agent.

Provides specialized support for:
- Marketing campaigns
- Brand management
- Content creation
- Social media
- Press releases
- Internal communications
"""

from langchain_core.tools import BaseTool, tool

from app.agents.domains.base_domain_agent import DomainAgent, DomainConfig, DomainType


@tool
def get_brand_guidelines() -> str:
    """Get current brand guidelines and assets."""
    return """Brand Guidelines Summary:
- Primary Colors: #1A73E8 (Blue), #34A853 (Green), #EA4335 (Red)
- Fonts: Roboto (headings), Open Sans (body)
- Logo usage: Minimum 40px height, clear space of 1x logo height
- Tone: Professional, friendly, innovative
- Contact brand@company.com for full brand kit"""


@tool
def search_marketing_assets(query: str) -> str:
    """Search for marketing assets like templates, images, logos.

    Args:
        query: Search query for assets.
    """
    return f"""Marketing Assets for '{query}':
1. Presentation Template - slides/template_2024.pptx
2. Email Banner - images/email_header.png
3. Social Media Kit - social/kit_q4.zip
4. Product Brochure - docs/product_brochure.pdf
Access via: marketing.company.com/assets"""


@tool
def check_campaign_status(campaign_name: str) -> str:
    """Check the status of a marketing campaign.

    Args:
        campaign_name: Name of the campaign to check.
    """
    return f"""Campaign Status: {campaign_name}
- Status: Active
- Start Date: 2024-01-15
- End Date: 2024-03-31
- Channels: Email, Social, Web
- Performance: 85% of target reached
- Budget Used: $45,000 / $60,000"""


@tool
def request_content_creation(content_type: str, description: str) -> str:
    """Submit a request for content creation.

    Args:
        content_type: Type of content (blog, social, email, etc.).
        description: Brief description of content needed.
    """
    return f"""Content Request Submitted:
- Type: {content_type}
- Description: {description}
- Request ID: CR-{hash(description) % 10000:04d}
- Estimated Turnaround: 3-5 business days
- Assigned To: Content Team
You'll receive an email confirmation shortly."""


class MarComAgent(DomainAgent):
    """Marketing & Communications specialist agent."""

    def get_config(self) -> DomainConfig:
        """Get MarCom configuration."""
        return DomainConfig(
            domain=DomainType.MARCOM,
            name="Marketing & Communications",
            description="Support for marketing campaigns, branding, content, and communications",
            expertise=[
                "brand guidelines",
                "marketing campaigns",
                "content creation",
                "social media",
                "press releases",
                "event marketing",
                "email campaigns",
                "advertising",
            ],
            escalation_keywords=[
                "crisis",
                "legal review",
                "competitor",
                "confidential",
                "executive approval",
            ],
            requires_approval=[
                "external press release",
                "crisis communication",
                "legal content",
            ],
        )

    def get_tools(self) -> list[BaseTool]:
        """Get MarCom tools."""
        return [
            get_brand_guidelines,
            search_marketing_assets,
            check_campaign_status,
            request_content_creation,
        ]

    def get_system_prompt(self) -> str:
        """Get MarCom system prompt."""
        return """You are the Marketing & Communications specialist for the IT support team.

Your expertise includes:
- Brand guidelines and identity
- Marketing campaigns and promotions
- Content creation and copywriting
- Social media management
- Press releases and media relations
- Internal and external communications
- Event marketing and coordination

When helping users:
1. Check brand guidelines for any branding questions
2. Search for existing assets before creating new ones
3. For new content requests, gather requirements before submitting
4. Provide timelines and set realistic expectations
5. Escalate anything involving legal review or crisis communications

Always maintain brand consistency and professional communication standards."""
