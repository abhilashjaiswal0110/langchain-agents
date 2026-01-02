"""Data & AI Domain Agent.

Provides specialized support for:
- Data analytics
- Machine learning
- Data pipelines
- AI tools and platforms
- Business intelligence
"""

from langchain_core.tools import BaseTool, tool

from app.agents.domains.base_domain_agent import DomainAgent, DomainConfig, DomainType


@tool
def list_data_sources(category: str = "all") -> str:
    """List available data sources and datasets.

    Args:
        category: Category of data (sales, hr, finance, etc.).
    """
    return f"""Available Data Sources ({category}):
1. Sales Data Warehouse - Updated hourly
2. Customer Analytics DB - Real-time
3. HR Metrics Cube - Daily refresh
4. Financial Reports - Monthly snapshot
5. Web Analytics - Real-time streaming

Access via: data.company.com
Catalog: datacatalog.company.com"""


@tool
def check_pipeline_status(pipeline_name: str) -> str:
    """Check status of a data pipeline.

    Args:
        pipeline_name: Name of the data pipeline.
    """
    return f"""Pipeline Status: {pipeline_name}
- Status: Running
- Last Run: 2 hours ago (Success)
- Records Processed: 1.2M
- Duration: 45 minutes
- Next Run: In 4 hours
- Failures (7 days): 1
View logs: pipelines.company.com/{pipeline_name.lower()}"""


@tool
def request_data_access(dataset: str, purpose: str) -> str:
    """Request access to a dataset.

    Args:
        dataset: Name of the dataset.
        purpose: Purpose for data access.
    """
    return f"""Data Access Request Submitted:
- Dataset: {dataset}
- Purpose: {purpose}
- Request ID: DATA-{hash(dataset) % 10000:04d}
- Status: Pending data governance review
- Expected Timeline: 2-3 business days
You'll be notified once approved."""


@tool
def search_reports(query: str) -> str:
    """Search for existing reports and dashboards.

    Args:
        query: Search query for reports.
    """
    return f"""Reports matching '{query}':
1. {query.title()} Dashboard - Power BI
2. {query.title()} Weekly Report - Automated email
3. {query.title()} Trends Analysis - Tableau
4. Executive {query.title()} Summary - PDF

Access via: reports.company.com
Request new reports: analytics@company.com"""


@tool
def get_ml_model_info(model_name: str) -> str:
    """Get information about an ML model.

    Args:
        model_name: Name of the ML model.
    """
    return f"""ML Model: {model_name}
- Status: Production
- Version: v2.3.1
- Accuracy: 94.5%
- Last Training: Nov 2024
- API Endpoint: api.company.com/ml/{model_name.lower()}
- Documentation: docs.company.com/ml/{model_name.lower()}"""


class DataAIAgent(DomainAgent):
    """Data & AI specialist agent."""

    def get_config(self) -> DomainConfig:
        """Get Data/AI configuration."""
        return DomainConfig(
            domain=DomainType.DATA_AI,
            name="Data & AI",
            description="Support for analytics, ML, data pipelines, and AI tools",
            expertise=[
                "data analytics",
                "machine learning",
                "data pipelines",
                "business intelligence",
                "data governance",
                "ai tools",
                "reporting",
                "data science",
            ],
            escalation_keywords=[
                "pii",
                "sensitive data",
                "data breach",
                "production model",
            ],
            requires_approval=[
                "production model deployment",
                "pii access",
                "external data sharing",
            ],
        )

    def get_tools(self) -> list[BaseTool]:
        """Get Data/AI tools."""
        return [
            list_data_sources,
            check_pipeline_status,
            request_data_access,
            search_reports,
            get_ml_model_info,
        ]

    def get_system_prompt(self) -> str:
        """Get Data/AI system prompt."""
        return """You are the Data & AI specialist for the IT support team.

Your expertise includes:
- Data analytics and visualization
- Machine learning and AI models
- Data pipelines (ETL/ELT)
- Business intelligence tools (Power BI, Tableau)
- Data governance and quality
- AI platforms and tools
- Predictive analytics

When helping users:
1. Check data catalog for existing datasets
2. Verify data access permissions
3. Search for existing reports before creating new
4. Consider data privacy for all requests
5. Document data lineage and transformations

Empower users with data-driven insights."""
