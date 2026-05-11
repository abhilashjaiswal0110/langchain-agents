"""AI Research Agent for comprehensive information gathering.

This module provides research capabilities:
- ResearchAgent: Basic research with web search
- DeepSearchAgent: Advanced multi-step research with planning
- ResearchPlanner: Query decomposition
- SourceManager: Citation tracking and credibility scoring
- SearchProviderManager: Multi-provider search abstraction

Example:
    >>> from app.agents.research import DeepSearchAgent, ResearchDepth
    >>> agent = DeepSearchAgent()
    >>> report = await agent.research("AI agents", depth=ResearchDepth.COMPREHENSIVE)
    >>> print(report.to_markdown())
"""

# Basic research agent
# Deep search agent
from app.agents.research.deep_search_agent import (
    DeepSearchAgent,
    ResearchDepth,
    ResearchFinding,
    ResearchReport,
    deep_research,
    get_deep_search_agent,
    quick_search,
    reset_deep_search_agent,
)

# Query planning
from app.agents.research.planner import (
    ExecutionStrategy,
    QueryIntent,
    ResearchPlan,
    ResearchPlanner,
    SubQuery,
    create_research_plan,
)
from app.agents.research.research_agent import ResearchAgent, ResearchState

# Search providers
from app.agents.research.search_providers import (
    DuckDuckGoSearchProvider,
    SearchProvider,
    SearchProviderManager,
    SearchProviderType,
    SearchResponse,
    SearchResult,
    SimulatedSearchProvider,
    TavilySearchProvider,
    get_search_manager,
    reset_search_manager,
    search,
)

# Source management
from app.agents.research.source_manager import (
    CitationFormat,
    CitationFormatter,
    CredibilityLevel,
    CredibilityScorer,
    Source,
    SourceCollection,
    SourceManager,
    SourceType,
    get_source_manager,
    reset_source_manager,
)

__all__ = [
    # Basic research
    "ResearchAgent",
    "ResearchState",
    # Query planning
    "ResearchPlanner",
    "ResearchPlan",
    "SubQuery",
    "QueryIntent",
    "ExecutionStrategy",
    "create_research_plan",
    # Source management
    "SourceManager",
    "Source",
    "SourceCollection",
    "SourceType",
    "CredibilityLevel",
    "CredibilityScorer",
    "CitationFormatter",
    "CitationFormat",
    "get_source_manager",
    "reset_source_manager",
    # Search providers
    "SearchProviderManager",
    "SearchProvider",
    "SearchProviderType",
    "SearchResult",
    "SearchResponse",
    "TavilySearchProvider",
    "DuckDuckGoSearchProvider",
    "SimulatedSearchProvider",
    "get_search_manager",
    "reset_search_manager",
    "search",
    # Deep search agent
    "DeepSearchAgent",
    "ResearchDepth",
    "ResearchReport",
    "ResearchFinding",
    "get_deep_search_agent",
    "reset_deep_search_agent",
    "deep_research",
    "quick_search",
]
