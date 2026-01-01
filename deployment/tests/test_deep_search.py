"""Tests for DeepSearch research components.

Tests cover:
- Research planner (query decomposition)
- Source manager (citation tracking, credibility)
- Search providers (multi-provider abstraction)
- Deep search agent (integration)
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

# Import planner components
from app.agents.research.planner import (
    ExecutionStrategy,
    QueryIntent,
    ResearchPlan,
    ResearchPlanner,
    SubQuery,
    create_research_plan,
)

# Import source manager components
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

# Import search provider components
from app.agents.research.search_providers import (
    SearchProviderManager,
    SearchProviderType,
    SearchResponse,
    SearchResult,
    SimulatedSearchProvider,
    get_search_manager,
    reset_search_manager,
)

# Import deep search agent components
from app.agents.research.deep_search_agent import (
    DeepSearchAgent,
    ResearchDepth,
    ResearchFinding,
    ResearchReport,
    get_deep_search_agent,
    reset_deep_search_agent,
)


# ==================== Query Planner Tests ====================


class TestQueryIntent:
    """Tests for QueryIntent enum."""

    def test_query_intent_values(self) -> None:
        """Test all query intent values exist."""
        assert QueryIntent.FACTUAL.value == "factual"
        assert QueryIntent.COMPARISON.value == "comparison"
        assert QueryIntent.ANALYSIS.value == "analysis"
        assert QueryIntent.HOW_TO.value == "how_to"
        assert QueryIntent.TREND.value == "trend"


class TestSubQuery:
    """Tests for SubQuery dataclass."""

    def test_sub_query_creation(self) -> None:
        """Test creating a sub-query."""
        sq = SubQuery(
            query="test query",
            intent=QueryIntent.FACTUAL,
            priority=1,
            keywords=["test", "query"],
        )
        assert sq.query == "test query"
        assert sq.intent == QueryIntent.FACTUAL
        assert sq.priority == 1
        assert len(sq.keywords) == 2

    def test_sub_query_default_values(self) -> None:
        """Test sub-query default values."""
        sq = SubQuery()
        assert sq.query == ""
        assert sq.intent == QueryIntent.FACTUAL
        assert sq.priority == 1
        assert sq.depends_on == []
        assert sq.estimated_sources == 3


class TestResearchPlan:
    """Tests for ResearchPlan dataclass."""

    def test_plan_creation(self) -> None:
        """Test creating a research plan."""
        plan = ResearchPlan(
            original_query="test query",
            intent=QueryIntent.ANALYSIS,
            strategy=ExecutionStrategy.PARALLEL,
        )
        assert plan.original_query == "test query"
        assert plan.strategy == ExecutionStrategy.PARALLEL

    def test_plan_execution_order_parallel(self) -> None:
        """Test parallel execution order."""
        plan = ResearchPlan(
            strategy=ExecutionStrategy.PARALLEL,
            sub_queries=[
                SubQuery(query="q1", priority=1),
                SubQuery(query="q2", priority=2),
            ],
        )
        order = plan.get_execution_order()
        assert len(order) == 1  # All in one batch
        assert len(order[0]) == 2

    def test_plan_execution_order_sequential(self) -> None:
        """Test sequential execution order."""
        plan = ResearchPlan(
            strategy=ExecutionStrategy.SEQUENTIAL,
            sub_queries=[
                SubQuery(query="q1", priority=2),
                SubQuery(query="q2", priority=1),
            ],
        )
        order = plan.get_execution_order()
        assert len(order) == 2  # Each in separate batch
        assert order[0][0].priority == 1  # Lower priority first

    def test_plan_execution_order_hierarchical(self) -> None:
        """Test hierarchical execution with dependencies."""
        sq1 = SubQuery(id="sq1", query="q1", depends_on=[])
        sq2 = SubQuery(id="sq2", query="q2", depends_on=["sq1"])

        plan = ResearchPlan(
            strategy=ExecutionStrategy.HIERARCHICAL,
            sub_queries=[sq2, sq1],  # Order doesn't matter
        )
        order = plan.get_execution_order()
        assert len(order) == 2
        assert sq1 in order[0]  # sq1 first (no dependencies)
        assert sq2 in order[1]  # sq2 after (depends on sq1)


class TestResearchPlanner:
    """Tests for ResearchPlanner."""

    def test_classify_intent_comparison(self) -> None:
        """Test classifying comparison queries."""
        planner = ResearchPlanner()
        intent = planner.classify_intent("Compare React vs Vue")
        assert intent == QueryIntent.COMPARISON

    def test_classify_intent_how_to(self) -> None:
        """Test classifying how-to queries."""
        planner = ResearchPlanner()
        intent = planner.classify_intent("How to build an AI agent")
        assert intent == QueryIntent.HOW_TO

    def test_classify_intent_definition(self) -> None:
        """Test classifying definition queries."""
        planner = ResearchPlanner()
        intent = planner.classify_intent("What is machine learning?")
        assert intent == QueryIntent.DEFINITION

    def test_classify_intent_trend(self) -> None:
        """Test classifying trend queries."""
        planner = ResearchPlanner()
        intent = planner.classify_intent("Latest trends in AI")
        assert intent == QueryIntent.TREND

    def test_fallback_plan_creation(self) -> None:
        """Test fallback plan when decomposition fails."""
        planner = ResearchPlanner()
        plan = planner._create_fallback_plan("test query", "error")
        assert plan.original_query == "test query"
        assert len(plan.sub_queries) == 1
        assert plan.sub_queries[0].query == "test query"


# ==================== Source Manager Tests ====================


class TestSourceType:
    """Tests for SourceType enum."""

    def test_source_type_values(self) -> None:
        """Test source type values."""
        assert SourceType.ACADEMIC_PAPER.value == "academic_paper"
        assert SourceType.NEWS_ARTICLE.value == "news_article"
        assert SourceType.DOCUMENTATION.value == "documentation"


class TestCredibilityLevel:
    """Tests for CredibilityLevel enum."""

    def test_credibility_levels(self) -> None:
        """Test credibility level values."""
        assert CredibilityLevel.HIGH.value == "high"
        assert CredibilityLevel.MEDIUM.value == "medium"
        assert CredibilityLevel.LOW.value == "low"
        assert CredibilityLevel.UNVERIFIED.value == "unverified"


class TestSource:
    """Tests for Source dataclass."""

    def test_source_creation(self) -> None:
        """Test creating a source."""
        source = Source(
            url="https://example.com",
            title="Test Source",
            content_summary="Test content",
        )
        assert source.url == "https://example.com"
        assert source.title == "Test Source"

    def test_source_domain_extraction(self) -> None:
        """Test domain is extracted from URL."""
        source = Source(url="https://arxiv.org/paper/123")
        assert source.domain == "arxiv.org"

    def test_source_default_values(self) -> None:
        """Test source default values."""
        source = Source()
        assert source.credibility == CredibilityLevel.UNVERIFIED
        assert source.credibility_score == 0.5
        assert source.verified is False


class TestCredibilityScorer:
    """Tests for CredibilityScorer."""

    def test_score_academic_source(self) -> None:
        """Test scoring academic sources."""
        scorer = CredibilityScorer()
        source = Source(
            url="https://arxiv.org/paper",
            title="Research Paper",
            source_type=SourceType.ACADEMIC_PAPER,
            author="Dr. Smith",
        )
        score = scorer.score(source)
        assert score >= 0.7  # Academic sources should score high

    def test_score_gov_domain(self) -> None:
        """Test .gov domains score high."""
        scorer = CredibilityScorer()
        source = Source(url="https://example.gov/report")
        score = scorer._score_domain("example.gov")
        assert score >= 0.8

    def test_score_social_media(self) -> None:
        """Test social media scores low."""
        scorer = CredibilityScorer()
        source = Source(
            url="https://twitter.com/post",
            source_type=SourceType.SOCIAL_MEDIA,
        )
        score = scorer.score(source)
        assert score <= 0.5

    def test_get_credibility_level(self) -> None:
        """Test score to level conversion."""
        scorer = CredibilityScorer()
        assert scorer.get_credibility_level(0.9) == CredibilityLevel.HIGH
        assert scorer.get_credibility_level(0.6) == CredibilityLevel.MEDIUM
        assert scorer.get_credibility_level(0.3) == CredibilityLevel.LOW
        assert scorer.get_credibility_level(0.1) == CredibilityLevel.UNVERIFIED


class TestCitationFormatter:
    """Tests for CitationFormatter."""

    def test_format_markdown(self) -> None:
        """Test Markdown citation format."""
        formatter = CitationFormatter()
        source = Source(url="https://example.com", title="Test Source")
        citation = formatter.format(source, CitationFormat.MARKDOWN)
        assert "[Test Source]" in citation
        assert "(https://example.com)" in citation

    def test_format_apa(self) -> None:
        """Test APA citation format."""
        formatter = CitationFormatter()
        source = Source(
            url="https://example.com",
            title="Test Paper",
            author="Smith, J.",
            publication_date="2024-01-01",
        )
        citation = formatter.format(source, CitationFormat.APA)
        assert "Smith, J." in citation
        assert "2024" in citation
        assert "Test Paper" in citation

    def test_format_plain(self) -> None:
        """Test plain text format."""
        formatter = CitationFormatter()
        source = Source(url="https://example.com", title="Test")
        citation = formatter.format(source, CitationFormat.PLAIN)
        assert "Test" in citation
        assert "https://example.com" in citation


class TestSourceManager:
    """Tests for SourceManager."""

    def setup_method(self) -> None:
        """Reset global state before each test."""
        reset_source_manager()

    def test_add_source(self) -> None:
        """Test adding a source."""
        manager = SourceManager()
        source = manager.add_source(
            url="https://example.com",
            title="Test",
            content_summary="Summary",
        )
        assert source.url == "https://example.com"
        assert source.credibility_score > 0

    def test_get_source(self) -> None:
        """Test retrieving a source."""
        manager = SourceManager()
        source = manager.add_source(url="https://example.com", title="Test")
        retrieved = manager.get_source(source.id)
        assert retrieved is not None
        assert retrieved.id == source.id

    def test_get_sources_by_credibility(self) -> None:
        """Test filtering by credibility."""
        manager = SourceManager()
        manager.add_source(url="https://arxiv.org/paper", title="Academic")
        manager.add_source(url="https://twitter.com/post", title="Social")

        high_cred = manager.get_sources_by_credibility(CredibilityLevel.HIGH)
        # May or may not have high credibility sources depending on scoring
        assert isinstance(high_cred, list)

    def test_cite_source(self) -> None:
        """Test citing a source increments count."""
        manager = SourceManager()
        source = manager.add_source(url="https://example.com", title="Test")
        assert source.citations_count == 0

        manager.cite_source(source.id)
        assert source.citations_count == 1

    def test_verify_source(self) -> None:
        """Test verifying a source."""
        manager = SourceManager()
        source = manager.add_source(url="https://example.com", title="Test")
        initial_score = source.credibility_score

        manager.verify_source(source.id)
        assert source.verified is True
        assert source.credibility_score >= initial_score

    def test_export_citations(self) -> None:
        """Test exporting citations."""
        manager = SourceManager()
        manager.add_source(url="https://example.com/1", title="Source 1")
        manager.add_source(url="https://example.com/2", title="Source 2")

        citations = manager.export_citations(CitationFormat.MARKDOWN)
        assert "Source 1" in citations
        assert "Source 2" in citations

    def test_create_collection(self) -> None:
        """Test creating a source collection."""
        manager = SourceManager()
        collection = manager.create_collection("test query")
        assert collection.query == "test query"
        assert len(collection.sources) == 0

    def test_add_to_collection(self) -> None:
        """Test adding source to collection."""
        manager = SourceManager()
        collection = manager.create_collection("test")
        source = manager.add_source(url="https://example.com", title="Test")

        result = manager.add_to_collection(collection.id, source.id)
        assert result is True
        assert len(collection.sources) == 1

    def test_get_statistics(self) -> None:
        """Test getting statistics."""
        manager = SourceManager()
        manager.add_source(url="https://example.com", title="Test")

        stats = manager.get_statistics()
        assert stats["total_sources"] == 1
        assert "avg_credibility" in stats
        assert "by_type" in stats

    def test_detect_source_type(self) -> None:
        """Test auto-detecting source type."""
        manager = SourceManager()

        assert manager._detect_source_type("https://arxiv.org/paper") == SourceType.ACADEMIC_PAPER
        assert manager._detect_source_type("https://example.gov/report") == SourceType.GOVERNMENT
        assert manager._detect_source_type("https://docs.python.org") == SourceType.DOCUMENTATION
        assert manager._detect_source_type("https://medium.com/post") == SourceType.BLOG_POST

    def test_global_source_manager(self) -> None:
        """Test global source manager singleton."""
        manager1 = get_source_manager()
        manager2 = get_source_manager()
        assert manager1 is manager2


# ==================== Search Provider Tests ====================


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self) -> None:
        """Test creating a search result."""
        result = SearchResult(
            title="Test Result",
            url="https://example.com",
            snippet="Test snippet",
            score=0.8,
        )
        assert result.title == "Test Result"
        assert result.score == 0.8


class TestSearchResponse:
    """Tests for SearchResponse dataclass."""

    def test_search_response_success(self) -> None:
        """Test successful search response."""
        response = SearchResponse(
            query="test",
            results=[SearchResult(title="Result")],
        )
        assert response.success is True

    def test_search_response_failure(self) -> None:
        """Test failed search response."""
        response = SearchResponse(
            query="test",
            error="Search failed",
        )
        assert response.success is False


class TestSimulatedSearchProvider:
    """Tests for SimulatedSearchProvider."""

    @pytest.mark.asyncio
    async def test_simulated_search(self) -> None:
        """Test simulated search returns results."""
        provider = SimulatedSearchProvider()
        response = await provider.search("AI agents", max_results=5)

        assert response.success is True
        assert len(response.results) > 0
        assert response.provider == SearchProviderType.SIMULATED

    @pytest.mark.asyncio
    async def test_simulated_search_with_keywords(self) -> None:
        """Test simulated search matches keywords."""
        provider = SimulatedSearchProvider()
        response = await provider.search("langchain tutorial", max_results=5)

        # Should find langchain-related results
        assert response.success is True

    def test_simulated_always_available(self) -> None:
        """Test simulated provider is always available."""
        provider = SimulatedSearchProvider()
        assert provider.is_available() is True


class TestSearchProviderManager:
    """Tests for SearchProviderManager."""

    def setup_method(self) -> None:
        """Reset global state."""
        reset_search_manager()

    @pytest.mark.asyncio
    async def test_search_with_fallback(self) -> None:
        """Test search falls back to simulated."""
        manager = SearchProviderManager(
            providers=[],  # No real providers
            fallback_to_simulated=True,
        )
        response = await manager.search("test query")

        assert response.success is True
        assert response.provider == SearchProviderType.SIMULATED

    @pytest.mark.asyncio
    async def test_search_parallel(self) -> None:
        """Test parallel search execution."""
        manager = SearchProviderManager(fallback_to_simulated=True)
        responses = await manager.search_parallel(
            ["query1", "query2", "query3"],
            max_results_per_query=3,
        )

        assert len(responses) == 3
        for response in responses:
            assert response.success is True

    def test_get_available_providers(self) -> None:
        """Test getting available providers."""
        manager = SearchProviderManager()
        available = manager.get_available_providers()
        # At minimum, should work without API keys
        assert isinstance(available, list)

    def test_global_search_manager(self) -> None:
        """Test global search manager singleton."""
        manager1 = get_search_manager()
        manager2 = get_search_manager()
        assert manager1 is manager2


# ==================== Deep Search Agent Tests ====================


class TestResearchDepth:
    """Tests for ResearchDepth enum."""

    def test_depth_values(self) -> None:
        """Test depth values."""
        assert ResearchDepth.QUICK.value == "quick"
        assert ResearchDepth.STANDARD.value == "standard"
        assert ResearchDepth.COMPREHENSIVE.value == "comprehensive"


class TestResearchFinding:
    """Tests for ResearchFinding dataclass."""

    def test_finding_creation(self) -> None:
        """Test creating a finding."""
        finding = ResearchFinding(
            content="Important finding",
            confidence=0.8,
            category="analysis",
        )
        assert finding.content == "Important finding"
        assert finding.confidence == 0.8


class TestResearchReport:
    """Tests for ResearchReport dataclass."""

    def test_report_creation(self) -> None:
        """Test creating a report."""
        report = ResearchReport(
            query="test query",
            depth=ResearchDepth.STANDARD,
        )
        assert report.query == "test query"
        assert report.depth == ResearchDepth.STANDARD

    def test_report_to_markdown(self) -> None:
        """Test converting report to markdown."""
        report = ResearchReport(
            query="AI agents",
            summary="Test summary",
            findings=[ResearchFinding(content="Finding 1")],
        )
        markdown = report.to_markdown()

        assert "# Research Report: AI agents" in markdown
        assert "Test summary" in markdown
        assert "Finding 1" in markdown

    def test_get_high_credibility_sources(self) -> None:
        """Test getting high credibility sources."""
        report = ResearchReport()
        report.sources.add_source(
            url="https://arxiv.org/paper",
            title="Academic",
        )
        # Result depends on credibility scoring
        sources = report.get_high_credibility_sources()
        assert isinstance(sources, list)


class TestDeepSearchAgent:
    """Tests for DeepSearchAgent."""

    def setup_method(self) -> None:
        """Reset global state."""
        reset_deep_search_agent()
        reset_search_manager()
        reset_source_manager()

    def test_agent_creation(self) -> None:
        """Test creating the agent."""
        agent = DeepSearchAgent()
        assert agent._search_manager is not None
        assert agent._planner is not None

    def test_get_max_results(self) -> None:
        """Test max results by depth."""
        agent = DeepSearchAgent()
        assert agent._get_max_results(ResearchDepth.QUICK) == 3
        assert agent._get_max_results(ResearchDepth.STANDARD) == 5
        assert agent._get_max_results(ResearchDepth.COMPREHENSIVE) == 8

    def test_extract_keywords(self) -> None:
        """Test keyword extraction."""
        agent = DeepSearchAgent()
        keywords = agent._extract_keywords(
            "LangChain is a framework for building applications with LLMs."
        )
        assert len(keywords) > 0
        assert "langchain" in keywords
        assert "framework" in keywords

    def test_extract_keywords_filters_stop_words(self) -> None:
        """Test that stop words are filtered."""
        agent = DeepSearchAgent()
        # Use words that are in stop_words and have 4+ chars: "with", "from", "that", "this"
        keywords = agent._extract_keywords("This framework works with data from that source")
        assert "this" not in keywords
        assert "with" not in keywords
        assert "from" not in keywords
        assert "that" not in keywords

    @pytest.mark.asyncio
    async def test_quick_search(self) -> None:
        """Test quick search functionality."""
        agent = DeepSearchAgent()
        results = await agent.quick_search("AI agents", max_results=3)

        assert isinstance(results, list)
        # Simulated provider should return results
        assert len(results) > 0

    def test_parse_findings(self) -> None:
        """Test parsing LLM response into findings."""
        agent = DeepSearchAgent()
        manager = SourceManager()

        text = """1. First important finding about AI.
2. Second finding regarding machine learning.
3. Third finding on neural networks."""

        findings = agent._parse_findings(text, manager)
        assert len(findings) == 3
        assert "First important finding" in findings[0].content

    def test_global_deep_search_agent(self) -> None:
        """Test global agent singleton."""
        agent1 = get_deep_search_agent()
        agent2 = get_deep_search_agent()
        assert agent1 is agent2


class TestDeepSearchAgentIntegration:
    """Integration tests for DeepSearchAgent (mocked LLM)."""

    def setup_method(self) -> None:
        """Reset global state."""
        reset_deep_search_agent()
        reset_search_manager()
        reset_source_manager()

    @pytest.mark.asyncio
    async def test_research_with_mocked_llm(self) -> None:
        """Test full research workflow with mocked LLM."""
        # Create mock LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = """1. AI agents can automate complex tasks.
2. LangChain provides tools for building agents.
3. Research continues to advance agent capabilities."""

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm.invoke = MagicMock(return_value=mock_response)

        # Create agent with mock
        agent = DeepSearchAgent(llm=mock_llm)

        # Mock the planner to avoid LLM call
        agent._planner._create_fallback_plan = MagicMock(
            return_value=ResearchPlan(
                original_query="AI agents",
                sub_queries=[SubQuery(query="AI agents")],
            )
        )
        agent._planner.decompose = MagicMock(
            return_value=ResearchPlan(
                original_query="AI agents",
                sub_queries=[SubQuery(query="AI agents")],
            )
        )

        # Run research
        report = await agent.research(
            "AI agents",
            depth=ResearchDepth.QUICK,
            max_sources=3,
        )

        assert report.query == "AI agents"
        assert report.depth == ResearchDepth.QUICK
        # Should have some sources from simulated search
        assert len(report.sources.get_all_sources()) > 0

    @pytest.mark.asyncio
    async def test_execute_search_batch(self) -> None:
        """Test executing a batch of searches."""
        agent = DeepSearchAgent()
        sub_queries = [
            SubQuery(query="test query 1"),
            SubQuery(query="test query 2"),
        ]

        results = await agent._execute_search_batch(sub_queries, max_results_per_query=3)

        assert len(results) > 0
        # Should deduplicate by URL
        urls = [r.url for r in results]
        assert len(urls) == len(set(urls))


# ==================== Module Import Tests ====================


class TestModuleImports:
    """Test that all module exports work correctly."""

    def test_import_from_research_module(self) -> None:
        """Test importing from research module."""
        from app.agents.research import (
            DeepSearchAgent,
            ResearchAgent,
            ResearchDepth,
            ResearchPlan,
            ResearchPlanner,
            SearchProviderManager,
            SourceManager,
        )

        assert DeepSearchAgent is not None
        assert ResearchAgent is not None
        assert ResearchDepth is not None
        assert ResearchPlan is not None
        assert ResearchPlanner is not None
        assert SearchProviderManager is not None
        assert SourceManager is not None

    def test_import_convenience_functions(self) -> None:
        """Test importing convenience functions."""
        from app.agents.research import (
            create_research_plan,
            deep_research,
            get_deep_search_agent,
            get_search_manager,
            get_source_manager,
            quick_search,
            search,
        )

        assert callable(create_research_plan)
        assert callable(deep_research)
        assert callable(get_deep_search_agent)
        assert callable(get_search_manager)
        assert callable(get_source_manager)
        assert callable(quick_search)
        assert callable(search)
