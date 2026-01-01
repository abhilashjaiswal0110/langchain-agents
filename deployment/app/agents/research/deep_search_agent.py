"""DeepSearch Agent for comprehensive multi-step research.

Provides advanced research capabilities:
- Query decomposition and planning
- Parallel multi-provider search
- Source credibility scoring
- Citation tracking and export
- Iterative refinement based on results
- Structured research reports

Following Enterprise Development Standards:
- Software Architect: Orchestration pattern with components
- Security Architect: Input validation, safe content handling
- Data Architect: Structured research output
- Software Engineer: Type-safe, async-first, well-documented
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from pydantic import BaseModel, Field

from app.agents.research.planner import (
    ResearchPlan,
    ResearchPlanner,
    SubQuery,
    QueryIntent,
)
from app.agents.research.source_manager import (
    CitationFormat,
    CredibilityLevel,
    Source,
    SourceManager,
    SourceType,
)
from app.agents.research.search_providers import (
    SearchProviderManager,
    SearchResponse,
    SearchResult,
    get_search_manager,
)


class ResearchDepth(str, Enum):
    """Research depth levels."""

    QUICK = "quick"  # 1-2 searches, basic summary
    STANDARD = "standard"  # 3-5 searches, comprehensive
    COMPREHENSIVE = "comprehensive"  # 5+ searches, exhaustive


@dataclass
class ResearchFinding:
    """A single research finding.

    Attributes:
        id: Finding identifier
        content: Finding content
        source_ids: Sources supporting this finding
        confidence: Confidence level (0.0-1.0)
        category: Finding category
    """

    id: str = field(default_factory=lambda: str(uuid4())[:8])
    content: str = ""
    source_ids: list[str] = field(default_factory=list)
    confidence: float = 0.5
    category: str = "general"


@dataclass
class ResearchReport:
    """Complete research report.

    Attributes:
        id: Report identifier
        query: Original research query
        plan: Research plan used
        summary: Executive summary
        findings: List of findings
        sources: Source manager with all sources
        citations: Formatted citations
        created_at: Report creation time
        depth: Research depth used
        metadata: Additional metadata
    """

    id: str = field(default_factory=lambda: str(uuid4())[:8])
    query: str = ""
    plan: ResearchPlan | None = None
    summary: str = ""
    findings: list[ResearchFinding] = field(default_factory=list)
    sources: SourceManager = field(default_factory=SourceManager)
    citations: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    depth: ResearchDepth = ResearchDepth.STANDARD
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_high_credibility_sources(self) -> list[Source]:
        """Get sources with high credibility.

        Returns:
            List of high-credibility sources
        """
        return self.sources.get_sources_by_credibility(CredibilityLevel.HIGH)

    def to_markdown(self) -> str:
        """Convert report to Markdown format.

        Returns:
            Markdown-formatted report
        """
        sections = [
            f"# Research Report: {self.query}",
            f"\n*Generated: {self.created_at}*",
            f"\n*Depth: {self.depth.value}*",
            "\n## Executive Summary\n",
            self.summary,
            "\n## Key Findings\n",
        ]

        for i, finding in enumerate(self.findings, 1):
            sections.append(f"{i}. {finding.content}")

        sections.extend([
            "\n## Sources\n",
            self.sources.export_citations(CitationFormat.MARKDOWN),
        ])

        stats = self.sources.get_statistics()
        sections.extend([
            "\n## Statistics\n",
            f"- Total Sources: {stats['total_sources']}",
            f"- Average Credibility: {stats['avg_credibility']:.2f}",
            f"- Verified Sources: {stats['verified_count']}",
        ])

        return "\n".join(sections)


class DeepSearchAgent:
    """Advanced research agent with multi-step capabilities.

    Features:
    - Query decomposition into focused sub-queries
    - Parallel search execution across providers
    - Source credibility scoring and tracking
    - Citation management and export
    - Iterative refinement based on results
    - Structured research report generation

    Example:
        >>> agent = DeepSearchAgent()
        >>> report = await agent.research(
        ...     "What are the best practices for AI agent development?",
        ...     depth=ResearchDepth.COMPREHENSIVE,
        ... )
        >>> print(report.to_markdown())
    """

    def __init__(
        self,
        llm: Any = None,
        search_manager: SearchProviderManager | None = None,
    ) -> None:
        """Initialize the DeepSearch agent.

        Args:
            llm: LangChain LLM instance (defaults to OpenAI)
            search_manager: Search provider manager
        """
        self._llm = llm
        self._search_manager = search_manager or get_search_manager()
        self._planner = ResearchPlanner(llm=llm)

    def _get_llm(self) -> Any:
        """Get or create LLM instance."""
        if self._llm is None:
            from app.agents.base.llm_factory import get_llm
            self._llm = get_llm()
        return self._llm

    @traceable(name="deep_search_research")
    async def research(
        self,
        query: str,
        depth: ResearchDepth = ResearchDepth.STANDARD,
        max_sources: int = 10,
        citation_format: CitationFormat = CitationFormat.MARKDOWN,
    ) -> ResearchReport:
        """Perform comprehensive research on a topic.

        Args:
            query: Research query
            depth: Research depth level
            max_sources: Maximum sources to collect
            citation_format: Format for citations

        Returns:
            Complete ResearchReport
        """
        # Initialize report
        source_manager = SourceManager()
        report = ResearchReport(
            query=query,
            depth=depth,
            sources=source_manager,
        )

        # Step 1: Create research plan
        depth_str = depth.value
        plan = self._planner.decompose(query, depth=depth_str)
        report.plan = plan

        # Step 2: Execute searches for each sub-query
        all_results: list[SearchResult] = []
        execution_order = plan.get_execution_order()

        for batch in execution_order:
            # Execute batch in parallel
            batch_results = await self._execute_search_batch(
                batch,
                max_results_per_query=self._get_max_results(depth),
            )
            all_results.extend(batch_results)

        # Step 3: Process and score sources
        for result in all_results[:max_sources]:
            source = source_manager.add_source(
                url=result.url,
                title=result.title,
                content_summary=result.snippet,
                keywords=self._extract_keywords(result.snippet),
            )

        # Step 4: Synthesize findings
        findings = await self._synthesize_findings(
            query=query,
            results=all_results,
            source_manager=source_manager,
        )
        report.findings = findings

        # Step 5: Generate summary
        summary = await self._generate_summary(
            query=query,
            findings=findings,
            sources=source_manager.get_all_sources(),
        )
        report.summary = summary

        # Step 6: Generate citations
        report.citations = source_manager.export_citations(citation_format)

        return report

    async def _execute_search_batch(
        self,
        sub_queries: list[SubQuery],
        max_results_per_query: int = 5,
    ) -> list[SearchResult]:
        """Execute a batch of sub-queries in parallel.

        Args:
            sub_queries: List of sub-queries to execute
            max_results_per_query: Max results per query

        Returns:
            Combined list of search results
        """
        queries = [sq.query for sq in sub_queries]
        responses = await self._search_manager.search_parallel(
            queries,
            max_results_per_query=max_results_per_query,
        )

        all_results = []
        for response in responses:
            if response.success:
                all_results.extend(response.results)

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        return unique_results

    async def _synthesize_findings(
        self,
        query: str,
        results: list[SearchResult],
        source_manager: SourceManager,
    ) -> list[ResearchFinding]:
        """Synthesize findings from search results.

        Args:
            query: Original query
            results: Search results
            source_manager: Source manager

        Returns:
            List of research findings
        """
        if not results:
            return [ResearchFinding(
                content="No search results were found for this query.",
                confidence=0.0,
                category="error",
            )]

        llm = self._get_llm()

        # Prepare content for synthesis
        content_parts = []
        for result in results[:10]:
            content_parts.append(
                f"Source: {result.title}\n"
                f"URL: {result.url}\n"
                f"Content: {result.snippet}\n"
            )

        combined_content = "\n---\n".join(content_parts)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research analyst. Extract key findings from the search results.

For each finding:
1. State the finding clearly and concisely
2. Note which source(s) support it
3. Assess confidence (high/medium/low)

Output as a numbered list of findings. Be specific and factual."""),
            ("human", """Research Query: {query}

Search Results:
{content}

Extract 3-7 key findings from these results."""),
        ])

        chain = prompt | llm

        try:
            response = await chain.ainvoke({
                "query": query,
                "content": combined_content,
            })

            # Parse response into findings
            findings = self._parse_findings(
                response.content if hasattr(response, "content") else str(response),
                source_manager,
            )

            return findings

        except Exception as e:
            return [ResearchFinding(
                content=f"Error synthesizing findings: {e}",
                confidence=0.0,
                category="error",
            )]

    def _parse_findings(
        self,
        text: str,
        source_manager: SourceManager,
    ) -> list[ResearchFinding]:
        """Parse LLM response into structured findings.

        Args:
            text: LLM response text
            source_manager: Source manager for linking

        Returns:
            List of ResearchFinding objects
        """
        findings = []
        lines = text.strip().split("\n")

        current_finding = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this starts a new numbered finding
            if line[0].isdigit() and "." in line[:3]:
                if current_finding:
                    findings.append(ResearchFinding(
                        content=current_finding.strip(),
                        confidence=0.7,
                        category="general",
                    ))
                current_finding = line.split(".", 1)[-1].strip()
            else:
                current_finding += " " + line

        # Don't forget the last finding
        if current_finding:
            findings.append(ResearchFinding(
                content=current_finding.strip(),
                confidence=0.7,
                category="general",
            ))

        return findings

    async def _generate_summary(
        self,
        query: str,
        findings: list[ResearchFinding],
        sources: list[Source],
    ) -> str:
        """Generate executive summary from findings.

        Args:
            query: Original query
            findings: Research findings
            sources: Sources used

        Returns:
            Executive summary text
        """
        if not findings:
            return "No findings available to summarize."

        llm = self._get_llm()

        findings_text = "\n".join(f"- {f.content}" for f in findings)
        sources_text = ", ".join(s.title for s in sources[:5])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research analyst writing an executive summary.
Write a clear, concise summary (2-4 paragraphs) that:
1. Directly addresses the research question
2. Synthesizes the key findings
3. Notes any limitations or areas for further research
Be factual and objective."""),
            ("human", """Research Query: {query}

Key Findings:
{findings}

Sources consulted include: {sources}

Write an executive summary."""),
        ])

        chain = prompt | llm

        try:
            response = await chain.ainvoke({
                "query": query,
                "findings": findings_text,
                "sources": sources_text,
            })

            return response.content if hasattr(response, "content") else str(response)

        except Exception as e:
            return f"Error generating summary: {e}"

    def _get_max_results(self, depth: ResearchDepth) -> int:
        """Get max results per query based on depth.

        Args:
            depth: Research depth

        Returns:
            Maximum results per query
        """
        return {
            ResearchDepth.QUICK: 3,
            ResearchDepth.STANDARD: 5,
            ResearchDepth.COMPREHENSIVE: 8,
        }.get(depth, 5)

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text.

        Args:
            text: Text to analyze

        Returns:
            List of keywords
        """
        # Simple keyword extraction
        import re

        # Remove common words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "of", "to", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "just", "and", "but", "if", "or",
            "because", "until", "while", "this", "that", "these", "those",
        }

        # Extract words
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())

        # Filter and deduplicate
        keywords = []
        seen = set()
        for word in words:
            if word not in stop_words and word not in seen:
                seen.add(word)
                keywords.append(word)
                if len(keywords) >= 10:
                    break

        return keywords

    @traceable(name="quick_search")
    async def quick_search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[SearchResult]:
        """Perform a quick search without full research workflow.

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            List of search results
        """
        response = await self._search_manager.search(query, max_results)
        return response.results

    @traceable(name="verify_claim")
    async def verify_claim(
        self,
        claim: str,
        num_sources: int = 3,
    ) -> dict[str, Any]:
        """Verify a claim by searching for supporting/contradicting sources.

        Args:
            claim: Claim to verify
            num_sources: Number of sources to check

        Returns:
            Verification result with supporting/contradicting evidence
        """
        # Search for the claim
        response = await self._search_manager.search(claim, num_sources * 2)

        if not response.success:
            return {
                "claim": claim,
                "verified": False,
                "confidence": 0.0,
                "error": response.error,
            }

        # Analyze results with LLM
        llm = self._get_llm()

        results_text = "\n".join(
            f"- {r.title}: {r.snippet}" for r in response.results[:num_sources]
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fact-checker. Analyze the search results to verify the claim.

Determine:
1. Whether the claim is supported, contradicted, or unverified
2. Confidence level (0.0 to 1.0)
3. Key supporting or contradicting evidence

Respond with JSON:
{{"status": "supported|contradicted|unverified", "confidence": 0.0-1.0, "evidence": "brief summary"}}"""),
            ("human", """Claim: {claim}

Search Results:
{results}

Analyze and verify this claim."""),
        ])

        chain = prompt | llm

        try:
            response = await chain.ainvoke({
                "claim": claim,
                "results": results_text,
            })

            # Parse response
            import json
            content = response.content if hasattr(response, "content") else str(response)

            # Extract JSON from response
            import re
            json_match = re.search(r"\{[^}]+\}", content)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "claim": claim,
                    "verified": result.get("status") == "supported",
                    "status": result.get("status", "unverified"),
                    "confidence": result.get("confidence", 0.5),
                    "evidence": result.get("evidence", ""),
                }

            return {
                "claim": claim,
                "verified": False,
                "confidence": 0.0,
                "error": "Could not parse verification result",
            }

        except Exception as e:
            return {
                "claim": claim,
                "verified": False,
                "confidence": 0.0,
                "error": str(e),
            }


# Global instance
_deep_search_agent: DeepSearchAgent | None = None


def get_deep_search_agent() -> DeepSearchAgent:
    """Get or create the global DeepSearch agent.

    Returns:
        DeepSearchAgent instance
    """
    global _deep_search_agent
    if _deep_search_agent is None:
        _deep_search_agent = DeepSearchAgent()
    return _deep_search_agent


def reset_deep_search_agent() -> None:
    """Reset the global DeepSearch agent."""
    global _deep_search_agent
    _deep_search_agent = None


# Convenience functions
async def deep_research(
    query: str,
    depth: ResearchDepth = ResearchDepth.STANDARD,
) -> ResearchReport:
    """Perform deep research on a topic.

    Args:
        query: Research query
        depth: Research depth

    Returns:
        ResearchReport with findings
    """
    agent = get_deep_search_agent()
    return await agent.research(query, depth=depth)


async def quick_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Perform a quick search.

    Args:
        query: Search query
        max_results: Maximum results

    Returns:
        List of search results
    """
    agent = get_deep_search_agent()
    return await agent.quick_search(query, max_results)
