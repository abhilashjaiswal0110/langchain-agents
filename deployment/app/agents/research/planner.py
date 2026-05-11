"""Research Query Planner for decomposing complex queries.

Provides intelligent query decomposition:
- Break complex questions into focused sub-queries
- Identify query intent and scope
- Prioritize research steps
- Support parallel and sequential execution strategies

Following Enterprise Development Standards:
- Software Architect: Modular decomposition pattern
- Security Architect: No sensitive data in queries
- Data Architect: Structured query planning
- Software Engineer: Type-safe, well-documented
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable


class QueryIntent(str, Enum):
    """Classification of query intent."""

    FACTUAL = "factual"  # Looking for specific facts
    COMPARISON = "comparison"  # Comparing multiple things
    ANALYSIS = "analysis"  # Deep analysis required
    HOW_TO = "how_to"  # Procedural/instructional
    TREND = "trend"  # Trends and developments
    OPINION = "opinion"  # Opinions/perspectives
    DEFINITION = "definition"  # Definitions/explanations


class ExecutionStrategy(str, Enum):
    """Strategy for executing sub-queries."""

    PARALLEL = "parallel"  # Execute all at once
    SEQUENTIAL = "sequential"  # Execute in order
    HIERARCHICAL = "hierarchical"  # Results feed into next


@dataclass
class SubQuery:
    """A decomposed sub-query for research.

    Attributes:
        id: Unique identifier
        query: The sub-query text
        intent: Query intent classification
        priority: Execution priority (1=highest)
        depends_on: List of sub-query IDs this depends on
        keywords: Key terms for search optimization
        estimated_sources: Expected number of sources needed
    """

    id: str = field(default_factory=lambda: str(uuid4())[:8])
    query: str = ""
    intent: QueryIntent = QueryIntent.FACTUAL
    priority: int = 1
    depends_on: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    estimated_sources: int = 3


@dataclass
class ResearchPlan:
    """A complete research plan with decomposed queries.

    Attributes:
        id: Unique plan identifier
        original_query: The original research query
        intent: Primary intent of the research
        sub_queries: List of decomposed sub-queries
        strategy: Execution strategy
        max_depth: Maximum research depth
        estimated_duration_minutes: Estimated time to complete
    """

    id: str = field(default_factory=lambda: str(uuid4())[:8])
    original_query: str = ""
    intent: QueryIntent = QueryIntent.ANALYSIS
    sub_queries: list[SubQuery] = field(default_factory=list)
    strategy: ExecutionStrategy = ExecutionStrategy.PARALLEL
    max_depth: int = 2
    estimated_duration_minutes: int = 5

    def get_execution_order(self) -> list[list[SubQuery]]:
        """Get sub-queries organized by execution order.

        Returns:
            List of batches, each batch can run in parallel.
        """
        if self.strategy == ExecutionStrategy.PARALLEL:
            return [self.sub_queries]

        if self.strategy == ExecutionStrategy.SEQUENTIAL:
            return [[sq] for sq in sorted(self.sub_queries, key=lambda x: x.priority)]

        # Hierarchical: group by dependency
        batches: list[list[SubQuery]] = []
        executed: set[str] = set()
        remaining = list(self.sub_queries)

        while remaining:
            # Find queries with all dependencies met
            ready = [sq for sq in remaining if all(dep in executed for dep in sq.depends_on)]

            if not ready:
                # Avoid infinite loop - add remaining with unmet deps
                ready = remaining
                remaining = []
            else:
                for sq in ready:
                    remaining.remove(sq)

            batches.append(ready)
            executed.update(sq.id for sq in ready)

        return batches


class ResearchPlanner:
    """Intelligent query decomposition and research planning.

    Uses LLM to break down complex queries into manageable
    sub-queries with proper execution strategy.

    Example:
        >>> planner = ResearchPlanner()
        >>> plan = planner.create_plan("Compare AI frameworks for production")
        >>> for batch in plan.get_execution_order():
        ...     print([sq.query for sq in batch])
    """

    def __init__(self, llm: Any = None) -> None:
        """Initialize the research planner.

        Args:
            llm: LangChain LLM instance (defaults to OpenAI)
        """
        self._llm = llm
        self._decomposition_prompt = self._create_decomposition_prompt()

    def _get_llm(self) -> Any:
        """Get or create LLM instance."""
        if self._llm is None:
            from app.agents.base.llm_factory import get_llm

            self._llm = get_llm()
        return self._llm

    def _create_decomposition_prompt(self) -> ChatPromptTemplate:
        """Create the prompt for query decomposition."""
        system = """You are a research planning expert. Given a complex query,
decompose it into focused sub-queries for comprehensive research.

Guidelines:
1. Break down the query into 2-5 focused sub-queries
2. Each sub-query should be specific and searchable
3. Identify the primary intent of each sub-query
4. Assign priorities (1=highest, 5=lowest)
5. Note dependencies between sub-queries

Output JSON format:
{{
    "intent": "factual|comparison|analysis|how_to|trend|opinion|definition",
    "strategy": "parallel|sequential|hierarchical",
    "max_depth": 1-3,
    "sub_queries": [
        {{
            "query": "specific search query",
            "intent": "factual|comparison|analysis|how_to|trend|opinion|definition",
            "priority": 1-5,
            "depends_on": [],
            "keywords": ["key", "terms"],
            "estimated_sources": 2-5
        }}
    ]
}}"""

        human = """Decompose this research query into sub-queries:

Query: {query}

Research depth: {depth}
Max sub-queries: {max_sub_queries}

Respond with valid JSON only."""

        return ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", human),
            ]
        )

    @traceable(name="decompose_query")
    def decompose(
        self,
        query: str,
        depth: str = "standard",
        max_sub_queries: int = 5,
    ) -> ResearchPlan:
        """Decompose a query into a research plan.

        Args:
            query: The research query to decompose
            depth: Research depth (quick/standard/comprehensive)
            max_sub_queries: Maximum number of sub-queries

        Returns:
            ResearchPlan with decomposed sub-queries
        """
        llm = self._get_llm()

        # Adjust based on depth
        depth_config = {
            "quick": {"max_sub_queries": 2, "max_depth": 1},
            "standard": {"max_sub_queries": 4, "max_depth": 2},
            "comprehensive": {"max_sub_queries": 6, "max_depth": 3},
        }

        config = depth_config.get(depth, depth_config["standard"])
        actual_max = min(max_sub_queries, config["max_sub_queries"])

        # Create chain
        chain = self._decomposition_prompt | llm | JsonOutputParser()

        try:
            result = chain.invoke(
                {
                    "query": query,
                    "depth": depth,
                    "max_sub_queries": actual_max,
                }
            )

            # Parse result into ResearchPlan
            return self._parse_plan(query, result, config["max_depth"])

        except Exception as e:
            # Fallback: create simple single-query plan
            return self._create_fallback_plan(query, str(e))

    def _parse_plan(
        self,
        original_query: str,
        result: dict[str, Any],
        max_depth: int,
    ) -> ResearchPlan:
        """Parse LLM output into ResearchPlan.

        Args:
            original_query: Original research query
            result: Parsed JSON from LLM
            max_depth: Maximum research depth

        Returns:
            ResearchPlan instance
        """
        # Parse intent
        intent_str = result.get("intent", "analysis")
        try:
            intent = QueryIntent(intent_str)
        except ValueError:
            intent = QueryIntent.ANALYSIS

        # Parse strategy
        strategy_str = result.get("strategy", "parallel")
        try:
            strategy = ExecutionStrategy(strategy_str)
        except ValueError:
            strategy = ExecutionStrategy.PARALLEL

        # Parse sub-queries
        sub_queries = []
        for sq_data in result.get("sub_queries", []):
            sq_intent_str = sq_data.get("intent", "factual")
            try:
                sq_intent = QueryIntent(sq_intent_str)
            except ValueError:
                sq_intent = QueryIntent.FACTUAL

            sub_queries.append(
                SubQuery(
                    query=sq_data.get("query", ""),
                    intent=sq_intent,
                    priority=sq_data.get("priority", 1),
                    depends_on=sq_data.get("depends_on", []),
                    keywords=sq_data.get("keywords", []),
                    estimated_sources=sq_data.get("estimated_sources", 3),
                )
            )

        # Estimate duration based on sub-queries
        estimated_minutes = len(sub_queries) * 2

        return ResearchPlan(
            original_query=original_query,
            intent=intent,
            sub_queries=sub_queries,
            strategy=strategy,
            max_depth=min(result.get("max_depth", 2), max_depth),
            estimated_duration_minutes=estimated_minutes,
        )

    def _create_fallback_plan(self, query: str, error: str) -> ResearchPlan:
        """Create a simple fallback plan when decomposition fails.

        Args:
            query: Original query
            error: Error message for logging

        Returns:
            Simple single-query ResearchPlan
        """
        return ResearchPlan(
            original_query=query,
            intent=QueryIntent.ANALYSIS,
            sub_queries=[
                SubQuery(
                    query=query,
                    intent=QueryIntent.ANALYSIS,
                    priority=1,
                    keywords=query.split()[:5],
                    estimated_sources=3,
                ),
            ],
            strategy=ExecutionStrategy.PARALLEL,
            max_depth=1,
            estimated_duration_minutes=3,
        )

    @traceable(name="refine_sub_query")
    def refine_sub_query(
        self,
        sub_query: SubQuery,
        context: str | None = None,
    ) -> SubQuery:
        """Refine a sub-query based on initial results.

        Args:
            sub_query: Sub-query to refine
            context: Context from previous results

        Returns:
            Refined SubQuery
        """
        if not context:
            return sub_query

        llm = self._get_llm()

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Refine this search query based on the context provided."),
                (
                    "human",
                    """Original query: {query}

Context from initial results:
{context}

Provide a refined, more specific search query. Respond with just the query text.""",
                ),
            ]
        )

        chain = prompt | llm

        try:
            response = chain.invoke(
                {
                    "query": sub_query.query,
                    "context": context[:1000],
                }
            )

            # Create refined sub-query
            return SubQuery(
                query=response.content if hasattr(response, "content") else str(response),
                intent=sub_query.intent,
                priority=sub_query.priority,
                depends_on=sub_query.depends_on,
                keywords=sub_query.keywords,
                estimated_sources=sub_query.estimated_sources,
            )

        except Exception:
            return sub_query

    def classify_intent(self, query: str) -> QueryIntent:
        """Quickly classify the intent of a query.

        Args:
            query: Query to classify

        Returns:
            QueryIntent classification
        """
        query_lower = query.lower()

        # Simple keyword-based classification
        if any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
            return QueryIntent.COMPARISON
        if any(word in query_lower for word in ["how to", "steps", "guide", "tutorial"]):
            return QueryIntent.HOW_TO
        if any(word in query_lower for word in ["what is", "define", "meaning"]):
            return QueryIntent.DEFINITION
        if any(word in query_lower for word in ["trend", "future", "emerging", "latest"]):
            return QueryIntent.TREND
        if any(word in query_lower for word in ["analysis", "analyze", "evaluate", "assess"]):
            return QueryIntent.ANALYSIS
        if any(word in query_lower for word in ["opinion", "think", "best", "recommend"]):
            return QueryIntent.OPINION

        return QueryIntent.FACTUAL


# Convenience function
def create_research_plan(
    query: str,
    depth: str = "standard",
    llm: Any = None,
) -> ResearchPlan:
    """Create a research plan from a query.

    Args:
        query: Research query
        depth: Research depth level
        llm: Optional LLM instance

    Returns:
        ResearchPlan with decomposed sub-queries
    """
    planner = ResearchPlanner(llm=llm)
    return planner.decompose(query, depth=depth)
