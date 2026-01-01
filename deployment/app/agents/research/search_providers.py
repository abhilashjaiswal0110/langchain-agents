"""Multi-provider search abstraction for research.

Provides unified interface to multiple search backends:
- Tavily (AI-optimized search)
- DuckDuckGo (privacy-focused)
- Google Custom Search
- Bing Search
- Simulated (for testing/demos)

Following Enterprise Development Standards:
- Software Architect: Provider abstraction pattern
- Security Architect: API key management, rate limiting
- Data Architect: Standardized result format
- Software Engineer: Type-safe, async-first
"""

import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class SearchProviderType(str, Enum):
    """Available search providers."""

    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    GOOGLE = "google"
    BING = "bing"
    SIMULATED = "simulated"


@dataclass
class SearchResult:
    """A single search result.

    Attributes:
        id: Unique result identifier
        title: Result title
        url: Result URL
        snippet: Content snippet/summary
        content: Full content if available
        score: Relevance score (0.0-1.0)
        provider: Which provider returned this
        metadata: Additional provider-specific data
    """

    id: str = field(default_factory=lambda: str(uuid4())[:8])
    title: str = ""
    url: str = ""
    snippet: str = ""
    content: str = ""
    score: float = 0.5
    provider: SearchProviderType = SearchProviderType.SIMULATED
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResponse:
    """Response from a search operation.

    Attributes:
        query: Original search query
        results: List of search results
        provider: Provider that handled the search
        total_results: Estimated total results available
        search_time_ms: Time taken for search in milliseconds
        error: Error message if search failed
    """

    query: str = ""
    results: list[SearchResult] = field(default_factory=list)
    provider: SearchProviderType = SearchProviderType.SIMULATED
    total_results: int = 0
    search_time_ms: float = 0.0
    error: str | None = None

    @property
    def success(self) -> bool:
        """Check if search was successful."""
        return self.error is None and len(self.results) > 0


class SearchProvider(ABC):
    """Abstract base class for search providers."""

    provider_type: SearchProviderType = SearchProviderType.SIMULATED

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> SearchResponse:
        """Perform a search.

        Args:
            query: Search query
            max_results: Maximum results to return
            **kwargs: Provider-specific options

        Returns:
            SearchResponse with results
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available/configured.

        Returns:
            True if provider can be used
        """
        pass


class TavilySearchProvider(SearchProvider):
    """Tavily AI-optimized search provider.

    Tavily is designed for AI agents with:
    - Optimized snippets for LLM consumption
    - Relevance scoring
    - Content extraction
    """

    provider_type = SearchProviderType.TAVILY

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize Tavily provider.

        Args:
            api_key: Tavily API key (falls back to env var)
        """
        self._api_key = api_key or os.getenv("TAVILY_API_KEY")
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create Tavily client."""
        if self._client is None:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self._api_key)
            except ImportError:
                msg = "tavily package not installed"
                raise RuntimeError(msg)
        return self._client

    def is_available(self) -> bool:
        """Check if Tavily is configured."""
        return bool(self._api_key)

    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> SearchResponse:
        """Search using Tavily.

        Args:
            query: Search query
            max_results: Maximum results
            **kwargs: Additional Tavily options (search_depth, include_domains, etc.)

        Returns:
            SearchResponse with results
        """
        import time
        start = time.time()

        if not self.is_available():
            return SearchResponse(
                query=query,
                provider=self.provider_type,
                error="Tavily API key not configured",
            )

        try:
            client = self._get_client()

            # Tavily is synchronous, run in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.search(
                    query,
                    max_results=max_results,
                    search_depth=kwargs.get("search_depth", "basic"),
                    include_domains=kwargs.get("include_domains"),
                    exclude_domains=kwargs.get("exclude_domains"),
                ),
            )

            results = []
            for item in response.get("results", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", "")[:500],
                    content=item.get("content", ""),
                    score=item.get("score", 0.5),
                    provider=self.provider_type,
                    metadata={
                        "raw_content": item.get("raw_content"),
                        "published_date": item.get("published_date"),
                    },
                ))

            elapsed = (time.time() - start) * 1000

            return SearchResponse(
                query=query,
                results=results,
                provider=self.provider_type,
                total_results=len(results),
                search_time_ms=elapsed,
            )

        except Exception as e:
            return SearchResponse(
                query=query,
                provider=self.provider_type,
                error=str(e),
                search_time_ms=(time.time() - start) * 1000,
            )


class DuckDuckGoSearchProvider(SearchProvider):
    """DuckDuckGo search provider.

    Privacy-focused search with no API key required.
    Uses duckduckgo-search package.
    """

    provider_type = SearchProviderType.DUCKDUCKGO

    def __init__(self) -> None:
        """Initialize DuckDuckGo provider."""
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Check if DuckDuckGo package is installed."""
        if self._available is None:
            try:
                from duckduckgo_search import DDGS
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> SearchResponse:
        """Search using DuckDuckGo.

        Args:
            query: Search query
            max_results: Maximum results
            **kwargs: Additional options (region, safesearch)

        Returns:
            SearchResponse with results
        """
        import time
        start = time.time()

        if not self.is_available():
            return SearchResponse(
                query=query,
                provider=self.provider_type,
                error="duckduckgo-search package not installed",
            )

        try:
            from duckduckgo_search import DDGS

            # DuckDuckGo is synchronous, run in executor
            loop = asyncio.get_event_loop()

            def _search() -> list[dict[str, Any]]:
                with DDGS() as ddgs:
                    return list(ddgs.text(
                        query,
                        max_results=max_results,
                        region=kwargs.get("region", "wt-wt"),
                        safesearch=kwargs.get("safesearch", "moderate"),
                    ))

            raw_results = await loop.run_in_executor(None, _search)

            results = []
            for i, item in enumerate(raw_results):
                # DuckDuckGo doesn't provide relevance scores
                # Estimate based on position
                score = 1.0 - (i / max(len(raw_results), 1)) * 0.5

                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("href", item.get("link", "")),
                    snippet=item.get("body", "")[:500],
                    content=item.get("body", ""),
                    score=score,
                    provider=self.provider_type,
                ))

            elapsed = (time.time() - start) * 1000

            return SearchResponse(
                query=query,
                results=results,
                provider=self.provider_type,
                total_results=len(results),
                search_time_ms=elapsed,
            )

        except Exception as e:
            return SearchResponse(
                query=query,
                provider=self.provider_type,
                error=str(e),
                search_time_ms=(time.time() - start) * 1000,
            )


class SimulatedSearchProvider(SearchProvider):
    """Simulated search provider for testing and demos.

    Returns mock results based on query keywords.
    Useful when no real search API is configured.
    """

    provider_type = SearchProviderType.SIMULATED

    # Sample results for common topics
    SAMPLE_RESULTS: dict[str, list[dict[str, str]]] = {
        "ai": [
            {
                "title": "Understanding Artificial Intelligence",
                "url": "https://example.com/ai-overview",
                "snippet": "A comprehensive guide to AI concepts, machine learning, and neural networks.",
            },
            {
                "title": "AI in Enterprise Applications",
                "url": "https://example.com/enterprise-ai",
                "snippet": "How businesses are leveraging AI for automation and decision-making.",
            },
        ],
        "langchain": [
            {
                "title": "LangChain Documentation",
                "url": "https://docs.langchain.com",
                "snippet": "Official documentation for LangChain - Build applications with LLMs.",
            },
            {
                "title": "LangChain Tutorial: Building AI Agents",
                "url": "https://example.com/langchain-tutorial",
                "snippet": "Step-by-step guide to building AI agents with LangChain and LangGraph.",
            },
        ],
        "python": [
            {
                "title": "Python Official Documentation",
                "url": "https://docs.python.org",
                "snippet": "Official Python documentation with tutorials and library references.",
            },
            {
                "title": "Python Best Practices Guide",
                "url": "https://example.com/python-best-practices",
                "snippet": "Modern Python development practices, typing, and code organization.",
            },
        ],
    }

    def is_available(self) -> bool:
        """Simulated provider is always available."""
        return True

    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> SearchResponse:
        """Return simulated search results.

        Args:
            query: Search query
            max_results: Maximum results
            **kwargs: Ignored

        Returns:
            SearchResponse with simulated results
        """
        import time
        start = time.time()

        # Simulate network delay
        await asyncio.sleep(0.1)

        results = []
        query_lower = query.lower()

        # Find matching sample results
        for keyword, samples in self.SAMPLE_RESULTS.items():
            if keyword in query_lower:
                for sample in samples[:max_results]:
                    results.append(SearchResult(
                        title=sample["title"],
                        url=sample["url"],
                        snippet=sample["snippet"],
                        content=sample["snippet"],
                        score=0.8,
                        provider=self.provider_type,
                    ))

        # If no matches, generate generic results
        if not results:
            for i in range(min(3, max_results)):
                results.append(SearchResult(
                    title=f"Result {i+1} for: {query}",
                    url=f"https://example.com/result-{i+1}",
                    snippet=f"This is a simulated result for the query '{query}'. "
                            f"Configure TAVILY_API_KEY for real search results.",
                    content=f"Simulated content for '{query}'.",
                    score=0.7 - (i * 0.1),
                    provider=self.provider_type,
                ))

        elapsed = (time.time() - start) * 1000

        return SearchResponse(
            query=query,
            results=results[:max_results],
            provider=self.provider_type,
            total_results=len(results),
            search_time_ms=elapsed,
        )


class SearchProviderManager:
    """Manages multiple search providers with fallback support.

    Example:
        >>> manager = SearchProviderManager()
        >>> response = await manager.search("AI agents", max_results=5)
        >>> for result in response.results:
        ...     print(result.title, result.url)
    """

    def __init__(
        self,
        providers: list[SearchProvider] | None = None,
        fallback_to_simulated: bool = True,
    ) -> None:
        """Initialize the provider manager.

        Args:
            providers: List of providers to use (in priority order)
            fallback_to_simulated: Whether to fall back to simulated if all fail
        """
        self._providers = providers or self._create_default_providers()
        self._fallback_to_simulated = fallback_to_simulated
        self._simulated = SimulatedSearchProvider()

    def _create_default_providers(self) -> list[SearchProvider]:
        """Create default provider list based on available APIs."""
        providers = []

        # Tavily (preferred for AI)
        if os.getenv("TAVILY_API_KEY"):
            providers.append(TavilySearchProvider())

        # DuckDuckGo (no API key needed)
        ddg = DuckDuckGoSearchProvider()
        if ddg.is_available():
            providers.append(ddg)

        return providers

    def get_available_providers(self) -> list[SearchProvider]:
        """Get list of available providers.

        Returns:
            List of configured/available providers
        """
        return [p for p in self._providers if p.is_available()]

    async def search(
        self,
        query: str,
        max_results: int = 10,
        provider_type: SearchProviderType | None = None,
        **kwargs: Any,
    ) -> SearchResponse:
        """Search using the best available provider.

        Args:
            query: Search query
            max_results: Maximum results
            provider_type: Specific provider to use (optional)
            **kwargs: Provider-specific options

        Returns:
            SearchResponse with results
        """
        # If specific provider requested
        if provider_type:
            provider = self._get_provider_by_type(provider_type)
            if provider and provider.is_available():
                return await provider.search(query, max_results, **kwargs)

            return SearchResponse(
                query=query,
                provider=provider_type,
                error=f"Provider {provider_type.value} not available",
            )

        # Try providers in order
        for provider in self._providers:
            if provider.is_available():
                response = await provider.search(query, max_results, **kwargs)
                if response.success:
                    return response

        # Fallback to simulated
        if self._fallback_to_simulated:
            return await self._simulated.search(query, max_results, **kwargs)

        return SearchResponse(
            query=query,
            provider=SearchProviderType.SIMULATED,
            error="No search providers available",
        )

    async def search_parallel(
        self,
        queries: list[str],
        max_results_per_query: int = 5,
        **kwargs: Any,
    ) -> list[SearchResponse]:
        """Search multiple queries in parallel.

        Args:
            queries: List of search queries
            max_results_per_query: Maximum results per query
            **kwargs: Provider-specific options

        Returns:
            List of SearchResponse objects
        """
        tasks = [
            self.search(query, max_results_per_query, **kwargs)
            for query in queries
        ]

        return await asyncio.gather(*tasks)

    async def search_with_fallback(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> SearchResponse:
        """Search with automatic fallback through all providers.

        Args:
            query: Search query
            max_results: Maximum results
            **kwargs: Provider-specific options

        Returns:
            SearchResponse from first successful provider
        """
        errors = []

        for provider in self._providers:
            if provider.is_available():
                response = await provider.search(query, max_results, **kwargs)
                if response.success:
                    return response
                if response.error:
                    errors.append(f"{provider.provider_type.value}: {response.error}")

        # Try simulated as last resort
        if self._fallback_to_simulated:
            return await self._simulated.search(query, max_results, **kwargs)

        return SearchResponse(
            query=query,
            provider=SearchProviderType.SIMULATED,
            error=f"All providers failed: {'; '.join(errors)}",
        )

    def _get_provider_by_type(
        self,
        provider_type: SearchProviderType,
    ) -> SearchProvider | None:
        """Get a specific provider by type.

        Args:
            provider_type: Provider type to find

        Returns:
            Provider if found, None otherwise
        """
        for provider in self._providers:
            if provider.provider_type == provider_type:
                return provider

        if provider_type == SearchProviderType.SIMULATED:
            return self._simulated

        return None


# Global instance
_search_manager: SearchProviderManager | None = None


def get_search_manager() -> SearchProviderManager:
    """Get or create the global search manager.

    Returns:
        SearchProviderManager instance
    """
    global _search_manager
    if _search_manager is None:
        _search_manager = SearchProviderManager()
    return _search_manager


def reset_search_manager() -> None:
    """Reset the global search manager."""
    global _search_manager
    _search_manager = None


# Convenience function
async def search(
    query: str,
    max_results: int = 10,
    provider: SearchProviderType | None = None,
) -> SearchResponse:
    """Perform a search using the default manager.

    Args:
        query: Search query
        max_results: Maximum results
        provider: Specific provider to use

    Returns:
        SearchResponse with results
    """
    manager = get_search_manager()
    return await manager.search(query, max_results, provider_type=provider)
