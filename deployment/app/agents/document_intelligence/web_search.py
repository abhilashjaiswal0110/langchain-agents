"""Domain-restricted web search for the Document Intelligence Agent.

This module wraps the existing SearchProviderManager with domain
restrictions configured via environment variables.

Following Enterprise Development Standards:
- Security Architect: Domain whitelist enforcement
- Software Engineer: Async-ready, type-safe
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class DomainRestrictedSearch:
    """Web search restricted to configured domains.

    Uses the platform's existing SearchProviderManager with domain
    filtering via the ALLOWED_SEARCH_DOMAINS environment variable.
    """

    def __init__(self) -> None:
        """Initialize the domain-restricted search."""
        self._manager = None
        self._allowed_domains = self._load_allowed_domains()

    def _load_allowed_domains(self) -> list[str]:
        """Load allowed domains from environment.

        Returns:
            List of allowed domain strings
        """
        domains_str = os.getenv("ALLOWED_SEARCH_DOMAINS", "")
        if not domains_str:
            logger.warning(
                "ALLOWED_SEARCH_DOMAINS not set. Web search will be unavailable. "
                "Set ALLOWED_SEARCH_DOMAINS=domain1.com,domain2.com in your .env file."
            )
            return []
        return [d.strip() for d in domains_str.split(",") if d.strip()]

    def _get_manager(self) -> Any:
        """Lazily load the SearchProviderManager.

        Returns:
            SearchProviderManager instance
        """
        if self._manager is None:
            try:
                from app.agents.research.search_providers import SearchProviderManager

                self._manager = SearchProviderManager()
            except ImportError:
                logger.error("SearchProviderManager not available")
                raise ImportError(
                    "SearchProviderManager not found. Ensure app.agents.research.search_providers is available."
                )
        return self._manager

    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> dict[str, Any]:
        """Search with domain restrictions.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            Search results dict with query, results, and metadata
        """
        if not self._allowed_domains:
            return {
                "query": query,
                "results": [],
                "error": "No allowed domains configured. Set ALLOWED_SEARCH_DOMAINS in .env",
                "allowed_domains": [],
            }

        try:
            manager = self._get_manager()

            # Use include_domains parameter for Tavily
            response = await manager.search(
                query=query,
                max_results=max_results,
                include_domains=self._allowed_domains,
            )

            # Format results
            results = []
            for result in response.results:
                results.append(
                    {
                        "title": result.title,
                        "url": result.url,
                        "snippet": result.snippet,
                        "score": result.score,
                    }
                )

            return {
                "query": query,
                "results": results,
                "total_results": len(results),
                "allowed_domains": self._allowed_domains,
                "provider": str(response.provider) if hasattr(response, "provider") else "unknown",
            }

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {
                "query": query,
                "results": [],
                "error": str(e),
                "allowed_domains": self._allowed_domains,
            }

    def search_sync(
        self,
        query: str,
        max_results: int = 5,
    ) -> dict[str, Any]:
        """Synchronous search wrapper.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            Search results dict
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.search(query, max_results))

    def get_allowed_domains(self) -> list[str]:
        """Get the list of allowed domains.

        Returns:
            List of allowed domain strings
        """
        return self._allowed_domains.copy()

    def is_domain_allowed(self, url: str) -> bool:
        """Check if a URL's domain is in the allowed list.

        Args:
            url: URL to check

        Returns:
            True if domain is allowed
        """
        if not self._allowed_domains:
            return False

        from urllib.parse import urlparse

        try:
            domain = urlparse(url).netloc.lower()
            return any(domain == allowed or domain.endswith(f".{allowed}") for allowed in self._allowed_domains)
        except Exception:
            return False


# Global instance
_search_instance: DomainRestrictedSearch | None = None


def get_domain_search() -> DomainRestrictedSearch:
    """Get the global domain-restricted search instance.

    Returns:
        DomainRestrictedSearch singleton instance
    """
    global _search_instance
    if _search_instance is None:
        _search_instance = DomainRestrictedSearch()
    return _search_instance
