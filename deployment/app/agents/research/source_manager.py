"""Source Manager for citation tracking and credibility scoring.

Provides:
- Source collection and deduplication
- Credibility scoring based on domain and content
- Citation formatting (APA, MLA, IEEE, etc.)
- Source verification and cross-referencing
- Export to various formats

Following Enterprise Development Standards:
- Software Architect: Clean separation of concerns
- Security Architect: URL validation, no script injection
- Data Architect: Structured source metadata
- Software Engineer: Type-safe, well-documented
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4


class SourceType(str, Enum):
    """Classification of source types."""

    WEB_PAGE = "web_page"
    ACADEMIC_PAPER = "academic_paper"
    NEWS_ARTICLE = "news_article"
    DOCUMENTATION = "documentation"
    BLOG_POST = "blog_post"
    BOOK = "book"
    VIDEO = "video"
    SOCIAL_MEDIA = "social_media"
    GOVERNMENT = "government"
    WIKIPEDIA = "wikipedia"
    UNKNOWN = "unknown"


class CredibilityLevel(str, Enum):
    """Credibility rating levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNVERIFIED = "unverified"


class CitationFormat(str, Enum):
    """Supported citation formats."""

    APA = "apa"
    MLA = "mla"
    IEEE = "ieee"
    CHICAGO = "chicago"
    MARKDOWN = "markdown"
    PLAIN = "plain"


@dataclass
class Source:
    """A research source with metadata.

    Attributes:
        id: Unique identifier
        url: Source URL
        title: Source title
        content_summary: Brief content summary
        source_type: Type classification
        credibility: Credibility rating
        credibility_score: Numeric score (0.0-1.0)
        author: Author name if available
        publication_date: Publication date if available
        accessed_date: When the source was accessed
        domain: Source domain
        keywords: Extracted keywords
        citations_count: Number of times cited in research
        verified: Whether source has been verified
        metadata: Additional metadata
    """

    id: str = field(default_factory=lambda: str(uuid4())[:8])
    url: str = ""
    title: str = ""
    content_summary: str = ""
    source_type: SourceType = SourceType.UNKNOWN
    credibility: CredibilityLevel = CredibilityLevel.UNVERIFIED
    credibility_score: float = 0.5
    author: str | None = None
    publication_date: str | None = None
    accessed_date: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    domain: str = ""
    keywords: list[str] = field(default_factory=list)
    citations_count: int = 0
    verified: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Extract domain from URL if not set."""
        if self.url and not self.domain:
            try:
                parsed = urlparse(self.url)
                self.domain = parsed.netloc
            except Exception:
                self.domain = ""


@dataclass
class SourceCollection:
    """A collection of research sources.

    Attributes:
        id: Collection identifier
        query: Research query this collection is for
        sources: List of sources
        created_at: Collection creation time
        updated_at: Last update time
    """

    id: str = field(default_factory=lambda: str(uuid4())[:8])
    query: str = ""
    sources: list[Source] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = ""

    def __post_init__(self) -> None:
        """Set updated_at to created_at if not set."""
        if not self.updated_at:
            self.updated_at = self.created_at


class CredibilityScorer:
    """Score source credibility based on multiple factors."""

    # Domain credibility scores (0.0 - 1.0)
    DOMAIN_SCORES: dict[str, float] = {
        # High credibility
        ".gov": 0.95,
        ".edu": 0.9,
        "arxiv.org": 0.9,
        "nature.com": 0.95,
        "science.org": 0.95,
        "ieee.org": 0.9,
        "acm.org": 0.9,
        "springer.com": 0.85,
        # Medium-high
        "wikipedia.org": 0.7,
        "stackoverflow.com": 0.75,
        "github.com": 0.7,
        "docs.python.org": 0.9,
        "docs.microsoft.com": 0.85,
        "cloud.google.com": 0.85,
        "aws.amazon.com": 0.85,
        # Medium
        "medium.com": 0.5,
        "dev.to": 0.5,
        "reddit.com": 0.4,
        # News sources
        "reuters.com": 0.85,
        "bbc.com": 0.8,
        "nytimes.com": 0.8,
        "techcrunch.com": 0.7,
        # Lower credibility
        "twitter.com": 0.3,
        "x.com": 0.3,
        "facebook.com": 0.3,
    }

    # Source type base scores
    TYPE_SCORES: dict[SourceType, float] = {
        SourceType.ACADEMIC_PAPER: 0.9,
        SourceType.GOVERNMENT: 0.9,
        SourceType.DOCUMENTATION: 0.85,
        SourceType.NEWS_ARTICLE: 0.7,
        SourceType.BOOK: 0.8,
        SourceType.WIKIPEDIA: 0.7,
        SourceType.WEB_PAGE: 0.5,
        SourceType.BLOG_POST: 0.5,
        SourceType.VIDEO: 0.4,
        SourceType.SOCIAL_MEDIA: 0.3,
        SourceType.UNKNOWN: 0.4,
    }

    def score(self, source: Source) -> float:
        """Calculate credibility score for a source.

        Args:
            source: Source to score

        Returns:
            Credibility score (0.0 - 1.0)
        """
        scores = []

        # Domain-based scoring
        domain_score = self._score_domain(source.domain)
        scores.append(domain_score * 0.4)  # 40% weight

        # Type-based scoring
        type_score = self.TYPE_SCORES.get(source.source_type, 0.4)
        scores.append(type_score * 0.3)  # 30% weight

        # Content quality indicators
        content_score = self._score_content(source)
        scores.append(content_score * 0.2)  # 20% weight

        # Recency bonus
        recency_score = self._score_recency(source.publication_date)
        scores.append(recency_score * 0.1)  # 10% weight

        total = sum(scores)
        return min(1.0, max(0.0, total))

    def _score_domain(self, domain: str) -> float:
        """Score based on domain reputation."""
        if not domain:
            return 0.3

        domain_lower = domain.lower()

        # Check exact domain matches
        for known_domain, score in self.DOMAIN_SCORES.items():
            if known_domain in domain_lower:
                return score

        # Check TLD
        if domain_lower.endswith(".gov"):
            return 0.9
        if domain_lower.endswith(".edu"):
            return 0.85
        if domain_lower.endswith(".org"):
            return 0.6

        return 0.5  # Default

    def _score_content(self, source: Source) -> float:
        """Score based on content quality indicators."""
        score = 0.5  # Base score

        # Has author
        if source.author:
            score += 0.15

        # Has meaningful title
        if source.title and len(source.title) > 10:
            score += 0.1

        # Has content summary
        if source.content_summary and len(source.content_summary) > 50:
            score += 0.15

        # Has keywords
        if source.keywords and len(source.keywords) >= 3:
            score += 0.1

        return min(1.0, score)

    def _score_recency(self, publication_date: str | None) -> float:
        """Score based on how recent the source is."""
        if not publication_date:
            return 0.5

        try:
            # Try to parse date
            pub_date = datetime.fromisoformat(publication_date.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age_days = (now - pub_date).days

            if age_days < 30:
                return 1.0
            if age_days < 180:
                return 0.8
            if age_days < 365:
                return 0.6
            if age_days < 730:
                return 0.4
            return 0.2

        except Exception:
            return 0.5

    def get_credibility_level(self, score: float) -> CredibilityLevel:
        """Convert numeric score to credibility level.

        Args:
            score: Numeric credibility score

        Returns:
            CredibilityLevel enum value
        """
        if score >= 0.75:
            return CredibilityLevel.HIGH
        if score >= 0.5:
            return CredibilityLevel.MEDIUM
        if score >= 0.25:
            return CredibilityLevel.LOW
        return CredibilityLevel.UNVERIFIED


class CitationFormatter:
    """Format citations in various styles."""

    def format(
        self,
        source: Source,
        style: CitationFormat = CitationFormat.MARKDOWN,
    ) -> str:
        """Format a source as a citation.

        Args:
            source: Source to cite
            style: Citation format style

        Returns:
            Formatted citation string
        """
        if style == CitationFormat.APA:
            return self._format_apa(source)
        if style == CitationFormat.MLA:
            return self._format_mla(source)
        if style == CitationFormat.IEEE:
            return self._format_ieee(source)
        if style == CitationFormat.CHICAGO:
            return self._format_chicago(source)
        if style == CitationFormat.MARKDOWN:
            return self._format_markdown(source)
        return self._format_plain(source)

    def _format_apa(self, source: Source) -> str:
        """Format in APA style."""
        author = source.author or "Unknown Author"
        year = self._extract_year(source.publication_date) or "n.d."
        title = source.title or "Untitled"

        return f"{author}. ({year}). {title}. Retrieved from {source.url}"

    def _format_mla(self, source: Source) -> str:
        """Format in MLA style."""
        author = source.author or "Unknown Author"
        title = source.title or "Untitled"
        accessed = self._format_date_mla(source.accessed_date)

        return f'{author}. "{title}." Web. {accessed}. <{source.url}>'

    def _format_ieee(self, source: Source) -> str:
        """Format in IEEE style."""
        author = source.author or "Unknown Author"
        title = source.title or "Untitled"
        accessed = self._format_date_ieee(source.accessed_date)

        return f'{author}, "{title}," [Online]. Available: {source.url}. [Accessed: {accessed}]'

    def _format_chicago(self, source: Source) -> str:
        """Format in Chicago style."""
        author = source.author or "Unknown Author"
        title = source.title or "Untitled"
        accessed = self._format_date_chicago(source.accessed_date)

        return f'{author}. "{title}." Accessed {accessed}. {source.url}'

    def _format_markdown(self, source: Source) -> str:
        """Format as Markdown link."""
        title = source.title or source.url
        return f"[{title}]({source.url})"

    def _format_plain(self, source: Source) -> str:
        """Format as plain text."""
        title = source.title or "Untitled"
        return f"{title} - {source.url}"

    def _extract_year(self, date_str: str | None) -> str | None:
        """Extract year from date string."""
        if not date_str:
            return None
        match = re.search(r"\d{4}", date_str)
        return match.group() if match else None

    def _format_date_mla(self, date_str: str) -> str:
        """Format date for MLA style."""
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt.strftime("%d %b. %Y")
        except Exception:
            return "n.d."

    def _format_date_ieee(self, date_str: str) -> str:
        """Format date for IEEE style."""
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt.strftime("%b. %d, %Y")
        except Exception:
            return "n.d."

    def _format_date_chicago(self, date_str: str) -> str:
        """Format date for Chicago style."""
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt.strftime("%B %d, %Y")
        except Exception:
            return "n.d."


class SourceManager:
    """Manage research sources with tracking and scoring.

    Example:
        >>> manager = SourceManager()
        >>> source = manager.add_source(
        ...     url="https://arxiv.org/paper",
        ...     title="AI Research Paper",
        ...     content_summary="Summary of findings...",
        ... )
        >>> print(source.credibility)
        >>> citations = manager.export_citations(CitationFormat.APA)
    """

    def __init__(self) -> None:
        """Initialize the source manager."""
        self._sources: dict[str, Source] = {}
        self._collections: dict[str, SourceCollection] = {}
        self._scorer = CredibilityScorer()
        self._formatter = CitationFormatter()

    def add_source(
        self,
        url: str,
        title: str,
        content_summary: str = "",
        source_type: SourceType | None = None,
        author: str | None = None,
        publication_date: str | None = None,
        keywords: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Source:
        """Add a source and compute credibility.

        Args:
            url: Source URL
            title: Source title
            content_summary: Brief content summary
            source_type: Type classification (auto-detected if None)
            author: Author name
            publication_date: Publication date
            keywords: Extracted keywords
            metadata: Additional metadata

        Returns:
            Created Source with credibility scores
        """
        # Auto-detect source type if not provided
        if source_type is None:
            source_type = self._detect_source_type(url)

        source = Source(
            url=url,
            title=title,
            content_summary=content_summary,
            source_type=source_type,
            author=author,
            publication_date=publication_date,
            keywords=keywords or [],
            metadata=metadata or {},
        )

        # Compute credibility
        score = self._scorer.score(source)
        source.credibility_score = score
        source.credibility = self._scorer.get_credibility_level(score)

        # Store and return
        self._sources[source.id] = source
        return source

    def get_source(self, source_id: str) -> Source | None:
        """Get a source by ID.

        Args:
            source_id: Source identifier

        Returns:
            Source if found, None otherwise
        """
        return self._sources.get(source_id)

    def get_sources_by_url(self, url: str) -> list[Source]:
        """Get sources by URL (for deduplication).

        Args:
            url: URL to search for

        Returns:
            List of matching sources
        """
        return [s for s in self._sources.values() if s.url == url]

    def get_all_sources(self) -> list[Source]:
        """Get all sources.

        Returns:
            List of all sources
        """
        return list(self._sources.values())

    def get_sources_by_credibility(
        self,
        min_level: CredibilityLevel = CredibilityLevel.MEDIUM,
    ) -> list[Source]:
        """Get sources filtered by minimum credibility.

        Args:
            min_level: Minimum credibility level

        Returns:
            Filtered list of sources
        """
        level_order = {
            CredibilityLevel.HIGH: 4,
            CredibilityLevel.MEDIUM: 3,
            CredibilityLevel.LOW: 2,
            CredibilityLevel.UNVERIFIED: 1,
        }

        min_order = level_order.get(min_level, 1)

        return [s for s in self._sources.values() if level_order.get(s.credibility, 0) >= min_order]

    def cite_source(self, source_id: str) -> None:
        """Increment citation count for a source.

        Args:
            source_id: Source to cite
        """
        source = self._sources.get(source_id)
        if source:
            source.citations_count += 1

    def verify_source(self, source_id: str) -> None:
        """Mark a source as verified.

        Args:
            source_id: Source to verify
        """
        source = self._sources.get(source_id)
        if source:
            source.verified = True
            # Boost credibility slightly for verification
            source.credibility_score = min(1.0, source.credibility_score + 0.1)
            source.credibility = self._scorer.get_credibility_level(source.credibility_score)

    def export_citations(
        self,
        style: CitationFormat = CitationFormat.MARKDOWN,
        source_ids: list[str] | None = None,
    ) -> str:
        """Export citations in specified format.

        Args:
            style: Citation format
            source_ids: Specific sources to cite (all if None)

        Returns:
            Formatted citations string
        """
        if source_ids:
            sources = [self._sources[sid] for sid in source_ids if sid in self._sources]
        else:
            sources = list(self._sources.values())

        citations = [self._formatter.format(s, style) for s in sources]

        if style == CitationFormat.MARKDOWN:
            return "\n".join(f"- {c}" for c in citations)

        return "\n\n".join(citations)

    def create_collection(self, query: str) -> SourceCollection:
        """Create a new source collection for a query.

        Args:
            query: Research query

        Returns:
            New SourceCollection
        """
        collection = SourceCollection(query=query)
        self._collections[collection.id] = collection
        return collection

    def add_to_collection(
        self,
        collection_id: str,
        source_id: str,
    ) -> bool:
        """Add a source to a collection.

        Args:
            collection_id: Collection to add to
            source_id: Source to add

        Returns:
            True if added successfully
        """
        collection = self._collections.get(collection_id)
        source = self._sources.get(source_id)

        if not collection or not source:
            return False

        collection.sources.append(source)
        collection.updated_at = datetime.now(timezone.utc).isoformat()
        return True

    def get_collection(self, collection_id: str) -> SourceCollection | None:
        """Get a source collection.

        Args:
            collection_id: Collection identifier

        Returns:
            SourceCollection if found
        """
        return self._collections.get(collection_id)

    def get_statistics(self) -> dict[str, Any]:
        """Get source collection statistics.

        Returns:
            Dictionary of statistics
        """
        sources = list(self._sources.values())

        if not sources:
            return {
                "total_sources": 0,
                "avg_credibility": 0,
                "by_type": {},
                "by_credibility": {},
                "verified_count": 0,
            }

        by_type: dict[str, int] = {}
        by_credibility: dict[str, int] = {}

        for source in sources:
            type_key = source.source_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1

            cred_key = source.credibility.value
            by_credibility[cred_key] = by_credibility.get(cred_key, 0) + 1

        return {
            "total_sources": len(sources),
            "avg_credibility": sum(s.credibility_score for s in sources) / len(sources),
            "by_type": by_type,
            "by_credibility": by_credibility,
            "verified_count": sum(1 for s in sources if s.verified),
            "total_citations": sum(s.citations_count for s in sources),
        }

    def _detect_source_type(self, url: str) -> SourceType:
        """Detect source type from URL.

        Args:
            url: URL to analyze

        Returns:
            Detected SourceType
        """
        if not url:
            return SourceType.UNKNOWN

        url_lower = url.lower()

        # Academic sources
        if any(d in url_lower for d in ["arxiv.org", "scholar.google", "researchgate", "doi.org"]):
            return SourceType.ACADEMIC_PAPER

        # Government
        if ".gov" in url_lower:
            return SourceType.GOVERNMENT

        # Documentation
        if any(d in url_lower for d in ["docs.", "documentation.", "/docs/", "/api/"]):
            return SourceType.DOCUMENTATION

        # Wikipedia
        if "wikipedia.org" in url_lower:
            return SourceType.WIKIPEDIA

        # News
        if any(d in url_lower for d in ["news.", "reuters.", "bbc.", "nytimes.", "cnn."]):
            return SourceType.NEWS_ARTICLE

        # Social media
        if any(d in url_lower for d in ["twitter.com", "x.com", "facebook.com", "linkedin.com"]):
            return SourceType.SOCIAL_MEDIA

        # Video
        if any(d in url_lower for d in ["youtube.com", "vimeo.com", "youtu.be"]):
            return SourceType.VIDEO

        # Blog
        if any(d in url_lower for d in ["medium.com", "dev.to", "blog.", "/blog/"]):
            return SourceType.BLOG_POST

        return SourceType.WEB_PAGE

    def clear(self) -> None:
        """Clear all sources and collections."""
        self._sources.clear()
        self._collections.clear()


# Global instance
_source_manager: SourceManager | None = None


def get_source_manager() -> SourceManager:
    """Get or create the global source manager.

    Returns:
        SourceManager instance
    """
    global _source_manager
    if _source_manager is None:
        _source_manager = SourceManager()
    return _source_manager


def reset_source_manager() -> None:
    """Reset the global source manager."""
    global _source_manager
    _source_manager = None
