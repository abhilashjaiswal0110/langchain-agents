"""Domain Router for fast intent-based request routing.

Provides lightweight routing without full supervisor overhead:
- Keyword-based intent classification
- LLM-based classification for ambiguous cases
- Direct routing to domain agents
- Confidence scoring for routing decisions

Following Enterprise Development Standards:
- Software Architect: Router pattern for simple requests
- Software Engineer: Fast, efficient classification
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from pydantic import BaseModel, Field

from app.agents.base.llm_factory import get_llm


class DomainIntent(str, Enum):
    """Domain intent classifications."""

    MARCOM = "marcom"
    HR = "hr"
    LND = "lnd"
    PRESALES = "presales"
    DATACENTER = "datacenter"
    CLOUD = "cloud"
    CYBERSECURITY = "cybersecurity"
    DATA_AI = "data_ai"
    FINANCE = "finance"
    GENERAL = "general"
    UNKNOWN = "unknown"


@dataclass
class RoutingResult:
    """Result of routing classification.

    Attributes:
        intent: Classified domain intent
        confidence: Confidence score (0.0 - 1.0)
        keywords_matched: Keywords that triggered the classification
        requires_supervisor: Whether full supervisor is needed
        reasoning: Brief explanation of the routing decision
    """

    intent: DomainIntent
    confidence: float
    keywords_matched: list[str]
    requires_supervisor: bool
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "keywords_matched": self.keywords_matched,
            "requires_supervisor": self.requires_supervisor,
            "reasoning": self.reasoning,
        }


class ClassificationResult(BaseModel):
    """Structured output for LLM classification."""

    domain: DomainIntent = Field(description="Primary domain for the request")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in classification")
    reasoning: str = Field(description="Brief reasoning for classification")
    is_complex: bool = Field(description="Whether request needs full supervisor")


# Domain keywords for fast classification
DOMAIN_KEYWORDS: dict[DomainIntent, list[str]] = {
    DomainIntent.MARCOM: [
        "marketing", "campaign", "brand", "branding", "content",
        "social media", "press release", "communications", "pr",
        "newsletter", "advertising", "ads", "creative", "design",
        "collateral", "logo", "messaging", "launch",
    ],
    DomainIntent.HR: [
        "hr", "human resources", "benefits", "payroll", "salary",
        "vacation", "pto", "leave", "onboarding", "offboarding",
        "hiring", "recruit", "performance review", "compensation",
        "employee", "policy", "handbook", "contract", "termination",
    ],
    DomainIntent.LND: [
        "training", "course", "learning", "certification", "certificate",
        "skill", "development", "workshop", "webinar", "education",
        "tutorial", "lesson", "exam", "assessment", "competency",
    ],
    DomainIntent.PRESALES: [
        "presales", "sales", "demo", "proposal", "rfp", "rfi",
        "quote", "pricing", "customer", "client", "prospect",
        "pitch", "presentation", "poc", "proof of concept",
    ],
    DomainIntent.DATACENTER: [
        "datacenter", "data center", "server", "rack", "storage",
        "san", "nas", "backup", "physical", "hardware", "cooling",
        "power", "ups", "network switch", "cable", "facility",
    ],
    DomainIntent.CLOUD: [
        "cloud", "azure", "aws", "gcp", "vm", "virtual machine",
        "container", "kubernetes", "docker", "iaas", "paas", "saas",
        "serverless", "function", "lambda", "storage account",
        "blob", "s3", "ec2", "aks", "eks", "gke",
    ],
    DomainIntent.CYBERSECURITY: [
        "security", "cybersecurity", "cyber", "vulnerability", "patch",
        "firewall", "antivirus", "malware", "phishing", "incident",
        "breach", "access control", "iam", "mfa", "2fa", "compliance",
        "audit", "pentest", "penetration", "soc", "siem", "threat",
    ],
    DomainIntent.DATA_AI: [
        "data", "analytics", "ai", "artificial intelligence", "ml",
        "machine learning", "model", "dataset", "pipeline", "etl",
        "bi", "business intelligence", "dashboard", "report",
        "visualization", "prediction", "llm", "nlp", "chatbot",
    ],
    DomainIntent.FINANCE: [
        "finance", "budget", "invoice", "expense", "accounts payable",
        "accounts receivable", "gl", "general ledger", "cost centre",
        "fiscal", "reimbursement", "purchase order", "po",
        "financial report", "variance", "actuals", "forecast",
    ],
    DomainIntent.GENERAL: [
        "password", "reset", "account", "login", "access", "vpn",
        "email", "outlook", "teams", "office", "software", "install",
        "printer", "wifi", "network", "laptop", "computer", "pc",
        "monitor", "keyboard", "mouse", "desk", "equipment",
    ],
}


class DomainRouter:
    """Fast router for domain classification.

    Uses a two-phase approach:
    1. Keyword matching for obvious cases (fast, no LLM call)
    2. LLM classification for ambiguous cases (accurate, slower)

    Example:
        >>> router = DomainRouter()
        >>> result = router.classify("I need help with Azure VMs")
        >>> print(result.intent)  # DomainIntent.CLOUD
        >>> print(result.confidence)  # 0.95
    """

    # Threshold for keyword-based classification
    KEYWORD_CONFIDENCE_THRESHOLD = 0.7

    # Threshold below which to use full supervisor
    SUPERVISOR_THRESHOLD = 0.5

    def __init__(self, llm: Any = None) -> None:
        """Initialize domain router.

        Args:
            llm: LangChain LLM for ambiguous classification.
        """
        self._llm = llm
        self._classification_chain = None

    def _get_llm(self) -> Any:
        """Get or create LLM instance."""
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def _create_classification_chain(self) -> Any:
        """Create LLM classification chain."""
        if self._classification_chain is not None:
            return self._classification_chain

        llm = self._get_llm()

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an IT request classifier. Classify the user request into ONE of these domains:
- marcom: Marketing & Communications (campaigns, branding, content)
- hr: Human Resources (benefits, policies, payroll, onboarding)
- lnd: Learning & Development (training, certifications, courses)
- presales: Presales/Sales (demos, proposals, customer inquiries)
- datacenter: Datacenter operations (servers, storage, physical infrastructure)
- cloud: Cloud infrastructure (Azure, AWS, GCP, VMs, containers)
- cybersecurity: Security (incidents, vulnerabilities, access control)
- data_ai: Data & AI (analytics, ML, data pipelines)
- general: General IT (account access, software, basic tech issues)
- unknown: Cannot determine

Set is_complex=true if the request:
- Spans multiple domains
- Requires human judgment
- Is sensitive or urgent
- Needs clarification

Respond with your classification."""),
            ("human", "{message}"),
        ])

        self._classification_chain = prompt | llm.with_structured_output(ClassificationResult)
        return self._classification_chain

    def _keyword_classify(self, text: str) -> RoutingResult | None:
        """Classify using keyword matching.

        Args:
            text: User message text.

        Returns:
            RoutingResult if confident, None if ambiguous.
        """
        text_lower = text.lower()
        scores: dict[DomainIntent, tuple[float, list[str]]] = {}

        for domain, keywords in DOMAIN_KEYWORDS.items():
            matched = []
            for keyword in keywords:
                if keyword in text_lower:
                    matched.append(keyword)

            if matched:
                # Score based on number and length of matches
                score = len(matched) / len(keywords)
                # Boost for longer keyword matches
                avg_len = sum(len(k) for k in matched) / len(matched)
                score = min(1.0, score * (1 + avg_len / 20))
                scores[domain] = (score, matched)

        if not scores:
            return None

        # Get top domain
        top_domain = max(scores.keys(), key=lambda d: scores[d][0])
        top_score, matched_keywords = scores[top_domain]

        # Check if confident enough
        if top_score >= self.KEYWORD_CONFIDENCE_THRESHOLD:
            return RoutingResult(
                intent=top_domain,
                confidence=top_score,
                keywords_matched=matched_keywords,
                requires_supervisor=False,
                reasoning=f"Keyword match: {', '.join(matched_keywords[:3])}",
            )

        # Check for multiple domains with similar scores (ambiguous)
        sorted_domains = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
        if len(sorted_domains) > 1:
            second_score = sorted_domains[1][1][0]
            if abs(top_score - second_score) < 0.2:
                # Ambiguous - needs LLM or supervisor
                return None

        return RoutingResult(
            intent=top_domain,
            confidence=top_score,
            keywords_matched=matched_keywords,
            requires_supervisor=top_score < self.SUPERVISOR_THRESHOLD,
            reasoning=f"Keyword match (low confidence): {', '.join(matched_keywords[:3])}",
        )

    @traceable(name="domain_router_classify")
    def classify(self, message: str | BaseMessage) -> RoutingResult:
        """Classify a message to a domain.

        Args:
            message: User message (string or BaseMessage).

        Returns:
            RoutingResult with domain and confidence.
        """
        if isinstance(message, BaseMessage):
            text = message.content if hasattr(message, "content") else str(message)
        else:
            text = message

        # Try keyword classification first
        keyword_result = self._keyword_classify(text)
        if keyword_result and keyword_result.confidence >= self.KEYWORD_CONFIDENCE_THRESHOLD:
            return keyword_result

        # Fall back to LLM classification
        return self._llm_classify_sync(text)

    @traceable(name="domain_router_classify_async")
    async def aclassify(self, message: str | BaseMessage) -> RoutingResult:
        """Async classify a message to a domain.

        Args:
            message: User message.

        Returns:
            RoutingResult with domain and confidence.
        """
        if isinstance(message, BaseMessage):
            text = message.content if hasattr(message, "content") else str(message)
        else:
            text = message

        # Try keyword classification first
        keyword_result = self._keyword_classify(text)
        if keyword_result and keyword_result.confidence >= self.KEYWORD_CONFIDENCE_THRESHOLD:
            return keyword_result

        # Fall back to LLM classification
        return await self._llm_classify_async(text)

    def _llm_classify_sync(self, text: str) -> RoutingResult:
        """Synchronous LLM classification.

        Args:
            text: Message text.

        Returns:
            RoutingResult from LLM.
        """
        chain = self._create_classification_chain()

        try:
            result = chain.invoke({"message": text})
            return RoutingResult(
                intent=result.domain,
                confidence=result.confidence,
                keywords_matched=[],
                requires_supervisor=result.is_complex or result.confidence < self.SUPERVISOR_THRESHOLD,
                reasoning=result.reasoning,
            )
        except Exception as e:
            return RoutingResult(
                intent=DomainIntent.UNKNOWN,
                confidence=0.0,
                keywords_matched=[],
                requires_supervisor=True,
                reasoning=f"Classification failed: {e}",
            )

    async def _llm_classify_async(self, text: str) -> RoutingResult:
        """Async LLM classification.

        Args:
            text: Message text.

        Returns:
            RoutingResult from LLM.
        """
        chain = self._create_classification_chain()

        try:
            result = await chain.ainvoke({"message": text})
            return RoutingResult(
                intent=result.domain,
                confidence=result.confidence,
                keywords_matched=[],
                requires_supervisor=result.is_complex or result.confidence < self.SUPERVISOR_THRESHOLD,
                reasoning=result.reasoning,
            )
        except Exception as e:
            return RoutingResult(
                intent=DomainIntent.UNKNOWN,
                confidence=0.0,
                keywords_matched=[],
                requires_supervisor=True,
                reasoning=f"Classification failed: {e}",
            )


# Singleton instance
_router: DomainRouter | None = None


def get_domain_router(llm: Any = None) -> DomainRouter:
    """Get or create domain router singleton.

    Args:
        llm: Optional LLM for classification.

    Returns:
        DomainRouter instance.
    """
    global _router
    if _router is None:
        _router = DomainRouter(llm=llm)
    return _router
