"""Evaluation datasets for enterprise IT agents.

This module provides test datasets for evaluating agent performance
across different scenarios and use cases.

Following Enterprise Development Standards:
- Data Architect: Structured test data organization
- Software Engineer: Type-safe dataset definitions
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TestCase:
    """A single test case for evaluation."""

    id: str
    input: str
    expected_output: str | None = None
    expected_keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalDataset:
    """A collection of test cases for a specific agent."""

    name: str
    description: str
    agent_type: str
    test_cases: list[TestCase] = field(default_factory=list)
    version: str = "1.0"


# =============================================================================
# Research Agent Dataset
# =============================================================================

RESEARCH_AGENT_DATASET = EvalDataset(
    name="research_agent_eval",
    description="Test cases for AI Research Agent",
    agent_type="research",
    test_cases=[
        TestCase(
            id="research_001",
            input="What are the latest trends in AI agent development?",
            expected_keywords=["LLM", "agents", "automation", "AI"],
            tags=["general", "trends"],
            difficulty="easy",
        ),
        TestCase(
            id="research_002",
            input="Compare LangChain vs LangGraph for building agents",
            expected_keywords=["LangChain", "LangGraph", "comparison", "agents"],
            tags=["comparison", "technical"],
            difficulty="medium",
        ),
        TestCase(
            id="research_003",
            input="Explain how RAG systems work with vector databases",
            expected_keywords=["RAG", "vector", "embeddings", "retrieval"],
            tags=["technical", "rag"],
            difficulty="medium",
        ),
        TestCase(
            id="research_004",
            input="What are best practices for securing AI agents in production?",
            expected_keywords=["security", "production", "authentication", "API"],
            tags=["security", "best-practices"],
            difficulty="hard",
        ),
    ],
)


# =============================================================================
# Content Agent Dataset
# =============================================================================

CONTENT_AGENT_DATASET = EvalDataset(
    name="content_agent_eval",
    description="Test cases for Content Generation Agent",
    agent_type="content",
    test_cases=[
        TestCase(
            id="content_001",
            input="Write a LinkedIn post about AI automation in IT support",
            expected_keywords=["AI", "automation", "IT", "support"],
            tags=["linkedin", "professional"],
            difficulty="easy",
        ),
        TestCase(
            id="content_002",
            input="Create a technical blog post about implementing LangGraph agents",
            expected_keywords=["LangGraph", "implementation", "code", "agents"],
            tags=["blog", "technical"],
            difficulty="medium",
        ),
        TestCase(
            id="content_003",
            input="Write a tweet thread about the benefits of AI-powered document generation",
            expected_keywords=["AI", "document", "automation", "benefits"],
            tags=["twitter", "social"],
            difficulty="easy",
        ),
    ],
)


# =============================================================================
# Data Analyst Agent Dataset
# =============================================================================

DATA_ANALYST_DATASET = EvalDataset(
    name="data_analyst_eval",
    description="Test cases for Data Analyst Agent",
    agent_type="data_analyst",
    test_cases=[
        TestCase(
            id="data_001",
            input="Analyze the sales data and provide insights on trends",
            expected_keywords=["trend", "analysis", "sales", "insight"],
            tags=["analysis", "trends"],
            difficulty="medium",
        ),
        TestCase(
            id="data_002",
            input="Calculate the correlation between marketing spend and revenue",
            expected_keywords=["correlation", "marketing", "revenue", "analysis"],
            tags=["statistical", "correlation"],
            difficulty="hard",
        ),
        TestCase(
            id="data_003",
            input="Generate a summary of the monthly performance metrics",
            expected_keywords=["summary", "performance", "metrics", "monthly"],
            tags=["summary", "reporting"],
            difficulty="easy",
        ),
    ],
)


# =============================================================================
# Document Agent Dataset
# =============================================================================

DOCUMENT_AGENT_DATASET = EvalDataset(
    name="document_agent_eval",
    description="Test cases for IT Document Generator Agent",
    agent_type="document",
    test_cases=[
        TestCase(
            id="doc_001",
            input="Generate an SOP for password reset procedure",
            expected_keywords=["password", "reset", "procedure", "step"],
            tags=["sop", "security"],
            difficulty="easy",
        ),
        TestCase(
            id="doc_002",
            input="Create a WLI for server maintenance tasks",
            expected_keywords=["server", "maintenance", "instruction", "step"],
            tags=["wli", "infrastructure"],
            difficulty="medium",
        ),
        TestCase(
            id="doc_003",
            input="Generate an IT security policy for remote work",
            expected_keywords=["security", "remote", "policy", "compliance"],
            tags=["policy", "security"],
            difficulty="hard",
        ),
    ],
)


# =============================================================================
# Multilingual RAG Agent Dataset
# =============================================================================

MULTILINGUAL_RAG_DATASET = EvalDataset(
    name="multilingual_rag_eval",
    description="Test cases for Multilingual RAG Agent",
    agent_type="multilingual_rag",
    test_cases=[
        TestCase(
            id="rag_001",
            input="What is the main topic of the uploaded document?",
            expected_keywords=["document", "topic", "content"],
            tags=["basic", "summarization"],
            difficulty="easy",
        ),
        TestCase(
            id="rag_002",
            input="Finden Sie relevante Informationen zu diesem Thema (German query)",
            expected_keywords=["Information", "relevant"],
            tags=["multilingual", "german"],
            difficulty="medium",
        ),
        TestCase(
            id="rag_003",
            input="Summarize the key points across all uploaded documents",
            expected_keywords=["summary", "key", "points"],
            tags=["summarization", "multi-doc"],
            difficulty="hard",
        ),
    ],
)


# =============================================================================
# HITL Support Agent Dataset
# =============================================================================

HITL_SUPPORT_DATASET = EvalDataset(
    name="hitl_support_eval",
    description="Test cases for Human-in-the-Loop IT Support Agent",
    agent_type="hitl_support",
    test_cases=[
        TestCase(
            id="hitl_001",
            input="I cannot access my email, please help",
            expected_keywords=["email", "access", "help", "ticket"],
            tags=["basic", "email"],
            difficulty="easy",
        ),
        TestCase(
            id="hitl_002",
            input="Need to reset admin password for the production server",
            expected_keywords=["password", "reset", "approval", "admin"],
            tags=["sensitive", "requires-approval"],
            difficulty="hard",
        ),
        TestCase(
            id="hitl_003",
            input="My VPN connection keeps dropping",
            expected_keywords=["VPN", "connection", "troubleshoot"],
            tags=["network", "vpn"],
            difficulty="medium",
        ),
    ],
)


# =============================================================================
# Code Assistant Agent Dataset
# =============================================================================

CODE_ASSISTANT_DATASET = EvalDataset(
    name="code_assistant_eval",
    description="Test cases for Code Assistant Agent",
    agent_type="code_assistant",
    test_cases=[
        TestCase(
            id="code_001",
            input="def old_function(x):\n    return x * 2",
            expected_keywords=["modernize", "type", "hint"],
            tags=["modernization", "python"],
            difficulty="easy",
        ),
        TestCase(
            id="code_002",
            input="SELECT * FROM users WHERE id = '" + "user_input" + "'",
            expected_keywords=["SQL", "injection", "security", "vulnerability"],
            tags=["security", "sql"],
            difficulty="hard",
        ),
        TestCase(
            id="code_003",
            input="function fetchData() { return fetch('/api/data').then(r => r.json()) }",
            expected_keywords=["async", "await", "modern", "JavaScript"],
            tags=["modernization", "javascript"],
            difficulty="medium",
        ),
    ],
)


# =============================================================================
# Dataset Registry
# =============================================================================

ALL_DATASETS: dict[str, EvalDataset] = {
    "research": RESEARCH_AGENT_DATASET,
    "content": CONTENT_AGENT_DATASET,
    "data_analyst": DATA_ANALYST_DATASET,
    "document": DOCUMENT_AGENT_DATASET,
    "multilingual_rag": MULTILINGUAL_RAG_DATASET,
    "hitl_support": HITL_SUPPORT_DATASET,
    "code_assistant": CODE_ASSISTANT_DATASET,
}


def get_dataset(agent_type: str) -> EvalDataset | None:
    """Get evaluation dataset for an agent type.

    Args:
        agent_type: The type of agent

    Returns:
        EvalDataset if found, None otherwise
    """
    return ALL_DATASETS.get(agent_type)


def get_test_cases_by_tag(tag: str) -> list[TestCase]:
    """Get all test cases with a specific tag.

    Args:
        tag: The tag to filter by

    Returns:
        List of matching test cases
    """
    cases = []
    for dataset in ALL_DATASETS.values():
        for test_case in dataset.test_cases:
            if tag in test_case.tags:
                cases.append(test_case)
    return cases


def get_test_cases_by_difficulty(difficulty: str) -> list[TestCase]:
    """Get all test cases with a specific difficulty.

    Args:
        difficulty: easy, medium, or hard

    Returns:
        List of matching test cases
    """
    cases = []
    for dataset in ALL_DATASETS.values():
        for test_case in dataset.test_cases:
            if test_case.difficulty == difficulty:
                cases.append(test_case)
    return cases
