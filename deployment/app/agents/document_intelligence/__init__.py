"""Document Intelligence Agent for multi-format document analysis.

This module provides comprehensive document processing, RAG-based querying,
restricted web search, and multi-lingual support.

Features:
- Multi-format ingestion: PDF, TXT, DOCX, PPTX, PNG/JPG (OCR)
- Semantic search with FAISS vector store
- Domain-restricted web search
- LLM-based translation (25+ languages)
- Session-scoped document storage
"""

from app.agents.document_intelligence.document_intelligence_agent import (
    DocumentIntelligenceAgent,
)
from app.agents.document_intelligence.state import DocumentIntelligenceState

__all__ = [
    "DocumentIntelligenceAgent",
    "DocumentIntelligenceState",
]
