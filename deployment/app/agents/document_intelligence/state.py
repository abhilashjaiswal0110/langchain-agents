"""State schema for the Document Intelligence Agent.

This module defines the state structure for document processing,
web search, and multi-lingual operations.

Following Enterprise Development Standards:
- Data Architect: Type-safe state management
- Software Engineer: Full type hints with Pydantic
"""

from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class DocumentIntelligenceState(BaseModel):
    """State schema for the Document Intelligence Agent.

    Extends BaseAgentState pattern with document processing,
    web search, and multi-lingual capabilities.
    """

    # Core message handling (from BaseAgentState pattern)
    messages: Annotated[list[BaseMessage], add_messages] = Field(
        default_factory=list, description="Conversation message history"
    )
    session_id: str | None = Field(default=None, description="Unique session identifier")
    user_id: str | None = Field(default=None, description="User identifier for personalization")

    # Document management
    documents: list[dict[str, Any]] = Field(
        default_factory=list, description="Loaded documents with metadata (id, filename, type, chunks, language)"
    )
    active_document_ids: list[str] = Field(
        default_factory=list, description="Currently active document IDs for querying"
    )

    # RAG state
    query: str = Field(default="", description="Current user query")
    retrieved_context: list[dict[str, Any]] = Field(
        default_factory=list, description="Retrieved document chunks with source info"
    )

    # Language handling
    detected_language: str = Field(default="en", description="Detected language of user input")
    target_language: str = Field(default="en", description="Target language for response")

    # Web search state
    web_search_results: list[dict[str, Any]] = Field(default_factory=list, description="Results from web searches")
    allowed_domains: list[str] = Field(default_factory=list, description="Allowed domains for web search")

    # Processing status
    status: Literal["idle", "processing", "searching", "translating", "complete"] = Field(
        default="idle", description="Current processing status"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional session metadata")
