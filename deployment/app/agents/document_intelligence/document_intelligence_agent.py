"""Document Intelligence Agent for comprehensive document analysis.

This agent provides multi-format document ingestion, RAG-based querying,
restricted web search, and multi-lingual support.

Following Enterprise Development Standards:
- Software Architect: Modular, extensible design
- Security Architect: Domain-restricted search, safe file handling
- Data Architect: Session-scoped storage, vector embeddings
- Software Engineer: Full type hints, error handling
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langsmith import traceable

from app.agents.base.agent_base import BaseAgent, AgentConfig
from app.agents.document_intelligence.state import DocumentIntelligenceState
from app.agents.document_intelligence.tools import (
    ALL_TOOLS,
    set_active_session,
)

logger = logging.getLogger(__name__)


class DocumentIntelligenceAgent(BaseAgent):
    """Document Intelligence Agent for comprehensive document analysis.

    Features:
    - Multi-format document ingestion (PDF, TXT, DOCX, PPTX, PNG with OCR)
    - RAG-based semantic search across documents
    - Domain-restricted web search
    - LLM-based translation
    - Session memory and conversation continuity

    Example:
        >>> agent = DocumentIntelligenceAgent()
        >>> result = agent.invoke("What documents do I have?", session_id="user-123")
        >>> print(agent.get_last_response(result))

        # With document upload (API-side)
        >>> agent.invoke("Summarize the uploaded document", session_id="user-123")
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize the Document Intelligence Agent.

        Args:
            config: Agent configuration. Uses defaults if not provided.
        """
        super().__init__(config)

        # CRITICAL: Set recursion limit for complex multi-tool workflows
        self._recursion_limit = 100

        # Register all tools
        self.register_tools(ALL_TOOLS)

        logger.info("DocumentIntelligenceAgent initialized with %d tools", len(ALL_TOOLS))

    def _get_system_prompt(self) -> str:
        """Get the agent's system prompt.

        Returns:
            Comprehensive system prompt for document intelligence operations
        """
        return """You are a Document Intelligence Agent specializing in document analysis,
information retrieval, and multi-lingual support.

## Your Capabilities:

### 1. Document Management
- **upload_document**: Process PDF, TXT, DOCX, PPTX, and images (PNG/JPG with OCR)
- **list_documents**: View all uploaded documents with metadata
- **clear_documents**: Remove documents from the session

### 2. Document Analysis
- **search_documents**: Semantic search across uploaded documents
- **summarize_document**: Generate brief, detailed, or executive summaries

### 3. Web Research
- **web_search**: Search the web within allowed domains
  (Restricted to domains configured by the administrator)

### 4. Language Support
- **translate_text**: Translate text between 25+ languages
- **detect_language**: Identify the language of any text

## Process Guidelines:

1. **For document questions**:
   - First check if documents are loaded (use list_documents)
   - Use search_documents to find relevant content
   - Synthesize answers from retrieved chunks
   - Always cite which document the information came from

2. **For web questions**:
   - Use web_search for current information
   - Combine with document search if relevant
   - Note when information comes from web vs documents

3. **For translation requests**:
   - Detect source language if not specified
   - Translate accurately while preserving meaning
   - Note the language pair used

4. **For summarization**:
   - Use appropriate summary type (brief/detailed/executive)
   - Offer to translate summaries if user's language differs

## Response Guidelines:

- Start with a direct answer to the user's question
- Include relevant quotes/excerpts when appropriate
- Always cite sources (document names or web URLs)
- Offer to elaborate, translate, or search further
- Be clear about limitations (e.g., if no documents loaded, if domain not in allowed list)

## Language Support:

You can process and respond in multiple languages. Supported languages include:
English, Spanish, French, German, Italian, Portuguese, Dutch, Russian,
Chinese, Japanese, Korean, Arabic, Hindi, and many more.

If the user writes in a non-English language:
1. Detect their language
2. Respond in the same language
3. Search documents in their language when possible"""

    def _build_graph(self) -> StateGraph:
        """Build the agent's workflow graph.

        Returns:
            Configured StateGraph with ReAct pattern
        """

        def call_model(state: DocumentIntelligenceState) -> dict[str, Any]:
            """Call the LLM with the current state."""
            system_prompt = SystemMessage(content=self._get_system_prompt())
            messages = [system_prompt] + list(state.messages)
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def should_continue(state: DocumentIntelligenceState) -> str:
            """Determine if we should continue to tools or end."""
            messages = list(state.messages)
            if not messages:
                return "end"

            last_message = messages[-1]

            # Check if the LLM wants to call tools
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"

            return "end"

        # Build the graph
        graph = StateGraph(DocumentIntelligenceState)

        # Add nodes
        graph.add_node("agent", call_model)
        graph.add_node("tools", ToolNode(self._tools))

        # Add edges
        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", "end": END}
        )
        graph.add_edge("tools", "agent")

        return graph

    @traceable(name="document_intelligence_invoke")
    def invoke(
        self,
        message: str,
        session_id: str | None = None,
        user_id: str | None = None,
        target_language: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Invoke the agent with a message.

        Args:
            message: User message to process
            session_id: Session ID for document isolation and continuity
            user_id: Optional user ID for personalization
            target_language: Optional target language for responses
            **kwargs: Additional state fields

        Returns:
            Agent response with messages and metadata
        """
        if self._compiled_graph is None:
            self.compile()

        # Set active session for tools
        effective_session = session_id or "default"
        set_active_session(effective_session)

        # Build input state
        input_state = {
            "messages": [HumanMessage(content=message)],
            "session_id": effective_session,
            "user_id": user_id,
            "target_language": target_language or "en",
            **kwargs,
        }

        # Configure with recursion limit
        config = {
            "configurable": {"thread_id": effective_session},
            "recursion_limit": self._recursion_limit,
        }

        # Invoke graph
        result = self._compiled_graph.invoke(input_state, config=config)

        return result

    @traceable(name="document_intelligence_chat")
    def chat(
        self,
        message: str,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Simplified chat interface returning response string.

        Args:
            message: User message
            session_id: Session identifier
            **kwargs: Additional arguments

        Returns:
            Dict with response string and metadata
        """
        result = self.invoke(message, session_id=session_id, **kwargs)
        response = self.get_last_response(result)

        return {
            "response": response,
            "session_id": session_id or "default",
            "messages": result.get("messages", []),
        }

    def upload_document(
        self,
        content: bytes,
        filename: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Upload a document directly (for API use).

        Args:
            content: File content as bytes
            filename: Original filename
            session_id: Session identifier

        Returns:
            Upload result with document ID
        """
        import base64
        from app.agents.document_intelligence.tools import upload_document

        # Set session for tools
        effective_session = session_id or "default"
        set_active_session(effective_session)

        # Encode content and call tool
        content_b64 = base64.b64encode(content).decode("utf-8")

        result = upload_document.invoke({
            "content_base64": content_b64,
            "filename": filename,
        })

        return {
            "success": "successfully" in result.lower(),
            "message": result,
            "session_id": effective_session,
        }

    def get_documents(self, session_id: str | None = None) -> list[dict[str, Any]]:
        """Get all documents in a session.

        Args:
            session_id: Session identifier

        Returns:
            List of document metadata
        """
        from app.agents.document_intelligence.vector_store import get_vector_store

        effective_session = session_id or "default"
        vector_store = get_vector_store()
        return vector_store.get_documents(effective_session)

    def clear_documents(
        self,
        session_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> int:
        """Clear documents from a session.

        Args:
            session_id: Session identifier
            document_ids: Specific documents to clear, or None for all

        Returns:
            Number of documents cleared
        """
        from app.agents.document_intelligence.vector_store import get_vector_store

        effective_session = session_id or "default"
        vector_store = get_vector_store()
        return vector_store.clear_documents(effective_session, document_ids)
