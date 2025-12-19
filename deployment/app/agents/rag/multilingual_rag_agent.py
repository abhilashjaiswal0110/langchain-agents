"""Multilingual RAG Agent for document analysis and Q&A.

This agent provides:
- Document upload and processing
- Multi-language support with auto-detection
- Semantic search across documents
- Summarization and analysis
- Cross-lingual question answering

Following Enterprise Development Standards:
- Software Architect: Vector store integration
- Security Architect: Safe document handling
- Data Architect: Chunking and embedding strategies
- Software Engineer: Async-ready, type-safe
"""

import os
from datetime import datetime
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langsmith import traceable
from pydantic import BaseModel, Field

from app.agents.base.agent_base import BaseAgent, AgentConfig
from app.agents.base.tools import tool_error_handler, chunk_text


class RAGState(BaseModel):
    """State schema for the Multilingual RAG Agent."""

    messages: Annotated[list, add_messages] = Field(
        default_factory=list,
        description="Conversation history"
    )
    session_id: str | None = Field(default=None)
    user_id: str | None = Field(default=None)
    documents: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Loaded documents with metadata"
    )
    language: str = Field(
        default="auto",
        description="Detected/specified language"
    )
    query: str = Field(default="", description="Current query")
    context: list[str] = Field(
        default_factory=list,
        description="Retrieved context chunks"
    )
    response: str | None = Field(default=None, description="Generated response")


# In-memory document store (use proper vector DB in production)
_document_store: dict[str, list[dict[str, Any]]] = {}


def _detect_language(text: str) -> str:
    """Detect language of text."""
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"  # Default to English


@tool
@tool_error_handler
def upload_document(
    content: str,
    filename: str,
    doc_type: str = "text"
) -> str:
    """Upload and process a document for RAG.

    Args:
        content: Document content
        filename: Original filename
        doc_type: Document type (text/pdf/docx)

    Returns:
        Upload confirmation with document stats
    """
    # Detect language
    language = _detect_language(content[:1000])

    # Chunk the document
    chunks = chunk_text(content, chunk_size=1000)

    # Store document
    doc_id = f"doc_{len(_document_store) + 1}"
    _document_store[doc_id] = {
        "filename": filename,
        "type": doc_type,
        "language": language,
        "chunks": chunks,
        "chunk_count": len(chunks),
        "uploaded_at": datetime.now().isoformat(),
    }

    return (
        f"Document uploaded successfully!\n\n"
        f"Document ID: {doc_id}\n"
        f"Filename: {filename}\n"
        f"Language detected: {language}\n"
        f"Chunks created: {len(chunks)}\n"
        f"Total characters: {len(content)}"
    )


@tool
@tool_error_handler
def search_documents(query: str, top_k: int = 5) -> str:
    """Search across uploaded documents.

    Args:
        query: Search query
        top_k: Number of results to return

    Returns:
        Relevant document chunks
    """
    if not _document_store:
        return "No documents uploaded. Please upload documents first."

    # Simple keyword search (use vector similarity in production)
    query_words = set(query.lower().split())
    results = []

    for doc_id, doc in _document_store.items():
        for i, chunk in enumerate(doc["chunks"]):
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words & chunk_words)
            if overlap > 0:
                results.append({
                    "doc_id": doc_id,
                    "filename": doc["filename"],
                    "chunk_idx": i,
                    "content": chunk[:500],
                    "relevance": overlap,
                })

    # Sort by relevance
    results.sort(key=lambda x: x["relevance"], reverse=True)
    results = results[:top_k]

    if not results:
        return "No relevant content found for your query."

    output = f"Found {len(results)} relevant chunks:\n\n"
    for i, r in enumerate(results, 1):
        output += (
            f"**Result {i}** (from {r['filename']}):\n"
            f"{r['content']}...\n\n"
        )

    return output


@tool
@tool_error_handler
def summarize_document(doc_id: str, language: str = "auto") -> str:
    """Generate a summary of a specific document.

    Args:
        doc_id: Document ID to summarize
        language: Output language (auto uses document language)

    Returns:
        Document summary
    """
    if doc_id not in _document_store:
        return f"Document {doc_id} not found."

    doc = _document_store[doc_id]
    content = " ".join(doc["chunks"][:5])  # First 5 chunks for summary

    # In production, this would use the LLM for summarization
    summary = (
        f"Summary of: {doc['filename']}\n"
        f"Language: {doc['language']}\n"
        f"Content preview:\n{content[:1000]}...\n\n"
        f"Note: For full summarization, the agent will process this content."
    )

    return summary


@tool
@tool_error_handler
def list_documents() -> str:
    """List all uploaded documents.

    Returns:
        List of documents with metadata
    """
    if not _document_store:
        return "No documents uploaded yet."

    output = "Uploaded Documents:\n" + "=" * 40 + "\n\n"
    for doc_id, doc in _document_store.items():
        output += (
            f"**{doc_id}**: {doc['filename']}\n"
            f"  - Language: {doc['language']}\n"
            f"  - Chunks: {doc['chunk_count']}\n"
            f"  - Uploaded: {doc['uploaded_at']}\n\n"
        )

    return output


@tool
@tool_error_handler
def translate_response(text: str, target_language: str) -> str:
    """Translate text to target language (placeholder).

    Args:
        text: Text to translate
        target_language: Target language code

    Returns:
        Translation note (actual translation requires translation API)
    """
    return (
        f"Translation requested to: {target_language}\n"
        f"Original text: {text[:200]}...\n\n"
        f"Note: For actual translation, configure a translation API "
        f"(Google Translate, DeepL, etc.)"
    )


class MultilingualRAGAgent(BaseAgent):
    """Multilingual RAG Agent for document Q&A.

    Features:
    - Multi-language document support
    - Semantic search across documents
    - Cross-lingual question answering
    - Document summarization

    Example:
        >>> agent = MultilingualRAGAgent()
        >>> agent.upload("document content...", "report.pdf")
        >>> result = agent.query("What are the main findings?")
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize the RAG Agent."""
        super().__init__(config)

        self.register_tools([
            upload_document,
            search_documents,
            summarize_document,
            list_documents,
            translate_response,
        ])

    def _get_system_prompt(self) -> str:
        """Get the RAG agent's system prompt."""
        return """You are a Multilingual Document Analysis Agent specializing in
document retrieval and question answering across multiple languages.

## Your Capabilities:
1. **Upload**: Process documents with upload_document
2. **Search**: Find relevant content with search_documents
3. **Summarize**: Generate summaries with summarize_document
4. **List**: Show available documents with list_documents
5. **Translate**: Translate responses with translate_response

## Process:
1. First check if documents are loaded (use list_documents)
2. For questions, search for relevant content
3. Synthesize answers from retrieved chunks
4. Translate if user requests different language

## Guidelines:
- Always cite which document information came from
- If content is in a different language, offer translation
- Be clear about limitations (e.g., if search found nothing)
- For summarization, cover key points comprehensively
- Maintain original meaning when translating

## Response Format:
- Start with direct answer to the question
- Include relevant quotes/excerpts
- Cite sources (document names)
- Offer to elaborate or translate if needed"""

    def _build_graph(self) -> StateGraph:
        """Build the RAG agent's workflow graph."""

        def call_model(state: RAGState) -> dict:
            system_prompt = SystemMessage(content=self._get_system_prompt())
            messages = [system_prompt] + list(state.messages)
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def should_continue(state: RAGState) -> str:
            messages = list(state.messages)
            if not messages:
                return "end"
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "end"

        graph = StateGraph(RAGState)
        graph.add_node("agent", call_model)
        graph.add_node("tools", ToolNode(self._tools))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
        graph.add_edge("tools", "agent")

        return graph

    @traceable(name="rag_query")
    def query(
        self,
        question: str,
        language: str = "auto",
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Query the document collection.

        Args:
            question: Question to answer
            language: Response language
            session_id: Optional session ID

        Returns:
            Answer based on documents
        """
        return self.invoke(
            message=question,
            session_id=session_id,
            query=question,
            language=language,
        )

    def upload(
        self,
        content: str,
        filename: str,
        doc_type: str = "text",
    ) -> str:
        """Upload a document to the collection.

        Args:
            content: Document content
            filename: Filename
            doc_type: Document type

        Returns:
            Upload confirmation
        """
        return upload_document.invoke({
            "content": content,
            "filename": filename,
            "doc_type": doc_type,
        })
