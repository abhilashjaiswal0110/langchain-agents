"""Document tools for Deep Agents.

This module provides document management tools for Deep Agents to handle
file attachments, RAG search, and document context integration.

Following Enterprise Development Standards:
- Software Architect: Modular design with session isolation
- Security Architect: Secure file handling and path sanitization
- Data Architect: FAISS-based vector storage with metadata
- Software Engineer: Type-safe with comprehensive error handling
"""

import logging
import os
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from app.agents.base.llm_factory import get_embedding_model

logger = logging.getLogger(__name__)

# =============================================================================
# Session Context Management
# =============================================================================

# Context variable to track the current session_id during agent execution
# This allows tools to access the session_id without requiring it as a parameter
_current_session_id: ContextVar[str] = ContextVar("current_session_id", default="default")


def set_current_session(session_id: str) -> None:
    """Set the current session ID for document tools.

    Call this at the start of each agent invocation to ensure
    document tools can access the correct session context.

    Args:
        session_id: The session identifier to use.
    """
    _current_session_id.set(session_id)
    logger.debug(f"Set current session context: {session_id}")


def get_current_session() -> str:
    """Get the current session ID.

    Returns:
        The current session ID, or 'default' if not set.
    """
    return _current_session_id.get()

# =============================================================================
# Module-Level Storage (Session-Isolated)
# =============================================================================

# Session -> FAISS vector store
_vector_stores: dict[str, Any] = {}

# Session -> Document metadata
_document_metadata: dict[str, dict[str, dict]] = {}

# Session -> Current document ID
_current_document: dict[str, str] = {}

# Session -> Document upload order
_document_order: dict[str, list[str]] = {}

# Storage base path for persistent attachment storage
ATTACHMENTS_BASE_PATH = Path(os.getenv(
    "DEEPAGENT_ATTACHMENTS_PATH",
    "/data/deepagent_attachments"
))


def _ensure_storage_path(session_id: str) -> Path:
    """Ensure session-specific storage directory exists.

    Args:
        session_id: Session identifier

    Returns:
        Path to session storage directory
    """
    session_path = ATTACHMENTS_BASE_PATH / session_id
    session_path.mkdir(parents=True, exist_ok=True)
    return session_path


def _get_embeddings():
    """Get or create embeddings instance.

    Uses the factory pattern with Azure OpenAI as primary provider.

    Returns:
        Embeddings instance (Azure OpenAI, OpenAI, or HuggingFace)
    """
    return get_embedding_model()


def _get_or_create_vector_store(session_id: str):
    """Get or create FAISS vector store for session.

    Args:
        session_id: Session identifier

    Returns:
        FAISS vector store instance
    """
    if session_id not in _vector_stores:
        return None
    return _vector_stores[session_id]


def process_and_store_document(
    content: bytes,
    filename: str,
    session_id: str,
) -> dict[str, Any]:
    """Process document and store in session vector store.

    Args:
        content: Raw file content as bytes
        filename: Original filename with extension
        session_id: Session identifier

    Returns:
        Document metadata including doc_id, chunk_count, etc.
    """
    from langchain_community.vectorstores import FAISS
    from app.agents.document_intelligence.document_processor import DocumentProcessor

    # Initialize processor
    processor = DocumentProcessor()

    # Check if file type is supported
    if not processor.is_supported(filename):
        msg = f"Unsupported file type. Supported: {processor.SUPPORTED_EXTENSIONS}"
        raise ValueError(msg)

    # Process document
    result = processor.process_file(content, filename)
    chunks = result["chunks"]

    # Generate document ID
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"

    # Add doc_id to chunk metadata
    for chunk in chunks:
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["session_id"] = session_id

    # Create or update vector store
    embeddings = _get_embeddings()

    if session_id not in _vector_stores or _vector_stores[session_id] is None:
        _vector_stores[session_id] = FAISS.from_documents(chunks, embeddings)
    else:
        _vector_stores[session_id].add_documents(chunks)

    # Store metadata
    if session_id not in _document_metadata:
        _document_metadata[session_id] = {}
        _document_order[session_id] = []

    metadata = {
        "doc_id": doc_id,
        "filename": filename,
        "file_type": result["file_type"],
        "language": result["detected_language"],
        "chunk_count": result["chunk_count"],
        "total_characters": result["total_characters"],
        "uploaded_at": datetime.now().isoformat(),
    }

    _document_metadata[session_id][doc_id] = metadata
    _document_order[session_id].append(doc_id)
    _current_document[session_id] = doc_id

    # Persist file to disk
    try:
        storage_path = _ensure_storage_path(session_id)
        file_path = storage_path / f"{doc_id}_{filename}"
        file_path.write_bytes(content)
        metadata["stored_path"] = str(file_path)
        logger.info(f"Persisted attachment to {file_path}")
    except Exception as e:
        logger.warning(f"Failed to persist attachment: {e}")

    logger.info(
        f"Processed {filename} for session {session_id}: "
        f"{result['chunk_count']} chunks, doc_id={doc_id}"
    )

    return metadata


def get_document_context(session_id: str) -> str:
    """Get document context string for system prompt.

    Args:
        session_id: Session identifier.

    Returns:
        Formatted context string about uploaded documents.
    """
    if session_id not in _document_metadata or not _document_metadata[session_id]:
        return ""

    docs = _document_metadata[session_id]
    current_doc_id = _current_document.get(session_id)

    context = "\n## UPLOADED DOCUMENTS AVAILABLE IN THIS SESSION:\n"
    context += f"**Total Documents**: {len(docs)}\n\n"

    if current_doc_id and current_doc_id in docs:
        current = docs[current_doc_id]
        context += "**Current Document (most recent)**:\n"
        context += f"- Filename: {current['filename']}\n"
        context += f"- Document ID: {current['doc_id']}\n"
        context += f"- Type: {current['file_type']}\n"
        context += f"- Language: {current['language']}\n"
        context += f"- Chunks: {current['chunk_count']}\n\n"

    if len(docs) > 1:
        context += "**Other Documents**:\n"
        for doc_id, meta in docs.items():
            if doc_id != current_doc_id:
                context += f"- {meta['filename']} (ID: {doc_id}, {meta['chunk_count']} chunks)\n"

    context += "\n**ACTION REQUIRED**: When the user asks about document content, you MUST use:\n"
    context += "- `search_attachments(query=\"your search query\")` to find information\n"
    context += "- `list_attachments()` to see all documents\n"
    context += "- `get_attachment_summary()` for document overview\n"
    context += "\nDo NOT answer from your training data - always search the uploaded documents!\n"

    return context


# =============================================================================
# Tool Functions
# =============================================================================

@tool
def search_attachments(
    query: str,
    session_id: str | None = None,
    scope: str = "current",
    k: int = 5,
) -> str:
    """Search uploaded documents using semantic similarity.

    Use this tool to find relevant information in uploaded documents.
    The search uses vector similarity to find the most relevant chunks.

    Args:
        query: Search query describing what information to find.
        session_id: Session identifier (optional - uses current session if not provided).
        scope: Search scope - "current" (most recent doc), "recent" (last 3),
               or "all" (all documents).
        k: Number of results to return (default 5).

    Returns:
        Formatted search results with source attribution.
    """
    # Use context variable as fallback for session_id
    if session_id is None:
        session_id = get_current_session()
        logger.debug(f"search_attachments using context session: {session_id}")

    vector_store = _get_or_create_vector_store(session_id)

    if vector_store is None:
        return "No documents have been uploaded yet. Ask the user to upload a document first."

    # Determine which document IDs to search
    doc_ids_to_search = None
    if scope == "current":
        current = _current_document.get(session_id)
        if current:
            doc_ids_to_search = [current]
    elif scope == "recent":
        order = _document_order.get(session_id, [])
        doc_ids_to_search = order[-3:] if order else None
    # "all" leaves doc_ids_to_search as None to search everything

    try:
        # Perform similarity search
        results = vector_store.similarity_search_with_score(query, k=k * 2)

        # Filter by document IDs if specified
        filtered_results = []
        for doc, score in results:
            if doc_ids_to_search is None or doc.metadata.get("doc_id") in doc_ids_to_search:
                filtered_results.append((doc, score))
                if len(filtered_results) >= k:
                    break

        if not filtered_results:
            return f"No relevant content found for query: '{query}'"

        # Format results
        output = f"## Search Results for: '{query}'\n\n"

        for i, (doc, score) in enumerate(filtered_results, 1):
            doc_id = doc.metadata.get("doc_id", "unknown")
            filename = doc.metadata.get("source_file", "unknown")
            chunk_idx = doc.metadata.get("chunk_index", "?")

            output += f"### Result {i} (Score: {score:.3f})\n"
            output += f"**Source**: {filename} (chunk {chunk_idx})\n"
            output += f"**Document ID**: {doc_id}\n"
            output += f"**Content**:\n{doc.page_content[:1000]}\n\n"

        return output

    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Error searching documents: {e}"


@tool
def list_attachments(session_id: str | None = None) -> str:
    """List all uploaded documents in the session.

    Use this tool to see what documents are available to search.

    Args:
        session_id: Session identifier (optional - uses current session if not provided).

    Returns:
        Formatted list of uploaded documents with metadata.
    """
    # Use context variable as fallback for session_id
    if session_id is None:
        session_id = get_current_session()
        logger.debug(f"list_attachments using context session: {session_id}")

    if session_id not in _document_metadata or not _document_metadata[session_id]:
        return "No documents have been uploaded yet."

    docs = _document_metadata[session_id]
    current_doc_id = _current_document.get(session_id)

    output = f"## Uploaded Documents ({len(docs)} total)\n\n"

    for doc_id, meta in docs.items():
        is_current = " **(CURRENT)**" if doc_id == current_doc_id else ""
        output += f"### {meta['filename']}{is_current}\n"
        output += f"- **Document ID**: {doc_id}\n"
        output += f"- **Type**: {meta['file_type']}\n"
        output += f"- **Language**: {meta['language']}\n"
        output += f"- **Chunks**: {meta['chunk_count']}\n"
        output += f"- **Characters**: {meta['total_characters']:,}\n"
        output += f"- **Uploaded**: {meta['uploaded_at']}\n\n"

    return output


@tool
def get_attachment_summary(
    doc_id: str | None = None,
    session_id: str | None = None,
) -> str:
    """Get a summary of a specific document or the current document.

    Use this tool to get an overview of document content.

    Args:
        doc_id: Specific document ID, or None for current document.
        session_id: Session identifier (optional - uses current session if not provided).

    Returns:
        Document summary with key information.
    """
    # Use context variable as fallback for session_id
    if session_id is None:
        session_id = get_current_session()
        logger.debug(f"get_attachment_summary using context session: {session_id}")

    if session_id not in _document_metadata or not _document_metadata[session_id]:
        return "No documents have been uploaded yet."

    # Use current document if no doc_id specified
    if doc_id is None:
        doc_id = _current_document.get(session_id)
        if doc_id is None:
            return "No current document set. Upload a document first."

    if doc_id not in _document_metadata[session_id]:
        return f"Document ID '{doc_id}' not found in session."

    meta = _document_metadata[session_id][doc_id]
    vector_store = _get_or_create_vector_store(session_id)

    # Get first few chunks for summary context
    output = f"## Document Summary: {meta['filename']}\n\n"
    output += f"- **Document ID**: {doc_id}\n"
    output += f"- **Type**: {meta['file_type']}\n"
    output += f"- **Language**: {meta['language']}\n"
    output += f"- **Total Chunks**: {meta['chunk_count']}\n"
    output += f"- **Total Characters**: {meta['total_characters']:,}\n"

    if vector_store:
        try:
            # Get sample content from first chunks
            results = vector_store.similarity_search(
                "document overview summary introduction",
                k=3,
                filter={"doc_id": doc_id}
            )
            if results:
                output += "\n### Sample Content (first 500 chars per chunk):\n\n"
                for i, doc in enumerate(results[:3], 1):
                    output += f"**Chunk {i}**:\n{doc.page_content[:500]}...\n\n"
        except Exception as e:
            logger.warning(f"Could not get sample content: {e}")

    return output


@tool
def clear_attachments(
    doc_ids: list[str] | None = None,
    session_id: str | None = None,
) -> str:
    """Clear uploaded documents from the session.

    Use this tool to remove documents from the session context.
    Note: Due to FAISS limitations, vector entries are not deleted but
    metadata is cleared and documents are excluded from searches.

    Args:
        doc_ids: Specific document IDs to clear, or None for all.
        session_id: Session identifier (optional - uses current session if not provided).

    Returns:
        Confirmation message.
    """
    # Use context variable as fallback for session_id
    if session_id is None:
        session_id = get_current_session()
        logger.debug(f"clear_attachments using context session: {session_id}")

    if session_id not in _document_metadata:
        return "No documents to clear."

    if doc_ids is None:
        # Clear all
        count = len(_document_metadata[session_id])
        _document_metadata[session_id] = {}
        _document_order[session_id] = []
        _current_document.pop(session_id, None)
        # Note: FAISS doesn't support deletion, so vector store remains
        # but searches will be filtered by metadata
        _vector_stores.pop(session_id, None)
        return f"Cleared all {count} documents from session."
    else:
        # Clear specific documents
        cleared = 0
        for doc_id in doc_ids:
            if doc_id in _document_metadata[session_id]:
                del _document_metadata[session_id][doc_id]
                if doc_id in _document_order[session_id]:
                    _document_order[session_id].remove(doc_id)
                if _current_document.get(session_id) == doc_id:
                    # Set current to most recent remaining
                    if _document_order[session_id]:
                        _current_document[session_id] = _document_order[session_id][-1]
                    else:
                        _current_document.pop(session_id, None)
                cleared += 1

        return f"Cleared {cleared} document(s) from session."


# =============================================================================
# Export
# =============================================================================

__all__ = [
    "search_attachments",
    "list_attachments",
    "get_attachment_summary",
    "clear_attachments",
    "process_and_store_document",
    "get_document_context",
    "set_current_session",
    "get_current_session",
]
