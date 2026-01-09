"""Tools for the Document Intelligence Agent.

This module provides 8 tools for document processing, search,
translation, and web search operations.

Following Enterprise Development Standards:
- Software Engineer: Type-safe tools with error handling
- Security Architect: Safe file handling, no data leakage
"""

import base64
import logging
from typing import Any

from langchain_core.tools import tool

from app.agents.base.tools import tool_error_handler

logger = logging.getLogger(__name__)


# Module-level session tracking for tools
# Maps session_id to document data
_active_session: str | None = None


def set_active_session(session_id: str) -> None:
    """Set the active session for tool operations.

    Args:
        session_id: Session identifier
    """
    global _active_session
    _active_session = session_id


def get_active_session() -> str:
    """Get the active session ID.

    Returns:
        Active session ID or 'default'
    """
    return _active_session or "default"


@tool
@tool_error_handler
def upload_document(
    content_base64: str,
    filename: str,
) -> str:
    """Upload and process a document for analysis.

    Supports PDF, TXT, DOC/DOCX, PPT/PPTX, and images (PNG/JPG).
    Images are processed with OCR (pytesseract).

    Args:
        content_base64: File content encoded as base64 string
        filename: Original filename with extension (e.g., 'report.pdf')

    Returns:
        Upload confirmation with document ID and statistics
    """
    from app.agents.document_intelligence.document_processor import DocumentProcessor
    from app.agents.document_intelligence.vector_store import get_vector_store

    session_id = get_active_session()

    # Decode base64 content
    try:
        content = base64.b64decode(content_base64)
    except Exception as e:
        return f"Error: Invalid base64 content - {e}"

    # Process document
    processor = DocumentProcessor()

    if not processor.is_supported(filename):
        supported = ", ".join(processor.SUPPORTED_EXTENSIONS)
        return f"Error: Unsupported file type. Supported formats: {supported}"

    result = processor.process_file(content, filename)

    # Add to vector store
    vector_store = get_vector_store()
    doc_id = vector_store.add_document(
        session_id=session_id,
        chunks=result["chunks"],
        filename=result["filename"],
        file_type=result["file_type"],
        language=result["detected_language"],
    )

    return (
        f"Document uploaded successfully!\n\n"
        f"**Document ID**: {doc_id}\n"
        f"**Filename**: {result['filename']}\n"
        f"**Type**: {result['file_type']}\n"
        f"**Language**: {result['detected_language']}\n"
        f"**Chunks created**: {result['chunk_count']}\n"
        f"**Total characters**: {result['total_characters']}"
    )


@tool
@tool_error_handler
def search_documents(
    query: str,
    top_k: int = 5,
    document_ids: str | None = None,
    scope: str = "current",
) -> str:
    """Search across uploaded documents using semantic search.

    Finds the most relevant document chunks for your query.

    IMPORTANT: Use the scope parameter to control which documents to search:
    - 'current': Search only the most recently uploaded document (DEFAULT - use for "this document", "the image I uploaded", etc.)
    - 'recent': Search the 3 most recently uploaded documents
    - 'all': Search all uploaded documents (use for general queries across all documents)

    Args:
        query: Search query (natural language question or keywords)
        top_k: Number of results to return (default: 5)
        document_ids: Optional comma-separated list of specific document IDs to search (overrides scope)
        scope: Document scope - 'current' (most recent), 'recent' (last 3), or 'all' (default: 'current')

    Returns:
        Relevant document chunks with source information
    """
    from app.agents.document_intelligence.vector_store import get_vector_store

    session_id = get_active_session()
    vector_store = get_vector_store()

    # Parse document IDs if provided
    doc_ids = None
    if document_ids:
        doc_ids = [d.strip() for d in document_ids.split(",")]

    # Validate scope
    valid_scopes = ("current", "recent", "all")
    if scope not in valid_scopes:
        scope = "current"

    results = vector_store.search(
        session_id=session_id,
        query=query,
        k=top_k,
        document_ids=doc_ids,
        scope=scope,
    )

    if not results:
        # Provide helpful context about what was searched
        scope_desc = {
            "current": "the most recently uploaded document",
            "recent": "the 3 most recently uploaded documents",
            "all": "all uploaded documents",
        }
        return (
            f"No relevant content found in {scope_desc.get(scope, 'documents')}.\n\n"
            "Suggestions:\n"
            "1. Try scope='all' to search across all documents\n"
            "2. Verify documents have been uploaded (use list_documents)\n"
            "3. Rephrase your query\n"
        )

    # Show which scope was used
    scope_info = {
        "current": "current document",
        "recent": "recent documents",
        "all": "all documents",
    }

    output = f"Found {len(results)} relevant chunks (searched {scope_info.get(scope, scope)}):\n\n"
    for i, r in enumerate(results, 1):
        output += (
            f"**Result {i}** (from {r['filename']}, chunk {r['chunk_index']}):\n"
            f"{r['content'][:500]}...\n\n"
        )

    return output


@tool
@tool_error_handler
def web_search(
    query: str,
    max_results: int = 5,
) -> str:
    """Search the web within allowed domains.

    Searches are restricted to domains configured in ALLOWED_SEARCH_DOMAINS.

    Args:
        query: Search query
        max_results: Maximum number of results (default: 5)

    Returns:
        Search results from allowed domains
    """
    from app.agents.document_intelligence.web_search import get_domain_search

    searcher = get_domain_search()
    allowed = searcher.get_allowed_domains()

    if not allowed:
        return (
            "Web search is not configured. "
            "Set ALLOWED_SEARCH_DOMAINS in your .env file.\n"
            "Example: ALLOWED_SEARCH_DOMAINS=docs.python.org,stackoverflow.com"
        )

    result = searcher.search_sync(query, max_results)

    if result.get("error"):
        return f"Search error: {result['error']}"

    if not result.get("results"):
        return f"No results found for '{query}' in allowed domains: {', '.join(allowed)}"

    output = f"Web search results for '{query}':\n"
    output += f"(Searching in: {', '.join(allowed)})\n\n"

    for i, r in enumerate(result["results"], 1):
        output += (
            f"**{i}. {r['title']}**\n"
            f"URL: {r['url']}\n"
            f"{r['snippet']}\n\n"
        )

    return output


@tool
@tool_error_handler
def translate_text(
    text: str,
    target_language: str,
    source_language: str = "auto",
) -> str:
    """Translate text to a target language.

    Uses LLM-based translation for high-quality results.

    Args:
        text: Text to translate
        target_language: Target language code (e.g., 'es', 'fr', 'de', 'ja', 'zh')
        source_language: Source language code or 'auto' for detection

    Returns:
        Translated text with language information
    """
    from app.agents.document_intelligence.translator import get_translator

    translator = get_translator()
    result = translator.translate(text, target_language, source_language)

    if not result["success"]:
        return f"Translation error: {result.get('error', 'Unknown error')}"

    return (
        f"**Translation** ({result['source_language_name']} -> {result['target_language_name']}):\n\n"
        f"{result['translated']}"
    )


@tool
@tool_error_handler
def summarize_document(
    document_id: str,
    summary_type: str = "brief",
    target_language: str = "en",
) -> str:
    """Generate a summary of a specific document.

    Args:
        document_id: Document ID to summarize (e.g., 'doc_abc123')
        summary_type: Type of summary - 'brief' (2-3 sentences), 'detailed' (paragraph), or 'executive' (bullet points)
        target_language: Language for the summary (default: 'en')

    Returns:
        Document summary in the specified language
    """
    from app.agents.document_intelligence.vector_store import get_vector_store
    from app.agents.document_intelligence.translator import get_translator

    session_id = get_active_session()
    vector_store = get_vector_store()

    # Get document metadata
    doc = vector_store.get_document(session_id, document_id)
    if not doc:
        return f"Document '{document_id}' not found. Use 'list_documents' to see available documents."

    # Search for key content
    results = vector_store.search(
        session_id=session_id,
        query=f"main points key findings summary of {doc['filename']}",
        k=10,
        document_ids=[document_id],
    )

    if not results:
        return f"Could not extract content from document '{document_id}'."

    # Combine content for summarization
    content = "\n\n".join([r["content"] for r in results])

    # Get LLM for summarization
    translator = get_translator()
    llm = translator._get_llm()

    # Create summary prompt based on type
    if summary_type == "brief":
        instruction = "Provide a brief 2-3 sentence summary of the following content."
    elif summary_type == "executive":
        instruction = "Provide an executive summary with 5-7 bullet points highlighting the key findings."
    else:  # detailed
        instruction = "Provide a detailed paragraph summary covering the main points, findings, and conclusions."

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"{instruction}\nRespond in {target_language}."),
        ("human", "Content to summarize:\n\n{content}"),
    ])

    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"content": content})

    return (
        f"**Summary of {doc['filename']}** ({summary_type}):\n\n"
        f"{summary}"
    )


@tool
@tool_error_handler
def list_documents() -> str:
    """List all uploaded documents with their metadata.

    Returns:
        Formatted list of documents with IDs, names, types, languages, and chunk counts.
        Shows which document is currently active (most recently uploaded).
    """
    from app.agents.document_intelligence.vector_store import get_vector_store

    session_id = get_active_session()
    vector_store = get_vector_store()

    docs = vector_store.get_documents(session_id)

    if not docs:
        return (
            "No documents uploaded yet.\n\n"
            "Use the upload_document tool to add documents for analysis.\n"
            "Supported formats: PDF, TXT, DOCX, PPTX, PNG/JPG (with OCR)"
        )

    stats = vector_store.get_session_stats(session_id)
    current_doc_id = vector_store.get_current_document_id(session_id)
    recent_doc_ids = vector_store.get_recent_document_ids(session_id, n=3)

    output = "**Uploaded Documents:**\n"
    output += "=" * 40 + "\n\n"

    for doc in docs:
        # Mark current and recent documents
        status_markers = []
        if doc['doc_id'] == current_doc_id:
            status_markers.append("CURRENT")
        elif doc['doc_id'] in recent_doc_ids:
            status_markers.append("recent")

        status_str = f" [{', '.join(status_markers)}]" if status_markers else ""

        output += (
            f"**{doc['doc_id']}**: {doc['filename']}{status_str}\n"
            f"  - Type: {doc['file_type']}\n"
            f"  - Language: {doc['language']}\n"
            f"  - Chunks: {doc['chunk_count']}\n"
            f"  - Uploaded: {doc['uploaded_at']}\n\n"
        )

    output += f"\n**Session Statistics:**\n"
    output += f"- Total documents: {stats['total_documents']}\n"
    output += f"- Total chunks: {stats['total_chunks']}\n"
    output += f"- Current document: {current_doc_id or 'None'}\n"
    output += f"- Languages: {', '.join(stats['languages'])}\n"
    output += f"- File types: {', '.join(stats['file_types'])}"

    return output


@tool
@tool_error_handler
def clear_documents(
    document_ids: str | None = None,
) -> str:
    """Clear documents from the session.

    Args:
        document_ids: Comma-separated list of document IDs to clear, or leave empty to clear all

    Returns:
        Confirmation of cleared documents
    """
    from app.agents.document_intelligence.vector_store import get_vector_store

    session_id = get_active_session()
    vector_store = get_vector_store()

    # Parse document IDs if provided
    doc_ids = None
    if document_ids:
        doc_ids = [d.strip() for d in document_ids.split(",")]

    count = vector_store.clear_documents(session_id, doc_ids)

    if count == 0:
        return "No documents to clear."

    if doc_ids:
        return f"Cleared {count} document(s): {', '.join(doc_ids)}"
    else:
        return f"Cleared all {count} document(s) from the session."


@tool
@tool_error_handler
def detect_language(text: str) -> str:
    """Detect the language of provided text.

    Args:
        text: Text to analyze

    Returns:
        Detected language code and name with confidence information
    """
    from app.agents.document_intelligence.translator import get_translator, SUPPORTED_LANGUAGES

    translator = get_translator()
    code = translator.detect_language(text)
    name = SUPPORTED_LANGUAGES.get(code, code.capitalize())

    return (
        f"**Detected Language:**\n"
        f"- Code: {code}\n"
        f"- Name: {name}\n\n"
        f"Sample analyzed: \"{text[:100]}{'...' if len(text) > 100 else ''}\""
    )


# Export all tools for registration
ALL_TOOLS = [
    upload_document,
    search_documents,
    web_search,
    translate_text,
    summarize_document,
    list_documents,
    clear_documents,
    detect_language,
]
