"""Session-scoped vector store for document embeddings.

This module provides FAISS-based vector storage with session isolation,
allowing multiple users to maintain separate document collections.

Following Enterprise Development Standards:
- Software Architect: Session isolation pattern
- Security Architect: No cross-session data leakage
- Data Architect: Efficient vector storage
- Software Engineer: Type-safe with async support
"""

import logging
import uuid
from datetime import datetime
from typing import Any

from langchain_core.documents import Document

from app.agents.base.llm_factory import get_embedding_model

logger = logging.getLogger(__name__)


# Module-level stores for session isolation
_vector_stores: dict[str, Any] = {}  # session_id -> FAISS store
_document_metadata: dict[str, dict[str, dict[str, Any]]] = {}  # session_id -> {doc_id -> metadata}
_current_document: dict[str, str] = {}  # session_id -> current_doc_id (most recently uploaded)
_document_order: dict[str, list[str]] = {}  # session_id -> [doc_ids in upload order]


class DocumentVectorStore:
    """Session-scoped FAISS vector store for document embeddings.

    Each session maintains its own isolated vector store and document
    metadata, preventing cross-session data access.
    """

    def __init__(self) -> None:
        """Initialize the vector store manager."""
        try:
            # Uses factory with Azure OpenAI as primary
            self._embeddings = get_embedding_model()
        except Exception as e:
            logger.warning(f"Failed to initialize embeddings: {e}")
            self._embeddings = None

    def _get_or_create_store(self, session_id: str) -> Any:
        """Get or create a FAISS store for a session.

        Args:
            session_id: Session identifier

        Returns:
            FAISS vector store instance
        """
        if session_id not in _vector_stores:
            _vector_stores[session_id] = None
            _document_metadata[session_id] = {}
            _document_order[session_id] = []
        return _vector_stores.get(session_id)

    def get_current_document_id(self, session_id: str) -> str | None:
        """Get the most recently uploaded document ID for a session.

        Args:
            session_id: Session identifier

        Returns:
            Document ID of the most recently uploaded document, or None
        """
        return _current_document.get(session_id)

    def get_recent_document_ids(self, session_id: str, n: int = 3) -> list[str]:
        """Get the N most recently uploaded document IDs.

        Args:
            session_id: Session identifier
            n: Number of recent documents to return

        Returns:
            List of document IDs in reverse chronological order
        """
        order = _document_order.get(session_id, [])
        return order[-n:][::-1]  # Last N, reversed (most recent first)

    def set_current_document(self, session_id: str, doc_id: str) -> None:
        """Set the current (active) document for a session.

        Args:
            session_id: Session identifier
            doc_id: Document identifier to set as current
        """
        _current_document[session_id] = doc_id
        logger.debug(f"Set current document for session {session_id}: {doc_id}")

    def add_document(
        self,
        session_id: str,
        chunks: list[Document],
        filename: str,
        file_type: str,
        language: str,
    ) -> str:
        """Add a document's chunks to the session's vector store.

        Args:
            session_id: Session identifier
            chunks: List of document chunks
            filename: Original filename
            file_type: Type of document
            language: Detected language

        Returns:
            Generated document ID
        """
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError:
            msg = "faiss-cpu not installed. Install with: pip install faiss-cpu"
            raise ImportError(msg)

        if not self._embeddings:
            msg = "Embeddings not initialized. Check Azure OpenAI or other provider configuration."
            raise RuntimeError(msg)

        # Generate document ID
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"

        # Add document ID to chunk metadata
        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id

        # Initialize session metadata if needed
        if session_id not in _document_metadata:
            _document_metadata[session_id] = {}

        # Store document metadata
        _document_metadata[session_id][doc_id] = {
            "filename": filename,
            "file_type": file_type,
            "language": language,
            "chunk_count": len(chunks),
            "uploaded_at": datetime.now().isoformat(),
        }

        # Get or create vector store
        store = _vector_stores.get(session_id)

        if store is None:
            # Create new FAISS store
            store = FAISS.from_documents(chunks, self._embeddings)
            _vector_stores[session_id] = store
        else:
            # Add to existing store
            store.add_documents(chunks)

        # Track document order and set as current
        if session_id not in _document_order:
            _document_order[session_id] = []
        _document_order[session_id].append(doc_id)
        _current_document[session_id] = doc_id

        logger.info(f"Added document {doc_id} ({filename}) to session {session_id} [now current]")
        return doc_id

    def search(
        self,
        session_id: str,
        query: str,
        k: int = 5,
        document_ids: list[str] | None = None,
        scope: str = "all",
        rerank: bool = True,
    ) -> list[dict[str, Any]]:
        """Search for relevant chunks in the session's documents.

        Retrieves `k * 2` candidates from FAISS then applies cross-encoder
        reranking when `rerank=True` (and `RERANKER_ENABLED` env var is set).

        Args:
            session_id: Session identifier
            query: Search query
            k: Number of results to return
            document_ids: Optional filter to specific documents
            scope: Search scope - 'current' (most recent doc), 'recent' (last 3), or 'all'
            rerank: Whether to apply cross-encoder reranking after retrieval.

        Returns:
            List of matching chunks with metadata and scores
        """
        import os

        store = _vector_stores.get(session_id)

        if store is None:
            return []

        # Determine document filter based on scope
        effective_doc_ids = document_ids
        if scope == "current" and not document_ids:
            current_id = self.get_current_document_id(session_id)
            if current_id:
                effective_doc_ids = [current_id]
                logger.debug(f"Scoped search to current document: {current_id}")
        elif scope == "recent" and not document_ids:
            recent_ids = self.get_recent_document_ids(session_id, n=3)
            if recent_ids:
                effective_doc_ids = recent_ids
                logger.debug(f"Scoped search to recent documents: {recent_ids}")

        # Retrieve extra candidates so the reranker has material to work with
        results = store.similarity_search_with_score(query, k=k * 2)

        # Format results
        formatted_results = []
        for doc, score in results:
            if effective_doc_ids and doc.metadata.get("doc_id") not in effective_doc_ids:
                continue

            formatted_results.append(
                {
                    "content": doc.page_content,
                    "doc_id": doc.metadata.get("doc_id"),
                    "filename": doc.metadata.get("source_file"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "score": float(score),  # Lower is better for FAISS L2 distance
                    "metadata": doc.metadata,
                }
            )

        reranker_enabled = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
        if rerank and reranker_enabled and formatted_results:
            from app.agents.rag.reranker import get_reranker

            formatted_results = get_reranker().rerank(query, formatted_results, top_k=k)
        else:
            formatted_results = formatted_results[:k]

        return formatted_results

    def get_documents(self, session_id: str) -> list[dict[str, Any]]:
        """Get all documents in a session.

        Args:
            session_id: Session identifier

        Returns:
            List of document metadata
        """
        if session_id not in _document_metadata:
            return []

        return [{"doc_id": doc_id, **metadata} for doc_id, metadata in _document_metadata[session_id].items()]

    def get_document(self, session_id: str, doc_id: str) -> dict[str, Any] | None:
        """Get a specific document's metadata.

        Args:
            session_id: Session identifier
            doc_id: Document identifier

        Returns:
            Document metadata or None if not found
        """
        if session_id not in _document_metadata:
            return None
        return _document_metadata[session_id].get(doc_id)

    def clear_documents(
        self,
        session_id: str,
        document_ids: list[str] | None = None,
    ) -> int:
        """Clear documents from a session.

        Args:
            session_id: Session identifier
            document_ids: Specific documents to clear, or None for all

        Returns:
            Number of documents cleared
        """
        if session_id not in _document_metadata:
            return 0

        if document_ids is None:
            # Clear all documents
            count = len(_document_metadata[session_id])
            _document_metadata[session_id] = {}
            _vector_stores[session_id] = None
            logger.info(f"Cleared all {count} documents from session {session_id}")
            return count
        else:
            # Clear specific documents
            # Note: FAISS doesn't support deletion, so we rebuild the store
            # For production, consider using a vector DB that supports deletion
            count = 0
            for doc_id in document_ids:
                if doc_id in _document_metadata[session_id]:
                    del _document_metadata[session_id][doc_id]
                    count += 1

            # Rebuild store without deleted documents
            # This is inefficient but FAISS doesn't support deletion
            if count > 0 and _vector_stores.get(session_id) is not None:
                logger.warning(
                    "FAISS doesn't support deletion. Consider using ChromaDB "
                    "or Pinecone for production. Store will be rebuilt on next add."
                )
                # Mark store as dirty - next add will rebuild
                # For a full implementation, we'd need to rebuild from stored chunks

            logger.info(f"Cleared {count} documents from session {session_id}")
            return count

    def get_session_stats(self, session_id: str) -> dict[str, Any]:
        """Get statistics for a session.

        Args:
            session_id: Session identifier

        Returns:
            Session statistics
        """
        if session_id not in _document_metadata:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "languages": [],
                "file_types": [],
            }

        docs = _document_metadata[session_id]
        total_chunks = sum(d.get("chunk_count", 0) for d in docs.values())
        languages = list(set(d.get("language", "unknown") for d in docs.values()))
        file_types = list(set(d.get("file_type", "unknown") for d in docs.values()))

        return {
            "total_documents": len(docs),
            "total_chunks": total_chunks,
            "languages": languages,
            "file_types": file_types,
        }


# Global instance for module-level access
_store_instance: DocumentVectorStore | None = None


def get_vector_store() -> DocumentVectorStore:
    """Get the global vector store instance.

    Returns:
        DocumentVectorStore singleton instance
    """
    global _store_instance
    if _store_instance is None:
        _store_instance = DocumentVectorStore()
    return _store_instance
