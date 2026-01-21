"""Semantic long-term memory using FAISS vector storage.

Provides cross-session memory capabilities:
- Store conversation summaries with embeddings
- Retrieve relevant past context by semantic similarity
- User preference learning across sessions
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.agents.base.llm_factory import get_embedding_model


@dataclass
class MemoryEntry:
    """A single memory entry with metadata.

    Args:
        id: Unique identifier for the memory.
        content: The text content of the memory.
        user_id: User who created this memory.
        session_id: Session where memory was created.
        agent_type: Type of agent that created memory.
        memory_type: Type of memory (summary, preference, fact).
        created_at: Timestamp of creation.
        metadata: Additional metadata.
    """

    id: str
    content: str
    user_id: str | None = None
    session_id: str | None = None
    agent_type: str | None = None
    memory_type: str = "summary"  # summary, preference, fact, context
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_document(self) -> Document:
        """Convert to LangChain Document for vector storage."""
        return Document(
            page_content=self.content,
            metadata={
                "id": self.id,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "agent_type": self.agent_type,
                "memory_type": self.memory_type,
                "created_at": self.created_at,
                **self.metadata,
            },
        )

    @classmethod
    def from_document(cls, doc: Document) -> "MemoryEntry":
        """Create from LangChain Document."""
        metadata = doc.metadata.copy()
        return cls(
            id=metadata.pop("id", str(uuid4())),
            content=doc.page_content,
            user_id=metadata.pop("user_id", None),
            session_id=metadata.pop("session_id", None),
            agent_type=metadata.pop("agent_type", None),
            memory_type=metadata.pop("memory_type", "summary"),
            created_at=metadata.pop("created_at", datetime.now(timezone.utc).isoformat()),
            metadata=metadata,
        )


class SemanticMemory:
    """Semantic long-term memory with FAISS vector storage.

    Stores and retrieves memories based on semantic similarity,
    enabling agents to recall relevant past interactions.
    """

    def __init__(
        self,
        embeddings: Embeddings | None = None,
        persist_directory: str | None = None,
    ):
        """Initialize semantic memory.

        Args:
            embeddings: Embedding model for vectorization.
                       If None, uses OpenAI embeddings.
            persist_directory: Directory for FAISS index persistence.
                              If None, uses in-memory storage.
        """
        self.persist_directory = persist_directory or os.getenv(
            "SEMANTIC_MEMORY_PATH", "./data/semantic_memory"
        )
        self._embeddings = embeddings
        self._vector_store = None
        self._initialized = False

    @property
    def embeddings(self) -> Embeddings:
        """Get or create embeddings model."""
        if self._embeddings is None:
            self._embeddings = self._create_embeddings()
        return self._embeddings

    def _create_embeddings(self) -> Embeddings:
        """Create default embeddings model.

        Uses the centralized embedding factory which supports:
        - Azure OpenAI (primary for production)
        - OpenAI (disabled by default)
        - HuggingFace (local fallback)
        """
        return get_embedding_model()

    @property
    def vector_store(self):
        """Get or create FAISS vector store."""
        if self._vector_store is None:
            self._vector_store = self._load_or_create_store()
        return self._vector_store

    def _load_or_create_store(self):
        """Load existing FAISS index or create new one."""
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError:
            raise ImportError(
                "FAISS vector store requires 'faiss-cpu' package. "
                "Install with: pip install faiss-cpu"
            )

        import pathlib

        index_path = pathlib.Path(self.persist_directory)

        # Try to load existing index
        if index_path.exists() and (index_path / "index.faiss").exists():
            try:
                store = FAISS.load_local(
                    str(index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                print(f"Loaded semantic memory from {index_path}")
                return store
            except Exception as e:
                print(f"Warning: Could not load existing index: {e}")

        # Create new empty store with a placeholder document
        # FAISS requires at least one document to initialize
        placeholder = Document(
            page_content="System initialization placeholder",
            metadata={"type": "system", "created_at": datetime.utcnow().isoformat()},
        )
        store = FAISS.from_documents([placeholder], self.embeddings)

        # Ensure directory exists and save
        index_path.mkdir(parents=True, exist_ok=True)
        store.save_local(str(index_path))
        print(f"Created new semantic memory at {index_path}")

        return store

    def add_memory(
        self,
        content: str,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_type: str | None = None,
        memory_type: str = "summary",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a new memory entry.

        Args:
            content: The memory content to store.
            user_id: User identifier.
            session_id: Session identifier.
            agent_type: Type of agent creating memory.
            memory_type: Type of memory (summary, preference, fact).
            metadata: Additional metadata.

        Returns:
            The ID of the created memory.
        """
        memory_id = str(uuid4())
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            user_id=user_id,
            session_id=session_id,
            agent_type=agent_type,
            memory_type=memory_type,
            metadata=metadata or {},
        )

        doc = entry.to_document()
        self.vector_store.add_documents([doc])

        # Persist to disk
        self._save()

        return memory_id

    def search(
        self,
        query: str,
        k: int = 5,
        user_id: str | None = None,
        agent_type: str | None = None,
        memory_type: str | None = None,
        score_threshold: float = 0.5,
    ) -> list[MemoryEntry]:
        """Search for relevant memories.

        Args:
            query: Search query text.
            k: Maximum number of results.
            user_id: Filter by user ID.
            agent_type: Filter by agent type.
            memory_type: Filter by memory type.
            score_threshold: Minimum similarity score (0-1).

        Returns:
            List of matching memory entries.
        """
        # Build filter dict
        filter_dict = {}
        if user_id:
            filter_dict["user_id"] = user_id
        if agent_type:
            filter_dict["agent_type"] = agent_type
        if memory_type:
            filter_dict["memory_type"] = memory_type

        # Search with scores
        try:
            if filter_dict:
                results = self.vector_store.similarity_search_with_score(
                    query, k=k * 2, filter=filter_dict  # Get more to filter by score
                )
            else:
                results = self.vector_store.similarity_search_with_score(query, k=k * 2)
        except Exception as e:
            print(f"Warning: Memory search failed: {e}")
            return []

        # Filter by score and convert to MemoryEntry
        memories = []
        for doc, score in results:
            # Skip system placeholder
            if doc.metadata.get("type") == "system":
                continue

            # FAISS returns L2 distance, convert to similarity
            # Lower distance = higher similarity
            similarity = 1 / (1 + score)

            if similarity >= score_threshold:
                entry = MemoryEntry.from_document(doc)
                entry.metadata["similarity_score"] = similarity
                memories.append(entry)

            if len(memories) >= k:
                break

        return memories

    def get_user_context(
        self,
        user_id: str,
        query: str | None = None,
        k: int = 3,
    ) -> str:
        """Get relevant context for a user.

        Retrieves user's past interactions and preferences
        to inject into agent context.

        Args:
            user_id: User identifier.
            query: Optional query for semantic search.
            k: Maximum number of memories to include.

        Returns:
            Formatted context string for agent injection.
        """
        if query:
            memories = self.search(query, k=k, user_id=user_id)
        else:
            # Get most recent memories for user
            memories = self.search(
                "user preferences and past interactions",
                k=k,
                user_id=user_id,
            )

        if not memories:
            return ""

        context_parts = ["## Relevant Past Context"]
        for mem in memories:
            context_parts.append(f"- [{mem.memory_type}] {mem.content}")

        return "\n".join(context_parts)

    def get_agent_context(
        self,
        agent_type: str,
        query: str,
        k: int = 3,
    ) -> str:
        """Get relevant context for an agent type.

        Retrieves past interactions for similar queries
        to help the agent provide consistent responses.

        Args:
            agent_type: Type of agent.
            query: Current query for semantic search.
            k: Maximum number of memories to include.

        Returns:
            Formatted context string for agent injection.
        """
        memories = self.search(query, k=k, agent_type=agent_type)

        if not memories:
            return ""

        context_parts = ["## Similar Past Interactions"]
        for mem in memories:
            context_parts.append(f"- {mem.content}")

        return "\n".join(context_parts)

    def _save(self) -> None:
        """Persist vector store to disk."""
        import pathlib

        index_path = pathlib.Path(self.persist_directory)
        index_path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(index_path))

    def clear(self, user_id: str | None = None) -> int:
        """Clear memories.

        Args:
            user_id: If provided, only clear memories for this user.
                    If None, clears all memories.

        Returns:
            Number of memories cleared.
        """
        # For now, recreate the store (FAISS doesn't support deletion well)
        # In production, use a database-backed store
        if user_id is None:
            self._vector_store = None
            import shutil

            shutil.rmtree(self.persist_directory, ignore_errors=True)
            print("Cleared all semantic memories")
            return -1  # Unknown count

        # User-specific clearing requires rebuild
        print(f"Warning: User-specific clearing not fully supported with FAISS")
        return 0


# Global semantic memory instance
_semantic_memory: SemanticMemory | None = None


def get_semantic_memory() -> SemanticMemory:
    """Get or create the global semantic memory instance."""
    global _semantic_memory
    if _semantic_memory is None:
        enabled = os.getenv("SEMANTIC_MEMORY_ENABLED", "false").lower() == "true"
        if enabled:
            _semantic_memory = SemanticMemory()
        else:
            print("Semantic memory disabled. Set SEMANTIC_MEMORY_ENABLED=true to enable.")
            _semantic_memory = SemanticMemory()  # Create but won't persist
    return _semantic_memory


def reset_semantic_memory() -> None:
    """Reset the global semantic memory instance."""
    global _semantic_memory
    _semantic_memory = None
