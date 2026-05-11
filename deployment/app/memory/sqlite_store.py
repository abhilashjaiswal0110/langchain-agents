"""SQLite session store implementation.

Provides persistent local storage using SQLite.
Suitable for single-instance deployments requiring persistence.
"""

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Generator

# Maximum number of messages to keep per session (0 = unlimited).
# Set MAX_HISTORY_MESSAGES environment variable to override.
MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "0"))

from app.memory.base import (
    BaseSessionStore,
    Message,
    Session,
    SessionMetadata,
)

logger = logging.getLogger(__name__)


class SQLiteSessionStore(BaseSessionStore):
    """SQLite-based session storage.

    Provides persistent local storage for sessions.
    Suitable for single-instance deployments where Redis
    is not available but persistence is required.
    """

    def __init__(
        self,
        db_path: str = "sessions.db",
        default_ttl_hours: int = 168,  # 7 days
    ) -> None:
        """Initialize SQLite store.

        Args:
            db_path: Path to SQLite database file.
            default_ttl_hours: Default session TTL in hours.
        """
        self._db_path = Path(db_path)
        self._default_ttl_hours = default_ttl_hours

        # Ensure directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()
        logger.info(f"SQLite session store initialized at {self._db_path}")

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    agent_type TEXT NOT NULL,
                    metadata TEXT,
                    context TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    tenant_id TEXT NOT NULL DEFAULT 'default'
                )
            """)

            # Add tenant_id column to existing databases that predate this feature.
            # The ALTER TABLE is a no-op if the column already exists; the exception
            # is silently swallowed so existing deployments are unaffected.
            try:
                conn.execute(
                    "ALTER TABLE sessions ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'"
                )
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise

            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)

            # Create indices
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_user_id
                ON sessions(user_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_agent_type
                ON sessions(agent_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_expires_at
                ON sessions(expires_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_tenant_id
                ON sessions(tenant_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON messages(session_id)
            """)

            conn.commit()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection with foreign keys enabled."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        # Enable foreign keys for data integrity
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    def create_session(
        self,
        agent_type: str,
        user_id: str = "",
        metadata: dict | None = None,
        ttl_hours: int | None = None,
        tenant_id: str = "default",
    ) -> str:
        """Create a new session.

        Args:
            agent_type: Type of agent for this session.
            user_id: User identifier.
            metadata: Additional metadata.
            ttl_hours: Session TTL in hours (None for default).
            tenant_id: Tenant identifier for session isolation.

        Returns:
            Session ID.
        """
        session_metadata = SessionMetadata(
            user_id=user_id,
            agent_type=agent_type,
            custom=metadata or {},
            tenant_id=tenant_id,
        )

        ttl = ttl_hours or self._default_ttl_hours
        expires_at = datetime.now() + timedelta(hours=ttl)

        session = Session(
            metadata=session_metadata,
            expires_at=expires_at,
        )

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, user_id, agent_type, metadata, context, created_at, updated_at, expires_at, tenant_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    user_id,
                    agent_type,
                    json.dumps(session_metadata.to_dict()),
                    json.dumps(session.context),
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                    session.expires_at.isoformat() if session.expires_at else None,
                    tenant_id,
                ),
            )
            conn.commit()

        logger.debug(f"Created SQLite session {session.id} for agent {agent_type} (tenant={tenant_id})")
        return session.id

    def get_session(self, session_id: str, tenant_id: str = "default") -> Session | None:
        """Get a session by ID.

        Args:
            session_id: Session identifier.
            tenant_id: Tenant identifier for session isolation.

        Returns:
            Session or None if not found.
        """
        with self._get_connection() as conn:
            # Get session, scoped to tenant
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ? AND tenant_id = ?",
                (session_id, tenant_id),
            ).fetchone()

            if not row:
                return None

            # Check expiration
            expires_at = row["expires_at"]
            if expires_at:
                expires_at = datetime.fromisoformat(expires_at)
                if datetime.now() > expires_at:
                    self.delete_session(session_id, tenant_id=tenant_id)
                    return None

            # Get messages
            messages_rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()

            messages = []
            for msg_row in messages_rows:
                messages.append(Message(
                    role=msg_row["role"],
                    content=msg_row["content"],
                    timestamp=datetime.fromisoformat(msg_row["timestamp"]),
                    metadata=json.loads(msg_row["metadata"] or "{}"),
                ))

            # Build session
            metadata = SessionMetadata.from_dict(json.loads(row["metadata"] or "{}"))
            context = json.loads(row["context"] or "{}")

            return Session(
                id=row["id"],
                metadata=metadata,
                messages=messages,
                context=context,
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                expires_at=expires_at,
            )

    def update_session(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: dict | None = None,
        tenant_id: str = "default",
    ) -> bool:
        """Update session with new messages.

        Uses a transaction to ensure atomicity - all changes succeed or none do.

        Args:
            session_id: Session identifier.
            user_message: User's message.
            assistant_message: Assistant's response.
            metadata: Optional additional metadata.
            tenant_id: Tenant identifier for session isolation.

        Returns:
            True if updated successfully.
        """
        session = self.get_session(session_id, tenant_id=tenant_id)
        if not session:
            return False

        now = datetime.now()

        with self._get_connection() as conn:
            try:
                # Begin transaction explicitly
                conn.execute("BEGIN IMMEDIATE")

                # Add user message
                conn.execute(
                    """
                    INSERT INTO messages (session_id, role, content, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_id, "user", user_message, now.isoformat(), "{}"),
                )

                # Add assistant message
                conn.execute(
                    """
                    INSERT INTO messages (session_id, role, content, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        "assistant",
                        assistant_message,
                        now.isoformat(),
                        json.dumps(metadata or {}),
                    ),
                )

                # Update session timestamp
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?",
                    (now.isoformat(), session_id),
                )

                # Trim oldest messages when a history limit is configured.
                # Uses ORDER BY id DESC (AUTOINCREMENT) rather than timestamp
                # to guarantee deterministic ordering when multiple messages
                # share the same timestamp within a single update_session call.
                if MAX_HISTORY_MESSAGES > 0:
                    conn.execute(
                        """
                        DELETE FROM messages
                        WHERE session_id = ? AND id NOT IN (
                            SELECT id FROM messages
                            WHERE session_id = ?
                            ORDER BY id DESC
                            LIMIT ?
                        )
                        """,
                        (session_id, session_id, MAX_HISTORY_MESSAGES),
                    )

                conn.commit()

                return True

            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to update session {session_id}: {e}")
                return False

    def delete_session(self, session_id: str, tenant_id: str = "default") -> bool:
        """Delete a session.

        Args:
            session_id: Session identifier.
            tenant_id: Tenant identifier for session isolation.

        Returns:
            True if deleted successfully.
        """
        with self._get_connection() as conn:
            # Delete messages first (cascade)
            conn.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,),
            )

            # Delete session scoped to tenant
            cursor = conn.execute(
                "DELETE FROM sessions WHERE id = ? AND tenant_id = ?",
                (session_id, tenant_id),
            )
            conn.commit()

            deleted = cursor.rowcount > 0
            if deleted:
                logger.debug(f"Deleted SQLite session {session_id} (tenant={tenant_id})")
            return deleted

    def list_sessions(
        self,
        user_id: str | None = None,
        agent_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
        tenant_id: str | None = None,
    ) -> list[Session]:
        """List sessions with optional filters.

        Args:
            user_id: Filter by user.
            agent_type: Filter by agent type.
            limit: Maximum number of sessions.
            offset: Offset for pagination.
            tenant_id: Filter by tenant. When provided, only sessions for that
                tenant are returned.

        Returns:
            List of sessions.
        """
        conditions = ["(expires_at IS NULL OR expires_at > ?)"]
        params: list[Any] = [datetime.now().isoformat()]

        # Always scope the SQL query by tenant so get_session is called with the
        # correct tenant and non-default-tenant sessions are not silently dropped.
        effective_tenant = tenant_id or "default"
        conditions.append("tenant_id = ?")
        params.append(effective_tenant)

        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)

        if agent_type:
            conditions.append("agent_type = ?")
            params.append(agent_type)

        where_clause = " AND ".join(conditions)

        with self._get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT id FROM sessions
                WHERE {where_clause}
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ).fetchall()

        sessions = []
        for row in rows:
            session = self.get_session(row["id"], tenant_id=effective_tenant)
            if session:
                sessions.append(session)

        return sessions

    def get_history(
        self,
        session_id: str,
        limit: int | None = None,
        tenant_id: str = "default",
    ) -> list[Message]:
        """Get conversation history for a session.

        Args:
            session_id: Session identifier.
            limit: Maximum number of messages.
            tenant_id: Tenant identifier for session isolation.

        Returns:
            List of messages.
        """
        with self._get_connection() as conn:
            if limit:
                rows = conn.execute(
                    """
                    SELECT * FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                ).fetchall()
                rows = list(reversed(rows))
            else:
                rows = conn.execute(
                    "SELECT * FROM messages WHERE session_id = ? ORDER BY id",
                    (session_id,),
                ).fetchall()

            messages = []
            for row in rows:
                messages.append(Message(
                    role=row["role"],
                    content=row["content"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    metadata=json.loads(row["metadata"] or "{}"),
                ))

            return messages

    def clear_session(self, session_id: str, tenant_id: str = "default") -> bool:
        """Clear messages from a session.

        Args:
            session_id: Session identifier.
            tenant_id: Tenant identifier for session isolation.

        Returns:
            True if cleared successfully.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,),
            )

            # Update session timestamp, scoped to tenant
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ? AND tenant_id = ?",
                (datetime.now().isoformat(), session_id, tenant_id),
            )

            conn.commit()
            return cursor.rowcount >= 0

    def set_context(
        self,
        session_id: str,
        context: dict[str, Any],
        tenant_id: str = "default",
    ) -> bool:
        """Set session context.

        Uses a transaction to ensure atomicity.

        Args:
            session_id: Session identifier.
            context: Context data to set.
            tenant_id: Tenant identifier for session isolation.

        Returns:
            True if set successfully.
        """
        # Get existing context
        session = self.get_session(session_id, tenant_id=tenant_id)
        if not session:
            return False

        session.context.update(context)

        with self._get_connection() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                cursor = conn.execute(
                    """
                    UPDATE sessions
                    SET context = ?, updated_at = ?
                    WHERE id = ? AND tenant_id = ?
                    """,
                    (
                        json.dumps(session.context),
                        datetime.now().isoformat(),
                        session_id,
                        tenant_id,
                    ),
                )
                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to set context for session {session_id}: {e}")
                return False

    def get_context(self, session_id: str, tenant_id: str = "default") -> dict[str, Any]:
        """Get session context.

        Args:
            session_id: Session identifier.
            tenant_id: Tenant identifier for session isolation.

        Returns:
            Context data.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT context FROM sessions WHERE id = ? AND tenant_id = ?",
                (session_id, tenant_id),
            ).fetchone()

            if not row:
                return {}

            return json.loads(row["context"] or "{}")

    def cleanup_expired(self) -> int:
        """Clean up expired sessions.

        Returns:
            Number of sessions removed.
        """
        with self._get_connection() as conn:
            # Get expired session IDs
            rows = conn.execute(
                """
                SELECT id FROM sessions
                WHERE expires_at IS NOT NULL AND expires_at < ?
                """,
                (datetime.now().isoformat(),),
            ).fetchall()

            expired_ids = [row["id"] for row in rows]

            if expired_ids:
                # Delete messages
                conn.execute(
                    f"""
                    DELETE FROM messages
                    WHERE session_id IN ({','.join('?' * len(expired_ids))})
                    """,
                    expired_ids,
                )

                # Delete sessions
                conn.execute(
                    f"""
                    DELETE FROM sessions
                    WHERE id IN ({','.join('?' * len(expired_ids))})
                    """,
                    expired_ids,
                )

                conn.commit()
                logger.info(f"Cleaned up {len(expired_ids)} expired sessions")

            return len(expired_ids)

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics.

        Returns:
            Statistics dictionary.
        """
        with self._get_connection() as conn:
            session_count = conn.execute(
                "SELECT COUNT(*) FROM sessions"
            ).fetchone()[0]

            message_count = conn.execute(
                "SELECT COUNT(*) FROM messages"
            ).fetchone()[0]

            active_count = conn.execute(
                """
                SELECT COUNT(*) FROM sessions
                WHERE expires_at IS NULL OR expires_at > ?
                """,
                (datetime.now().isoformat(),),
            ).fetchone()[0]

            by_agent = {}
            rows = conn.execute(
                """
                SELECT agent_type, COUNT(*) as count
                FROM sessions
                WHERE expires_at IS NULL OR expires_at > ?
                GROUP BY agent_type
                """,
                (datetime.now().isoformat(),),
            ).fetchall()
            for row in rows:
                by_agent[row["agent_type"]] = row["count"]

            return {
                "total_sessions": session_count,
                "active_sessions": active_count,
                "total_messages": message_count,
                "by_agent": by_agent,
                "db_path": str(self._db_path),
            }

    def vacuum(self) -> None:
        """Optimize database by vacuuming."""
        with self._get_connection() as conn:
            conn.execute("VACUUM")
            logger.info("SQLite database vacuumed")

    def close(self) -> None:
        """Close the store (no-op for SQLite, connections are per-operation)."""
        pass
