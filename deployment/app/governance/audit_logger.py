"""Audit logging for enterprise IT agents.

Provides:
- Structured audit logging in JSON Lines format
- Compliance-ready log entries
- Async and sync logging support
- Log rotation and export
"""

import asyncio
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4


class AuditAction(str, Enum):
    """Types of auditable actions."""

    # Agent actions
    AGENT_INVOKE = "agent:invoke"
    AGENT_RESPONSE = "agent:response"
    AGENT_ERROR = "agent:error"

    # Approval actions
    APPROVAL_REQUEST = "approval:request"
    APPROVAL_GRANTED = "approval:granted"
    APPROVAL_DENIED = "approval:denied"
    APPROVAL_TIMEOUT = "approval:timeout"

    # Access actions
    AUTH_SUCCESS = "auth:success"
    AUTH_FAILURE = "auth:failure"
    PERMISSION_DENIED = "permission:denied"

    # System actions
    SYSTEM_START = "system:start"
    SYSTEM_STOP = "system:stop"
    CONFIG_CHANGE = "config:change"

    # Data actions
    DATA_ACCESS = "data:access"
    DATA_EXPORT = "data:export"


class AuditLevel(str, Enum):
    """Severity levels for audit entries."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEntry:
    """A single audit log entry.

    Attributes:
        timestamp: ISO 8601 timestamp
        action: Type of action being logged
        user_id: User performing the action
        level: Severity level
        agent_type: Type of agent involved
        input_hash: SHA-256 hash of input (for privacy)
        output_hash: SHA-256 hash of output (for privacy)
        duration_ms: Operation duration in milliseconds
        status: Success/failure status
        request_id: Unique request identifier
        session_id: Session identifier
        metadata: Additional context
    """

    timestamp: str
    action: AuditAction
    user_id: str
    level: AuditLevel = AuditLevel.INFO
    agent_type: str | None = None
    input_hash: str | None = None
    output_hash: str | None = None
    duration_ms: int | None = None
    status: Literal["success", "failure", "pending"] = "success"
    request_id: str = field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON string."""
        data = asdict(self)
        data["action"] = self.action.value
        data["level"] = self.level.value
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> "AuditEntry":
        """Create from JSON string."""
        data = json.loads(json_str)
        data["action"] = AuditAction(data["action"])
        data["level"] = AuditLevel(data["level"])
        return cls(**data)


@dataclass
class AuditConfig:
    """Configuration for audit logging.

    Attributes:
        enabled: Whether audit logging is enabled
        log_path: Path for audit log files
        log_inputs: Whether to log input content (vs hash only)
        log_outputs: Whether to log output content (vs hash only)
        max_file_size_mb: Maximum log file size before rotation
        retention_days: Days to retain log files
        async_logging: Whether to use async logging
        console_output: Whether to also print to console
    """

    enabled: bool = True
    log_path: str = "./logs/audit.jsonl"
    log_inputs: bool = False  # Privacy: hash by default
    log_outputs: bool = False  # Privacy: hash by default
    max_file_size_mb: int = 100
    retention_days: int = 90
    async_logging: bool = True
    console_output: bool = False

    @classmethod
    def from_env(cls) -> "AuditConfig":
        """Create config from environment variables."""
        return cls(
            enabled=os.getenv("AUDIT_ENABLED", "true").lower() == "true",
            log_path=os.getenv("AUDIT_LOG_PATH", "./logs/audit.jsonl"),
            log_inputs=os.getenv("AUDIT_LOG_INPUTS", "false").lower() == "true",
            log_outputs=os.getenv("AUDIT_LOG_OUTPUTS", "false").lower() == "true",
            max_file_size_mb=int(os.getenv("AUDIT_MAX_FILE_SIZE_MB", "100")),
            retention_days=int(os.getenv("AUDIT_RETENTION_DAYS", "90")),
            async_logging=os.getenv("AUDIT_ASYNC", "true").lower() == "true",
            console_output=os.getenv("AUDIT_CONSOLE", "false").lower() == "true",
        )


class AuditLogger:
    """Audit logger for compliance and tracking.

    Provides:
    - JSON Lines logging for easy parsing
    - Privacy-preserving hashing
    - Async write support
    - Log rotation
    """

    def __init__(self, config: AuditConfig | None = None) -> None:
        """Initialize audit logger.

        Args:
            config: Audit configuration.
        """
        self.config = config or AuditConfig.from_env()
        self._queue: asyncio.Queue[AuditEntry] | None = None
        self._writer_task: asyncio.Task | None = None
        self._initialized = False

    def _ensure_log_directory(self) -> None:
        """Ensure log directory exists."""
        log_path = Path(self.config.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def _hash_content(self, content: str | None) -> str | None:
        """Create SHA-256 hash of content.

        Args:
            content: Content to hash.

        Returns:
            Hex digest of hash, or None if content is None.
        """
        if content is None:
            return None
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _should_rotate(self) -> bool:
        """Check if log file should be rotated."""
        log_path = Path(self.config.log_path)
        if not log_path.exists():
            return False
        size_mb = log_path.stat().st_size / (1024 * 1024)
        return size_mb >= self.config.max_file_size_mb

    def _rotate_log(self) -> None:
        """Rotate the log file."""
        log_path = Path(self.config.log_path)
        if log_path.exists():
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            rotated_name = f"{log_path.stem}-{timestamp}{log_path.suffix}"
            rotated_path = log_path.parent / rotated_name
            log_path.rename(rotated_path)

    def log(
        self,
        action: AuditAction,
        user_id: str,
        level: AuditLevel = AuditLevel.INFO,
        agent_type: str | None = None,
        input_text: str | None = None,
        output_text: str | None = None,
        duration_ms: int | None = None,
        status: Literal["success", "failure", "pending"] = "success",
        request_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Log an audit entry synchronously.

        Args:
            action: Type of action.
            user_id: User performing action.
            level: Severity level.
            agent_type: Type of agent.
            input_text: Input content (will be hashed if config.log_inputs is False).
            output_text: Output content (will be hashed if config.log_outputs is False).
            duration_ms: Operation duration.
            status: Operation status.
            request_id: Request identifier.
            session_id: Session identifier.
            metadata: Additional context.

        Returns:
            The created AuditEntry.
        """
        if not self.config.enabled:
            return AuditEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                action=action,
                user_id=user_id,
            )

        entry = self._create_entry(
            action=action,
            user_id=user_id,
            level=level,
            agent_type=agent_type,
            input_text=input_text,
            output_text=output_text,
            duration_ms=duration_ms,
            status=status,
            request_id=request_id,
            session_id=session_id,
            metadata=metadata,
        )

        self._write_entry(entry)
        return entry

    async def log_async(
        self,
        action: AuditAction,
        user_id: str,
        level: AuditLevel = AuditLevel.INFO,
        agent_type: str | None = None,
        input_text: str | None = None,
        output_text: str | None = None,
        duration_ms: int | None = None,
        status: Literal["success", "failure", "pending"] = "success",
        request_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Log an audit entry asynchronously.

        Args:
            action: Type of action.
            user_id: User performing action.
            level: Severity level.
            agent_type: Type of agent.
            input_text: Input content.
            output_text: Output content.
            duration_ms: Operation duration.
            status: Operation status.
            request_id: Request identifier.
            session_id: Session identifier.
            metadata: Additional context.

        Returns:
            The created AuditEntry.
        """
        if not self.config.enabled:
            return AuditEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                action=action,
                user_id=user_id,
            )

        entry = self._create_entry(
            action=action,
            user_id=user_id,
            level=level,
            agent_type=agent_type,
            input_text=input_text,
            output_text=output_text,
            duration_ms=duration_ms,
            status=status,
            request_id=request_id,
            session_id=session_id,
            metadata=metadata,
        )

        if self.config.async_logging:
            await self._queue_entry(entry)
        else:
            self._write_entry(entry)

        return entry

    def _create_entry(
        self,
        action: AuditAction,
        user_id: str,
        level: AuditLevel,
        agent_type: str | None,
        input_text: str | None,
        output_text: str | None,
        duration_ms: int | None,
        status: Literal["success", "failure", "pending"],
        request_id: str | None,
        session_id: str | None,
        metadata: dict[str, Any] | None,
    ) -> AuditEntry:
        """Create an audit entry."""
        # Handle input/output based on privacy settings
        if self.config.log_inputs and input_text:
            input_hash = input_text  # Store actual content
            if metadata is None:
                metadata = {}
            metadata["input_logged"] = True
        else:
            input_hash = self._hash_content(input_text)

        if self.config.log_outputs and output_text:
            output_hash = output_text  # Store actual content
            if metadata is None:
                metadata = {}
            metadata["output_logged"] = True
        else:
            output_hash = self._hash_content(output_text)

        return AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            action=action,
            user_id=user_id,
            level=level,
            agent_type=agent_type,
            input_hash=input_hash,
            output_hash=output_hash,
            duration_ms=duration_ms,
            status=status,
            request_id=request_id or str(uuid4()),
            session_id=session_id,
            metadata=metadata or {},
        )

    def _write_entry(self, entry: AuditEntry) -> None:
        """Write entry to log file."""
        self._ensure_log_directory()

        # Check for rotation
        if self._should_rotate():
            self._rotate_log()

        # Write entry
        with open(self.config.log_path, "a") as f:
            f.write(entry.to_json() + "\n")

        # Console output if enabled
        if self.config.console_output:
            print(f"[AUDIT] {entry.action.value}: {entry.user_id} - {entry.status}")

    async def _queue_entry(self, entry: AuditEntry) -> None:
        """Queue entry for async writing."""
        if self._queue is None:
            self._queue = asyncio.Queue()
            self._writer_task = asyncio.create_task(self._async_writer())

        await self._queue.put(entry)

    async def _async_writer(self) -> None:
        """Background task to write queued entries."""
        while True:
            try:
                entry = await self._queue.get()
                self._write_entry(entry)
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Audit write error: {e}")

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        user_id: str | None = None,
        action: AuditAction | None = None,
        level: AuditLevel | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query audit log entries.

        Args:
            start_time: Start of time range.
            end_time: End of time range.
            user_id: Filter by user ID.
            action: Filter by action type.
            level: Filter by severity level.
            limit: Maximum entries to return.

        Returns:
            List of matching audit entries.
        """
        log_path = Path(self.config.log_path)
        if not log_path.exists():
            return []

        entries = []
        with open(log_path) as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    entry = AuditEntry.from_json(line)

                    # Apply filters
                    if start_time:
                        entry_time = datetime.fromisoformat(entry.timestamp)
                        if entry_time < start_time:
                            continue
                    if end_time:
                        entry_time = datetime.fromisoformat(entry.timestamp)
                        if entry_time > end_time:
                            continue
                    if user_id and entry.user_id != user_id:
                        continue
                    if action and entry.action != action:
                        continue
                    if level and entry.level != level:
                        continue

                    entries.append(entry)

                    if len(entries) >= limit:
                        break

                except Exception:
                    continue

        return entries

    def export(
        self,
        output_path: str,
        format: Literal["json", "csv"] = "json",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> int:
        """Export audit logs.

        Args:
            output_path: Path for export file.
            format: Export format.
            start_time: Start of time range.
            end_time: End of time range.

        Returns:
            Number of entries exported.
        """
        entries = self.query(start_time=start_time, end_time=end_time, limit=100000)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output, "w") as f:
                json.dump([asdict(e) for e in entries], f, indent=2, default=str)
        elif format == "csv":
            import csv

            with open(output, "w", newline="") as f:
                if entries:
                    writer = csv.DictWriter(f, fieldnames=asdict(entries[0]).keys())
                    writer.writeheader()
                    for entry in entries:
                        row = asdict(entry)
                        row["action"] = entry.action.value
                        row["level"] = entry.level.value
                        row["metadata"] = json.dumps(row["metadata"])
                        writer.writerow(row)

        return len(entries)

    async def close(self) -> None:
        """Close the audit logger and flush pending writes."""
        if self._queue:
            await self._queue.join()
        if self._writer_task:
            self._writer_task.cancel()
            try:
                await self._writer_task
            except asyncio.CancelledError:
                pass


# Global audit logger instance
_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get or create the global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def reset_audit_logger() -> None:
    """Reset the global audit logger."""
    global _audit_logger
    _audit_logger = None


# Convenience functions


def audit_agent_invoke(
    user_id: str,
    agent_type: str,
    input_text: str,
    request_id: str | None = None,
    session_id: str | None = None,
) -> AuditEntry:
    """Log an agent invocation.

    Args:
        user_id: User invoking agent.
        agent_type: Type of agent.
        input_text: Input to agent.
        request_id: Request identifier.
        session_id: Session identifier.

    Returns:
        Created audit entry.
    """
    logger = get_audit_logger()
    return logger.log(
        action=AuditAction.AGENT_INVOKE,
        user_id=user_id,
        agent_type=agent_type,
        input_text=input_text,
        status="pending",
        request_id=request_id,
        session_id=session_id,
    )


def audit_agent_response(
    user_id: str,
    agent_type: str,
    input_text: str,
    output_text: str,
    duration_ms: int,
    request_id: str | None = None,
    session_id: str | None = None,
    success: bool = True,
) -> AuditEntry:
    """Log an agent response.

    Args:
        user_id: User who invoked agent.
        agent_type: Type of agent.
        input_text: Input to agent.
        output_text: Agent response.
        duration_ms: Response duration.
        request_id: Request identifier.
        session_id: Session identifier.
        success: Whether response was successful.

    Returns:
        Created audit entry.
    """
    logger = get_audit_logger()
    return logger.log(
        action=AuditAction.AGENT_RESPONSE,
        user_id=user_id,
        agent_type=agent_type,
        input_text=input_text,
        output_text=output_text,
        duration_ms=duration_ms,
        status="success" if success else "failure",
        request_id=request_id,
        session_id=session_id,
    )


def audit_permission_denied(
    user_id: str,
    permission: str,
    request_id: str | None = None,
) -> AuditEntry:
    """Log a permission denied event.

    Args:
        user_id: User who was denied.
        permission: Permission that was denied.
        request_id: Request identifier.

    Returns:
        Created audit entry.
    """
    logger = get_audit_logger()
    return logger.log(
        action=AuditAction.PERMISSION_DENIED,
        user_id=user_id,
        level=AuditLevel.WARNING,
        status="failure",
        request_id=request_id,
        metadata={"permission": permission},
    )
