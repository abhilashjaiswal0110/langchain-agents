"""Anomaly detection for agent usage patterns.

Detects unusual patterns in agent usage that may indicate:
- Security threats (brute force, credential stuffing)
- Abuse (excessive usage, prompt injection attempts)
- System issues (error spikes, latency anomalies)
- Data exfiltration attempts

Usage:
    from app.governance.anomaly_detector import (
        AnomalyDetector, Anomaly, AnomalyType, AnomalyConfig,
        get_anomaly_detector, check_for_anomalies, record_event,
    )

    # Record events
    record_event(
        user_id="user123",
        agent_type="research",
        event_type="request",
        metadata={"input_length": 500, "response_time_ms": 1200},
    )

    # Check for anomalies
    anomalies = check_for_anomalies(user_id="user123")
    for anomaly in anomalies:
        print(f"Anomaly detected: {anomaly.anomaly_type} - {anomaly.description}")

    # Get detector with custom config
    detector = get_anomaly_detector()
    detector.add_rule("large_input", lambda e: e.metadata.get("input_length", 0) > 10000)
"""

import logging
import statistics
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected."""

    # Rate-based anomalies
    HIGH_REQUEST_RATE = "high_request_rate"
    BURST_ACTIVITY = "burst_activity"
    OFF_HOURS_ACTIVITY = "off_hours_activity"

    # Error-based anomalies
    HIGH_ERROR_RATE = "high_error_rate"
    REPEATED_FAILURES = "repeated_failures"
    AUTH_FAILURES = "auth_failures"

    # Content-based anomalies
    LARGE_INPUT = "large_input"
    LARGE_OUTPUT = "large_output"
    UNUSUAL_CONTENT = "unusual_content"
    PROMPT_INJECTION = "prompt_injection"

    # Pattern-based anomalies
    SEQUENTIAL_ACCESS = "sequential_access"
    DATA_EXFILTRATION = "data_exfiltration"
    CREDENTIAL_STUFFING = "credential_stuffing"

    # Performance anomalies
    HIGH_LATENCY = "high_latency"
    RESOURCE_EXHAUSTION = "resource_exhaustion"

    # Custom rules
    CUSTOM = "custom"


class AnomalySeverity(str, Enum):
    """Severity level of detected anomaly."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Event:
    """An event to analyze for anomalies.

    Attributes:
        event_id: Unique event identifier.
        user_id: User who triggered the event.
        agent_type: Type of agent involved.
        event_type: Type of event (request, error, etc.).
        timestamp: When the event occurred.
        success: Whether the event was successful.
        metadata: Additional event data.
    """

    event_id: str
    user_id: str
    agent_type: str
    event_type: str
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    metadata: dict = field(default_factory=dict)


@dataclass
class Anomaly:
    """A detected anomaly.

    Attributes:
        anomaly_id: Unique anomaly identifier.
        anomaly_type: Type of anomaly.
        severity: Severity level.
        user_id: Affected user.
        agent_type: Affected agent type.
        description: Human-readable description.
        evidence: Supporting evidence.
        timestamp: When the anomaly was detected.
        recommended_action: Suggested response.
    """

    anomaly_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    user_id: str
    agent_type: str
    description: str
    evidence: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    recommended_action: str = ""

    def __repr__(self) -> str:
        return f"Anomaly({self.anomaly_type.value}, severity={self.severity.value}, user={self.user_id})"


@dataclass
class RateConfig:
    """Configuration for rate-based anomaly detection.

    Attributes:
        window_seconds: Time window for rate calculation.
        max_requests_per_window: Maximum requests allowed.
        burst_threshold: Requests per second for burst detection.
        off_hours_start: Start of off-hours (24h format).
        off_hours_end: End of off-hours (24h format).
    """

    window_seconds: int = 60
    max_requests_per_window: int = 100
    burst_threshold: int = 10
    off_hours_start: int = 22  # 10 PM
    off_hours_end: int = 6  # 6 AM


@dataclass
class ErrorConfig:
    """Configuration for error-based anomaly detection.

    Attributes:
        window_seconds: Time window for error rate calculation.
        max_error_rate: Maximum error rate (0.0-1.0).
        consecutive_failures: Number of consecutive failures to trigger.
        auth_failure_threshold: Auth failures before alerting.
    """

    window_seconds: int = 300
    max_error_rate: float = 0.5
    consecutive_failures: int = 5
    auth_failure_threshold: int = 3


@dataclass
class ContentConfig:
    """Configuration for content-based anomaly detection.

    Attributes:
        max_input_length: Maximum input length.
        max_output_length: Maximum output length.
        injection_patterns: Patterns indicating prompt injection.
    """

    max_input_length: int = 50000
    max_output_length: int = 100000
    injection_patterns: list[str] = field(
        default_factory=lambda: [
            "ignore previous",
            "ignore all previous",
            "disregard previous",
            "forget your instructions",
            "new instructions:",
            "system prompt:",
            "you are now",
            "pretend you are",
            "act as if",
            "reveal your prompt",
            "show your instructions",
            "what are your instructions",
            "bypass",
            "jailbreak",
            "DAN mode",
        ]
    )


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detector.

    Attributes:
        enabled: Whether anomaly detection is enabled.
        rate_config: Rate-based detection config.
        error_config: Error-based detection config.
        content_config: Content-based detection config.
        event_retention_hours: How long to keep events.
        alert_callback: Callback for anomaly alerts.
        auto_block: Whether to auto-block on critical anomalies.
    """

    enabled: bool = True
    rate_config: RateConfig = field(default_factory=RateConfig)
    error_config: ErrorConfig = field(default_factory=ErrorConfig)
    content_config: ContentConfig = field(default_factory=ContentConfig)
    event_retention_hours: int = 24
    alert_callback: Callable[[Anomaly], None] | None = None
    auto_block: bool = False


class AnomalyDetector:
    """Detects anomalies in agent usage patterns.

    Uses statistical analysis and rule-based detection to identify
    unusual patterns that may indicate security threats or abuse.
    """

    def __init__(self, config: AnomalyConfig | None = None) -> None:
        """Initialize anomaly detector.

        Args:
            config: Detector configuration.
        """
        self.config = config or AnomalyConfig()

        # Event storage (per user)
        self._events: dict[str, deque[Event]] = defaultdict(lambda: deque(maxlen=10000))

        # Per-user statistics
        self._user_stats: dict[str, dict] = defaultdict(
            lambda: {
                "request_times": deque(maxlen=1000),
                "error_times": deque(maxlen=100),
                "response_times_ms": deque(maxlen=100),
                "consecutive_failures": 0,
                "last_success_time": 0,
            }
        )

        # Global statistics
        self._global_stats = {
            "avg_response_time_ms": 0.0,
            "std_response_time_ms": 0.0,
            "avg_input_length": 0.0,
            "std_input_length": 0.0,
        }

        # Custom rules
        self._custom_rules: dict[str, tuple[Callable[[Event], bool], AnomalySeverity, str]] = {}

        # Detected anomalies
        self._anomalies: list[Anomaly] = []

        # Blocked users (for auto-block)
        self._blocked_users: set[str] = set()

    def record_event(self, event: Event) -> list[Anomaly]:
        """Record an event and check for anomalies.

        Args:
            event: Event to record.

        Returns:
            List of detected anomalies.
        """
        if not self.config.enabled:
            return []

        # Store event
        self._events[event.user_id].append(event)

        # Update statistics
        self._update_stats(event)

        # Check for anomalies
        anomalies = self._check_anomalies(event)

        # Store detected anomalies
        self._anomalies.extend(anomalies)

        # Trigger alerts
        for anomaly in anomalies:
            self._trigger_alert(anomaly)

        return anomalies

    def _update_stats(self, event: Event) -> None:
        """Update statistics based on event.

        Args:
            event: Event to process.
        """
        user_stats = self._user_stats[event.user_id]

        # Track request times
        user_stats["request_times"].append(event.timestamp)

        # Track errors
        if not event.success:
            user_stats["error_times"].append(event.timestamp)
            user_stats["consecutive_failures"] += 1
        else:
            user_stats["consecutive_failures"] = 0
            user_stats["last_success_time"] = event.timestamp

        # Track response times
        if "response_time_ms" in event.metadata:
            response_time = event.metadata["response_time_ms"]
            user_stats["response_times_ms"].append(response_time)

            # Update global averages
            all_response_times = []
            for stats in self._user_stats.values():
                all_response_times.extend(stats["response_times_ms"])

            if all_response_times:
                self._global_stats["avg_response_time_ms"] = statistics.mean(all_response_times)
                if len(all_response_times) > 1:
                    self._global_stats["std_response_time_ms"] = statistics.stdev(all_response_times)

    def _check_anomalies(self, event: Event) -> list[Anomaly]:
        """Check for anomalies based on current event.

        Args:
            event: Current event.

        Returns:
            List of detected anomalies.
        """
        anomalies = []

        # Rate-based checks
        anomalies.extend(self._check_rate_anomalies(event))

        # Error-based checks
        anomalies.extend(self._check_error_anomalies(event))

        # Content-based checks
        anomalies.extend(self._check_content_anomalies(event))

        # Performance checks
        anomalies.extend(self._check_performance_anomalies(event))

        # Custom rule checks
        anomalies.extend(self._check_custom_rules(event))

        return anomalies

    def _check_rate_anomalies(self, event: Event) -> list[Anomaly]:
        """Check for rate-based anomalies.

        Args:
            event: Current event.

        Returns:
            List of detected anomalies.
        """
        anomalies = []
        config = self.config.rate_config
        user_stats = self._user_stats[event.user_id]

        # Calculate request rate in window
        now = event.timestamp
        window_start = now - config.window_seconds
        recent_requests = [t for t in user_stats["request_times"] if t >= window_start]
        request_count = len(recent_requests)

        # High request rate
        if request_count > config.max_requests_per_window:
            anomalies.append(
                self._create_anomaly(
                    AnomalyType.HIGH_REQUEST_RATE,
                    AnomalySeverity.HIGH,
                    event,
                    f"Request rate ({request_count}/{config.window_seconds}s) exceeds limit",
                    {"count": request_count, "limit": config.max_requests_per_window},
                    "Consider rate limiting or blocking user",
                )
            )

        # Burst detection
        if len(recent_requests) >= 2:
            # Check requests in last 1 second
            one_second_ago = now - 1
            burst_count = sum(1 for t in recent_requests if t >= one_second_ago)
            if burst_count > config.burst_threshold:
                anomalies.append(
                    self._create_anomaly(
                        AnomalyType.BURST_ACTIVITY,
                        AnomalySeverity.MEDIUM,
                        event,
                        f"Burst activity detected ({burst_count} requests/second)",
                        {"burst_count": burst_count, "threshold": config.burst_threshold},
                        "Monitor for continued burst activity",
                    )
                )

        # Off-hours activity
        current_hour = datetime.fromtimestamp(now).hour
        is_off_hours = current_hour >= config.off_hours_start or current_hour < config.off_hours_end
        if is_off_hours and request_count > 10:
            anomalies.append(
                self._create_anomaly(
                    AnomalyType.OFF_HOURS_ACTIVITY,
                    AnomalySeverity.LOW,
                    event,
                    f"Unusual off-hours activity (hour {current_hour}, {request_count} requests)",
                    {"hour": current_hour, "request_count": request_count},
                    "Review if activity is expected",
                )
            )

        return anomalies

    def _check_error_anomalies(self, event: Event) -> list[Anomaly]:
        """Check for error-based anomalies.

        Args:
            event: Current event.

        Returns:
            List of detected anomalies.
        """
        anomalies = []
        config = self.config.error_config
        user_stats = self._user_stats[event.user_id]

        # Consecutive failures
        if user_stats["consecutive_failures"] >= config.consecutive_failures:
            anomalies.append(
                self._create_anomaly(
                    AnomalyType.REPEATED_FAILURES,
                    AnomalySeverity.MEDIUM,
                    event,
                    f"User has {user_stats['consecutive_failures']} consecutive failures",
                    {"consecutive_failures": user_stats["consecutive_failures"]},
                    "Check for misconfiguration or attack",
                )
            )

        # Error rate
        now = event.timestamp
        window_start = now - config.window_seconds
        recent_requests = [t for t in user_stats["request_times"] if t >= window_start]
        recent_errors = [t for t in user_stats["error_times"] if t >= window_start]

        if len(recent_requests) > 10:
            error_rate = len(recent_errors) / len(recent_requests)
            if error_rate > config.max_error_rate:
                anomalies.append(
                    self._create_anomaly(
                        AnomalyType.HIGH_ERROR_RATE,
                        AnomalySeverity.HIGH,
                        event,
                        f"Error rate ({error_rate:.1%}) exceeds threshold",
                        {
                            "error_rate": error_rate,
                            "errors": len(recent_errors),
                            "total": len(recent_requests),
                        },
                        "Investigate error causes",
                    )
                )

        # Auth failures
        if event.metadata.get("error_type") == "auth_failure":
            auth_failures = sum(
                1
                for e in self._events[event.user_id]
                if e.metadata.get("error_type") == "auth_failure" and e.timestamp >= window_start
            )
            if auth_failures >= config.auth_failure_threshold:
                anomalies.append(
                    self._create_anomaly(
                        AnomalyType.AUTH_FAILURES,
                        AnomalySeverity.CRITICAL,
                        event,
                        f"Multiple auth failures ({auth_failures}) detected",
                        {"auth_failures": auth_failures},
                        "Consider blocking user or requiring additional verification",
                    )
                )

        return anomalies

    def _check_content_anomalies(self, event: Event) -> list[Anomaly]:
        """Check for content-based anomalies.

        Args:
            event: Current event.

        Returns:
            List of detected anomalies.
        """
        anomalies = []
        config = self.config.content_config

        # Large input
        input_length = event.metadata.get("input_length", 0)
        if input_length > config.max_input_length:
            anomalies.append(
                self._create_anomaly(
                    AnomalyType.LARGE_INPUT,
                    AnomalySeverity.MEDIUM,
                    event,
                    f"Large input detected ({input_length} chars)",
                    {"input_length": input_length, "max": config.max_input_length},
                    "Review input content for abuse",
                )
            )

        # Large output
        output_length = event.metadata.get("output_length", 0)
        if output_length > config.max_output_length:
            anomalies.append(
                self._create_anomaly(
                    AnomalyType.LARGE_OUTPUT,
                    AnomalySeverity.MEDIUM,
                    event,
                    f"Large output detected ({output_length} chars)",
                    {"output_length": output_length, "max": config.max_output_length},
                    "Review for data exfiltration",
                )
            )

        # Prompt injection detection
        input_text = event.metadata.get("input_text", "").lower()
        for pattern in config.injection_patterns:
            if pattern.lower() in input_text:
                anomalies.append(
                    self._create_anomaly(
                        AnomalyType.PROMPT_INJECTION,
                        AnomalySeverity.HIGH,
                        event,
                        f"Possible prompt injection detected (pattern: '{pattern}')",
                        {"pattern": pattern},
                        "Block request and investigate user",
                    )
                )
                break  # Only report first match

        return anomalies

    def _check_performance_anomalies(self, event: Event) -> list[Anomaly]:
        """Check for performance anomalies.

        Args:
            event: Current event.

        Returns:
            List of detected anomalies.
        """
        anomalies = []

        response_time = event.metadata.get("response_time_ms", 0)
        if response_time == 0:
            return anomalies

        avg = self._global_stats["avg_response_time_ms"]
        std = self._global_stats["std_response_time_ms"]

        # Detect high latency (> 3 standard deviations)
        if avg > 0 and std > 0:
            z_score = (response_time - avg) / std if std > 0 else 0
            if z_score > 3:
                anomalies.append(
                    self._create_anomaly(
                        AnomalyType.HIGH_LATENCY,
                        AnomalySeverity.LOW,
                        event,
                        f"High latency detected ({response_time}ms, avg={avg:.0f}ms)",
                        {
                            "response_time_ms": response_time,
                            "avg_ms": avg,
                            "z_score": z_score,
                        },
                        "Check system resources",
                    )
                )

        return anomalies

    def _check_custom_rules(self, event: Event) -> list[Anomaly]:
        """Check custom detection rules.

        Args:
            event: Current event.

        Returns:
            List of detected anomalies.
        """
        anomalies = []

        for rule_name, (rule_func, severity, description) in self._custom_rules.items():
            try:
                if rule_func(event):
                    anomalies.append(
                        self._create_anomaly(
                            AnomalyType.CUSTOM,
                            severity,
                            event,
                            description or f"Custom rule '{rule_name}' triggered",
                            {"rule": rule_name},
                            "Review custom rule criteria",
                        )
                    )
            except Exception as e:
                logger.warning(f"Custom rule '{rule_name}' failed: {e}")

        return anomalies

    def _create_anomaly(
        self,
        anomaly_type: AnomalyType,
        severity: AnomalySeverity,
        event: Event,
        description: str,
        evidence: dict,
        recommended_action: str,
    ) -> Anomaly:
        """Create an anomaly record.

        Args:
            anomaly_type: Type of anomaly.
            severity: Severity level.
            event: Triggering event.
            description: Description.
            evidence: Supporting evidence.
            recommended_action: Suggested action.

        Returns:
            Anomaly record.
        """
        import uuid

        return Anomaly(
            anomaly_id=str(uuid.uuid4()),
            anomaly_type=anomaly_type,
            severity=severity,
            user_id=event.user_id,
            agent_type=event.agent_type,
            description=description,
            evidence=evidence,
            recommended_action=recommended_action,
        )

    def _trigger_alert(self, anomaly: Anomaly) -> None:
        """Trigger alert for detected anomaly.

        Args:
            anomaly: Detected anomaly.
        """
        logger.warning(
            f"Anomaly detected: {anomaly.anomaly_type.value} "
            f"(severity={anomaly.severity.value}, user={anomaly.user_id})"
        )

        # Auto-block on critical
        if self.config.auto_block and anomaly.severity == AnomalySeverity.CRITICAL:
            self._blocked_users.add(anomaly.user_id)
            logger.warning(f"User {anomaly.user_id} auto-blocked due to critical anomaly")

        # Call custom callback
        if self.config.alert_callback:
            try:
                self.config.alert_callback(anomaly)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def add_rule(
        self,
        name: str,
        rule_func: Callable[[Event], bool],
        severity: AnomalySeverity = AnomalySeverity.MEDIUM,
        description: str = "",
    ) -> None:
        """Add a custom detection rule.

        Args:
            name: Rule name.
            rule_func: Function that returns True if anomaly detected.
            severity: Severity level for this rule.
            description: Description when triggered.
        """
        self._custom_rules[name] = (rule_func, severity, description)

    def remove_rule(self, name: str) -> bool:
        """Remove a custom rule.

        Args:
            name: Rule name.

        Returns:
            True if rule was removed.
        """
        if name in self._custom_rules:
            del self._custom_rules[name]
            return True
        return False

    def is_blocked(self, user_id: str) -> bool:
        """Check if a user is blocked.

        Args:
            user_id: User to check.

        Returns:
            True if user is blocked.
        """
        return user_id in self._blocked_users

    def unblock_user(self, user_id: str) -> bool:
        """Unblock a user.

        Args:
            user_id: User to unblock.

        Returns:
            True if user was unblocked.
        """
        if user_id in self._blocked_users:
            self._blocked_users.discard(user_id)
            return True
        return False

    def get_anomalies(
        self,
        user_id: str = "",
        anomaly_type: AnomalyType | None = None,
        severity: AnomalySeverity | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[Anomaly]:
        """Get detected anomalies with filtering.

        Args:
            user_id: Filter by user.
            anomaly_type: Filter by type.
            severity: Filter by severity.
            since: Filter by timestamp.
            limit: Maximum to return.

        Returns:
            Filtered list of anomalies.
        """
        anomalies = self._anomalies

        if user_id:
            anomalies = [a for a in anomalies if a.user_id == user_id]
        if anomaly_type:
            anomalies = [a for a in anomalies if a.anomaly_type == anomaly_type]
        if severity:
            anomalies = [a for a in anomalies if a.severity == severity]
        if since:
            anomalies = [a for a in anomalies if a.timestamp >= since]

        # Sort by timestamp descending
        anomalies = sorted(anomalies, key=lambda a: a.timestamp, reverse=True)

        return anomalies[:limit]

    def get_user_risk_score(self, user_id: str) -> float:
        """Calculate risk score for a user.

        Args:
            user_id: User to score.

        Returns:
            Risk score (0.0 to 1.0).
        """
        # Get recent anomalies for user
        one_hour_ago = time.time() - 3600
        anomalies = self.get_anomalies(user_id=user_id, since=one_hour_ago)

        if not anomalies:
            return 0.0

        # Weight by severity
        severity_weights = {
            AnomalySeverity.LOW: 0.1,
            AnomalySeverity.MEDIUM: 0.3,
            AnomalySeverity.HIGH: 0.6,
            AnomalySeverity.CRITICAL: 1.0,
        }

        score = sum(severity_weights.get(a.severity, 0.1) for a in anomalies)

        # Normalize to 0-1 range (cap at 10 weighted anomalies)
        return min(score / 10.0, 1.0)

    def cleanup_old_events(self) -> int:
        """Remove events older than retention period.

        Returns:
            Number of events removed.
        """
        cutoff = time.time() - (self.config.event_retention_hours * 3600)
        removed = 0

        for user_id in list(self._events.keys()):
            original_len = len(self._events[user_id])
            self._events[user_id] = deque(
                (e for e in self._events[user_id] if e.timestamp >= cutoff),
                maxlen=10000,
            )
            removed += original_len - len(self._events[user_id])

        if removed > 0:
            logger.info(f"Cleaned up {removed} old events")

        return removed


# Singleton pattern for global detector
_anomaly_detector: AnomalyDetector | None = None


def get_anomaly_detector(config: AnomalyConfig | None = None) -> AnomalyDetector:
    """Get or create global anomaly detector instance.

    Args:
        config: Optional configuration (used only on first call).

    Returns:
        Global anomaly detector instance.
    """
    global _anomaly_detector
    if _anomaly_detector is None:
        _anomaly_detector = AnomalyDetector(config)
    return _anomaly_detector


def reset_anomaly_detector() -> None:
    """Reset global anomaly detector instance."""
    global _anomaly_detector
    _anomaly_detector = None


def record_event(
    user_id: str,
    agent_type: str,
    event_type: str = "request",
    success: bool = True,
    metadata: dict | None = None,
) -> list[Anomaly]:
    """Convenience function to record an event.

    Args:
        user_id: User identifier.
        agent_type: Agent type.
        event_type: Event type.
        success: Whether event was successful.
        metadata: Additional data.

    Returns:
        List of detected anomalies.
    """
    import uuid

    detector = get_anomaly_detector()
    event = Event(
        event_id=str(uuid.uuid4()),
        user_id=user_id,
        agent_type=agent_type,
        event_type=event_type,
        success=success,
        metadata=metadata or {},
    )
    return detector.record_event(event)


def check_for_anomalies(
    user_id: str = "",
    since: float | None = None,
) -> list[Anomaly]:
    """Convenience function to get recent anomalies.

    Args:
        user_id: Filter by user.
        since: Filter by timestamp.

    Returns:
        List of anomalies.
    """
    detector = get_anomaly_detector()
    return detector.get_anomalies(user_id=user_id, since=since)


class AnomalyBlockedError(Exception):
    """Raised when user is blocked due to anomalies."""

    def __init__(
        self,
        message: str,
        user_id: str,
        anomalies: list[Anomaly],
    ) -> None:
        """Initialize error.

        Args:
            message: Error message.
            user_id: Blocked user.
            anomalies: Related anomalies.
        """
        super().__init__(message)
        self.user_id = user_id
        self.anomalies = anomalies
