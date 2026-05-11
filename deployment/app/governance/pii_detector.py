"""PII (Personally Identifiable Information) detection and masking.

Provides privacy protection for agent inputs and outputs by detecting
and optionally masking PII such as:
- Email addresses
- Phone numbers
- Credit card numbers
- SSN/National IDs
- IP addresses
- Names (using Presidio when available)
- Addresses
- Dates of birth

Usage:
    from app.governance.pii_detector import (
        PIIDetector, PIIType, PIIMatch, PIIConfig,
        get_pii_detector, detect_pii, mask_pii,
    )

    # Simple detection
    matches = detect_pii("Contact john@email.com or 555-123-4567")

    # Masking
    masked = mask_pii("My email is john@email.com")
    # Returns: "My email is [EMAIL_REDACTED]"

    # Full detector with config
    detector = get_pii_detector()
    result = detector.analyze("Text with PII")
    masked_text = detector.mask(result)
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PIIType(str, Enum):
    """Types of PII that can be detected."""

    EMAIL = "email"
    PHONE = "phone"
    CREDIT_CARD = "credit_card"
    SSN = "ssn"
    IP_ADDRESS = "ip_address"
    DATE_OF_BIRTH = "date_of_birth"
    NAME = "name"
    ADDRESS = "address"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    BANK_ACCOUNT = "bank_account"
    API_KEY = "api_key"
    PASSWORD = "password"
    IBAN = "iban"
    URL = "url"
    CUSTOM = "custom"


class PIISeverity(str, Enum):
    """Severity level of PII detection."""

    LOW = "low"  # Public info like URLs
    MEDIUM = "medium"  # Semi-public like names, addresses
    HIGH = "high"  # Sensitive like phone, email
    CRITICAL = "critical"  # Very sensitive like SSN, credit cards


@dataclass
class PIIMatch:
    """A detected PII match.

    Attributes:
        pii_type: Type of PII detected.
        value: The matched value.
        start: Start position in text.
        end: End position in text.
        severity: Severity level.
        confidence: Detection confidence (0.0-1.0).
        context: Surrounding text context.
    """

    pii_type: PIIType
    value: str
    start: int
    end: int
    severity: PIISeverity = PIISeverity.HIGH
    confidence: float = 1.0
    context: str = ""

    @property
    def masked_value(self) -> str:
        """Get masked representation of the PII."""
        return f"[{self.pii_type.value.upper()}_REDACTED]"

    def __repr__(self) -> str:
        return f"PIIMatch({self.pii_type.value}, pos={self.start}-{self.end}, conf={self.confidence:.2f})"


@dataclass
class PIIAnalysisResult:
    """Result of PII analysis.

    Attributes:
        text: Original text analyzed.
        matches: List of detected PII matches.
        has_pii: Whether any PII was detected.
        severity: Highest severity found.
    """

    text: str
    matches: list[PIIMatch] = field(default_factory=list)

    @property
    def has_pii(self) -> bool:
        """Check if any PII was detected."""
        return len(self.matches) > 0

    @property
    def severity(self) -> PIISeverity | None:
        """Get highest severity level found."""
        if not self.matches:
            return None

        severity_order = [
            PIISeverity.LOW,
            PIISeverity.MEDIUM,
            PIISeverity.HIGH,
            PIISeverity.CRITICAL,
        ]
        max_severity = PIISeverity.LOW
        for match in self.matches:
            if severity_order.index(match.severity) > severity_order.index(max_severity):
                max_severity = match.severity
        return max_severity

    @property
    def pii_types_found(self) -> set[PIIType]:
        """Get set of PII types found."""
        return {m.pii_type for m in self.matches}

    def get_matches_by_type(self, pii_type: PIIType) -> list[PIIMatch]:
        """Get matches of a specific type."""
        return [m for m in self.matches if m.pii_type == pii_type]

    def get_matches_by_severity(self, severity: PIISeverity) -> list[PIIMatch]:
        """Get matches of a specific severity."""
        return [m for m in self.matches if m.severity == severity]


@dataclass
class PIIConfig:
    """Configuration for PII detector.

    Attributes:
        enabled: Whether PII detection is enabled.
        enabled_types: Which PII types to detect (None = all).
        min_confidence: Minimum confidence threshold.
        use_presidio: Whether to use Presidio if available.
        mask_char: Character to use for partial masking.
        redaction_format: Format for redaction placeholder.
        log_detections: Whether to log detections.
        block_on_pii: Whether to block requests with PII.
        allowed_pii_types: PII types that are allowed through.
    """

    enabled: bool = True
    enabled_types: set[PIIType] | None = None
    min_confidence: float = 0.5
    use_presidio: bool = True
    mask_char: str = "*"
    redaction_format: str = "[{type}_REDACTED]"
    log_detections: bool = True
    block_on_pii: bool = False
    allowed_pii_types: set[PIIType] = field(default_factory=set)


class PIIDetector:
    """Detects and masks PII in text.

    Supports both regex-based detection and Presidio-based detection
    when the presidio-analyzer package is available.
    """

    # Regex patterns for common PII types
    PATTERNS: dict[PIIType, tuple[str, PIISeverity, float]] = {
        PIIType.EMAIL: (
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            PIISeverity.HIGH,
            0.95,
        ),
        PIIType.PHONE: (
            r"\b(?:\+?1[-.\s]?)?(?:\([0-9]{3}\)|[0-9]{3})[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
            PIISeverity.HIGH,
            0.85,
        ),
        PIIType.CREDIT_CARD: (
            r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
            PIISeverity.CRITICAL,
            0.95,
        ),
        PIIType.SSN: (
            r"\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b",
            PIISeverity.CRITICAL,
            0.9,
        ),
        PIIType.IP_ADDRESS: (
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
            PIISeverity.MEDIUM,
            0.95,
        ),
        PIIType.DATE_OF_BIRTH: (
            r"\b(?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12][0-9]|3[01])[-/](?:19|20)\d{2}\b",
            PIISeverity.HIGH,
            0.7,
        ),
        PIIType.IBAN: (
            r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]?){0,16}\b",
            PIISeverity.CRITICAL,
            0.9,
        ),
        PIIType.API_KEY: (
            r"\b(?:sk-|pk-|api[-_]?key[-_]?)[A-Za-z0-9]{20,}\b",
            PIISeverity.CRITICAL,
            0.85,
        ),
        PIIType.PASSWORD: (
            r"(?i)\b(?:password|passwd|pwd)\s*[:=]\s*[^\s]+",
            PIISeverity.CRITICAL,
            0.8,
        ),
    }

    def __init__(self, config: PIIConfig | None = None) -> None:
        """Initialize PII detector.

        Args:
            config: Detector configuration.
        """
        self.config = config or PIIConfig()
        self._presidio_analyzer = None
        self._presidio_available = False
        self._custom_patterns: dict[str, tuple[str, PIISeverity, float]] = {}

        # Try to load Presidio if enabled
        if self.config.use_presidio:
            self._init_presidio()

    def _init_presidio(self) -> None:
        """Initialize Presidio analyzer if available."""
        try:
            from presidio_analyzer import AnalyzerEngine

            self._presidio_analyzer = AnalyzerEngine()
            self._presidio_available = True
            logger.info("Presidio analyzer loaded successfully")
        except ImportError:
            logger.info("Presidio not available, using regex-based detection only")
            self._presidio_available = False

    def add_custom_pattern(
        self,
        name: str,
        pattern: str,
        severity: PIISeverity = PIISeverity.HIGH,
        confidence: float = 0.8,
    ) -> None:
        """Add a custom regex pattern for detection.

        Args:
            name: Name for the pattern.
            pattern: Regex pattern.
            severity: Severity level.
            confidence: Base confidence score.
        """
        self._custom_patterns[name] = (pattern, severity, confidence)

    def analyze(self, text: str) -> PIIAnalysisResult:
        """Analyze text for PII.

        Args:
            text: Text to analyze.

        Returns:
            Analysis result with detected PII.
        """
        if not self.config.enabled or not text:
            return PIIAnalysisResult(text=text)

        matches: list[PIIMatch] = []

        # Regex-based detection
        matches.extend(self._detect_with_regex(text))

        # Presidio-based detection
        if self._presidio_available and self.config.use_presidio:
            matches.extend(self._detect_with_presidio(text))

        # Custom patterns
        matches.extend(self._detect_custom_patterns(text))

        # Deduplicate overlapping matches
        matches = self._deduplicate_matches(matches)

        # Filter by enabled types
        if self.config.enabled_types:
            matches = [m for m in matches if m.pii_type in self.config.enabled_types]

        # Filter by confidence
        matches = [m for m in matches if m.confidence >= self.config.min_confidence]

        # Sort by position
        matches.sort(key=lambda m: m.start)

        result = PIIAnalysisResult(text=text, matches=matches)

        if self.config.log_detections and result.has_pii:
            logger.info(
                f"PII detected: {len(matches)} matches, types: {result.pii_types_found}, severity: {result.severity}"
            )

        return result

    def _detect_with_regex(self, text: str) -> list[PIIMatch]:
        """Detect PII using regex patterns.

        Args:
            text: Text to analyze.

        Returns:
            List of detected matches.
        """
        matches = []

        for pii_type, (pattern, severity, base_confidence) in self.PATTERNS.items():
            try:
                regex = re.compile(pattern, re.IGNORECASE)
                for match in regex.finditer(text):
                    # Get surrounding context
                    start = max(0, match.start() - 20)
                    end = min(len(text), match.end() + 20)
                    context = text[start:end]

                    matches.append(
                        PIIMatch(
                            pii_type=pii_type,
                            value=match.group(),
                            start=match.start(),
                            end=match.end(),
                            severity=severity,
                            confidence=base_confidence,
                            context=context,
                        )
                    )
            except re.error as e:
                logger.warning(f"Invalid regex pattern for {pii_type}: {e}")

        return matches

    def _detect_with_presidio(self, text: str) -> list[PIIMatch]:
        """Detect PII using Presidio analyzer.

        Args:
            text: Text to analyze.

        Returns:
            List of detected matches.
        """
        if not self._presidio_analyzer:
            return []

        matches = []

        try:
            results = self._presidio_analyzer.analyze(
                text=text,
                language="en",
                entities=None,  # Detect all entities
            )

            # Map Presidio entity types to our PIIType
            type_mapping = {
                "EMAIL_ADDRESS": PIIType.EMAIL,
                "PHONE_NUMBER": PIIType.PHONE,
                "CREDIT_CARD": PIIType.CREDIT_CARD,
                "US_SSN": PIIType.SSN,
                "IP_ADDRESS": PIIType.IP_ADDRESS,
                "DATE_TIME": PIIType.DATE_OF_BIRTH,
                "PERSON": PIIType.NAME,
                "LOCATION": PIIType.ADDRESS,
                "US_PASSPORT": PIIType.PASSPORT,
                "US_DRIVER_LICENSE": PIIType.DRIVER_LICENSE,
                "US_BANK_NUMBER": PIIType.BANK_ACCOUNT,
                "IBAN_CODE": PIIType.IBAN,
                "URL": PIIType.URL,
            }

            severity_mapping = {
                "EMAIL_ADDRESS": PIISeverity.HIGH,
                "PHONE_NUMBER": PIISeverity.HIGH,
                "CREDIT_CARD": PIISeverity.CRITICAL,
                "US_SSN": PIISeverity.CRITICAL,
                "IP_ADDRESS": PIISeverity.MEDIUM,
                "DATE_TIME": PIISeverity.MEDIUM,
                "PERSON": PIISeverity.MEDIUM,
                "LOCATION": PIISeverity.LOW,
                "US_PASSPORT": PIISeverity.CRITICAL,
                "US_DRIVER_LICENSE": PIISeverity.HIGH,
                "US_BANK_NUMBER": PIISeverity.CRITICAL,
                "IBAN_CODE": PIISeverity.CRITICAL,
                "URL": PIISeverity.LOW,
            }

            for result in results:
                pii_type = type_mapping.get(result.entity_type, PIIType.CUSTOM)
                severity = severity_mapping.get(result.entity_type, PIISeverity.HIGH)

                # Get context
                start = max(0, result.start - 20)
                end = min(len(text), result.end + 20)
                context = text[start:end]

                matches.append(
                    PIIMatch(
                        pii_type=pii_type,
                        value=text[result.start : result.end],
                        start=result.start,
                        end=result.end,
                        severity=severity,
                        confidence=result.score,
                        context=context,
                    )
                )

        except Exception as e:
            logger.warning(f"Presidio analysis failed: {e}")

        return matches

    def _detect_custom_patterns(self, text: str) -> list[PIIMatch]:
        """Detect PII using custom patterns.

        Args:
            text: Text to analyze.

        Returns:
            List of detected matches.
        """
        matches = []

        for name, (pattern, severity, confidence) in self._custom_patterns.items():
            try:
                regex = re.compile(pattern, re.IGNORECASE)
                for match in regex.finditer(text):
                    start = max(0, match.start() - 20)
                    end = min(len(text), match.end() + 20)
                    context = text[start:end]

                    matches.append(
                        PIIMatch(
                            pii_type=PIIType.CUSTOM,
                            value=match.group(),
                            start=match.start(),
                            end=match.end(),
                            severity=severity,
                            confidence=confidence,
                            context=context,
                        )
                    )
            except re.error as e:
                logger.warning(f"Invalid custom pattern '{name}': {e}")

        return matches

    def _deduplicate_matches(self, matches: list[PIIMatch]) -> list[PIIMatch]:
        """Remove overlapping matches, keeping highest confidence.

        Args:
            matches: List of matches to deduplicate.

        Returns:
            Deduplicated list.
        """
        if not matches:
            return []

        # Sort by start position, then by confidence (descending)
        sorted_matches = sorted(matches, key=lambda m: (m.start, -m.confidence))

        deduplicated = []
        last_end = -1

        for match in sorted_matches:
            # If this match starts after the last one ended, keep it
            if match.start >= last_end:
                deduplicated.append(match)
                last_end = match.end
            # If overlapping, only keep if higher confidence and same type
            elif match.confidence > deduplicated[-1].confidence:
                deduplicated[-1] = match
                last_end = match.end

        return deduplicated

    def mask(
        self,
        result: PIIAnalysisResult,
        mask_types: set[PIIType] | None = None,
    ) -> str:
        """Mask detected PII in text.

        Args:
            result: Analysis result with detected PII.
            mask_types: Specific types to mask (None = all).

        Returns:
            Text with PII masked.
        """
        if not result.has_pii:
            return result.text

        text = result.text
        # Process matches in reverse order to preserve positions
        matches = sorted(result.matches, key=lambda m: m.start, reverse=True)

        for match in matches:
            # Skip if not in allowed types
            if match.pii_type in self.config.allowed_pii_types:
                continue

            # Skip if mask_types specified and not included
            if mask_types and match.pii_type not in mask_types:
                continue

            # Generate redaction text
            redaction = self.config.redaction_format.format(type=match.pii_type.value.upper())

            text = text[: match.start] + redaction + text[match.end :]

        return text

    def mask_text(self, text: str, mask_types: set[PIIType] | None = None) -> str:
        """Analyze and mask PII in one step.

        Args:
            text: Text to analyze and mask.
            mask_types: Specific types to mask (None = all).

        Returns:
            Text with PII masked.
        """
        result = self.analyze(text)
        return self.mask(result, mask_types)


# Singleton pattern for global detector
_pii_detector: PIIDetector | None = None


def get_pii_detector(config: PIIConfig | None = None) -> PIIDetector:
    """Get or create global PII detector instance.

    Args:
        config: Optional configuration (used only on first call).

    Returns:
        Global PII detector instance.
    """
    global _pii_detector
    if _pii_detector is None:
        _pii_detector = PIIDetector(config)
    return _pii_detector


def reset_pii_detector() -> None:
    """Reset global PII detector instance."""
    global _pii_detector
    _pii_detector = None


def detect_pii(text: str) -> list[PIIMatch]:
    """Convenience function to detect PII in text.

    Args:
        text: Text to analyze.

    Returns:
        List of detected PII matches.
    """
    detector = get_pii_detector()
    result = detector.analyze(text)
    return result.matches


def mask_pii(text: str, mask_types: set[PIIType] | None = None) -> str:
    """Convenience function to mask PII in text.

    Args:
        text: Text to mask.
        mask_types: Specific types to mask (None = all).

    Returns:
        Text with PII masked.
    """
    detector = get_pii_detector()
    return detector.mask_text(text, mask_types)


def check_for_pii(text: str, block_types: set[PIIType] | None = None) -> bool:
    """Check if text contains blocked PII types.

    Args:
        text: Text to check.
        block_types: Types that should trigger blocking (None = all critical).

    Returns:
        True if blocked PII is found.
    """
    if block_types is None:
        block_types = {PIIType.CREDIT_CARD, PIIType.SSN, PIIType.API_KEY, PIIType.PASSWORD}

    detector = get_pii_detector()
    result = detector.analyze(text)

    for match in result.matches:
        if match.pii_type in block_types:
            return True

    return False


class PIIBlockedError(Exception):
    """Raised when PII is detected and blocking is enabled."""

    def __init__(
        self,
        message: str,
        pii_types: set[PIIType],
        severity: PIISeverity,
    ) -> None:
        """Initialize error.

        Args:
            message: Error message.
            pii_types: Types of PII found.
            severity: Highest severity found.
        """
        super().__init__(message)
        self.pii_types = pii_types
        self.severity = severity
