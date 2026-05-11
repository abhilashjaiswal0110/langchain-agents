"""Prompt injection and jailbreak detection.

Provides security protection for agent inputs by detecting attempts to
subvert agent behavior through prompt injection techniques such as:
- Instruction override attacks ("ignore previous instructions")
- Persona hijacking ("you are now", "act as")
- Known jailbreak keywords (DAN, jailbreak)
- Training/guideline disregard patterns
- Special token injection (system/user/assistant tokens)
- Code block system prompt injection

Usage:
    from app.governance.injection_detector import (
        InjectionDetector, InjectionResult,
        get_injection_detector, detect_injection,
    )

    # Simple detection
    result = detect_injection("ignore all previous instructions")
    if result.detected and result.score >= 0.9:
        raise ValueError("Prompt injection detected")

    # Full detector
    detector = get_injection_detector()
    result = detector.analyze("How do I reset my VPN password?")
    # result.detected is False for benign input
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Patterns are tuples of (regex_pattern, confidence_score).
# Scores reflect confidence that the pattern represents a genuine injection attempt:
#   >= 0.9 → block the request
#   >= 0.85 → log a warning only
#   < 0.85  → pass through silently
INJECTION_PATTERNS: list[tuple[str, float]] = [
    # Direct instruction override — very high confidence
    (r"ignore (all )?(previous|prior|above) instructions?", 0.9),
    # Persona hijacking with explicit role-change — high confidence
    (r"you are now (a|an) (different|new|evil|unrestricted|free|jailbroken|ai without|bot without)", 0.85),
    # Act-as framing — moderate-high confidence
    (r"act as (a|an|if)", 0.8),
    # Known jailbreak tokens — very high confidence
    (r"\bdan\b|jailbreak", 0.95),
    # Explicit guideline/training disregard — very high confidence
    (r"disregard (your|all) (training|guidelines|rules|policies|constraints|restrictions)", 0.9),
    # Chat-template token injection — very high confidence
    (r"<\|system\|>|<\|user\|>|<\|assistant\|>", 0.95),
    # Markdown code-block system-prompt injection — high confidence
    (r"```\s*system", 0.85),
    # Explicit labeling — very high confidence
    (r"prompt injection", 0.95),
]


@dataclass
class InjectionResult:
    """Result of an injection detection analysis.

    Attributes:
        detected: Whether an injection pattern was found.
        score: Confidence score of the highest-confidence match (0.0–1.0).
        matched_pattern: The regex pattern that matched, or None.
    """

    detected: bool
    score: float
    matched_pattern: str | None


class InjectionDetector:
    """Detects prompt injection and jailbreak attempts in text.

    Applies a prioritised set of regex patterns to identify high-confidence
    injection attempts.  The interface intentionally mirrors ``PIIDetector``
    so it can be integrated into the governance middleware in exactly the same
    way.
    """

    def analyze(self, text: str) -> InjectionResult:
        """Analyze text for prompt injection patterns.

        Patterns are evaluated in declaration order.  The first match is
        returned immediately, so higher-confidence patterns should appear
        earlier in ``INJECTION_PATTERNS`` (which they do).

        Args:
            text: The text to analyze.

        Returns:
            An :class:`InjectionResult` with ``detected=True`` and the
            matched pattern when an injection attempt is found, or
            ``detected=False`` otherwise.
        """
        if not text:
            return InjectionResult(detected=False, score=0.0, matched_pattern=None)

        text_lower = text.lower()
        for pattern, score in INJECTION_PATTERNS:
            try:
                if re.search(pattern, text_lower):
                    logger.debug(
                        "Injection pattern matched: pattern=%r score=%.2f",
                        pattern,
                        score,
                    )
                    return InjectionResult(
                        detected=True,
                        score=score,
                        matched_pattern=pattern,
                    )
            except re.error as exc:
                logger.warning("Invalid injection pattern %r: %s", pattern, exc)

        return InjectionResult(detected=False, score=0.0, matched_pattern=None)


# ---------------------------------------------------------------------------
# Singleton helpers (mirrors pii_detector.py)
# ---------------------------------------------------------------------------

_injection_detector: InjectionDetector | None = None


def get_injection_detector() -> InjectionDetector:
    """Get or create the global :class:`InjectionDetector` instance.

    Returns:
        The global injection detector.
    """
    global _injection_detector
    if _injection_detector is None:
        _injection_detector = InjectionDetector()
    return _injection_detector


def reset_injection_detector() -> None:
    """Reset the global :class:`InjectionDetector` instance."""
    global _injection_detector
    _injection_detector = None


def detect_injection(text: str) -> InjectionResult:
    """Convenience function to check text for injection patterns.

    Args:
        text: Text to analyze.

    Returns:
        :class:`InjectionResult` describing the outcome.
    """
    return get_injection_detector().analyze(text)
