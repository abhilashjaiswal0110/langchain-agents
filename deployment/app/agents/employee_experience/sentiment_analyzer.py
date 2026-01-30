"""Employee Sentiment and Burnout Risk Analysis.

Provides sentiment detection and burnout risk assessment for employee conversations.
Uses pattern matching and keyword analysis for real-time detection.

For production use, consider integrating with:
- Claude API for advanced sentiment analysis
- ML models trained on employee conversation data
- Integration with engagement platforms (Qualtrics, CultureAmp, Glint)
"""

from typing import Literal


# =============================================================================
# Sentiment Analysis Patterns
# =============================================================================

# Sentiment keyword patterns
POSITIVE_KEYWORDS = [
    # Strong positive
    "excited", "thrilled", "love", "amazing", "excellent", "outstanding", "fantastic", "wonderful",
    "grateful", "thankful", "appreciate", "happy", "joy", "delighted", "pleased",
    # Moderate positive
    "good", "great", "helpful", "satisfied", "content", "comfortable", "confident",
    "optimistic", "hopeful", "motivated", "engaged", "inspired",
]

NEGATIVE_KEYWORDS = [
    # Strong negative
    "frustrated", "angry", "furious", "hate", "terrible", "awful", "horrible", "miserable",
    "overwhelmed", "exhausted", "burned out", "burnt out", "stressed", "anxious",
    "depressed", "hopeless", "helpless", "desperate",
    # Moderate negative
    "worried", "concerned", "disappointed", "upset", "sad", "unhappy", "dissatisfied",
    "confused", "lost", "stuck", "struggling", "difficult", "hard", "challenging",
]

STRESS_INDICATORS = [
    "too much work", "can't keep up", "falling behind", "no time", "too busy",
    "working late", "working weekends", "no breaks", "constant deadlines",
    "never enough time", "always rushing", "back-to-back meetings",
    "drowning", "swamped", "buried", "slammed",
]

BURNOUT_INDICATORS = [
    "burned out", "burnt out", "exhausted", "drained", "depleted",
    "can't do this anymore", "want to quit", "looking for other jobs",
    "not motivated", "don't care", "going through the motions",
    "disconnected", "isolated", "alone", "unsupported",
]

DISENGAGEMENT_INDICATORS = [
    "not excited", "lost interest", "don't see the point",
    "why bother", "doesn't matter", "nobody cares",
    "checking out", "counting days", "just a job",
    "no growth", "stuck", "going nowhere",
]

CONFLICT_INDICATORS = [
    "conflict with", "disagreement with", "problem with", "issue with",
    "tension", "difficult relationship", "not getting along",
    "micromanaging", "not listening", "not supported by",
    "unfair", "discriminated", "harassed", "bullied",
]


# =============================================================================
# Sentiment Analysis Functions
# =============================================================================


def analyze_employee_sentiment(message: str) -> dict:
    """Analyze sentiment of employee message.

    Args:
        message: Employee message text.

    Returns:
        Dictionary with sentiment score, label, and indicators.
    """
    message_lower = message.lower()

    # Count positive and negative keywords
    positive_count = sum(1 for keyword in POSITIVE_KEYWORDS if keyword in message_lower)
    negative_count = sum(1 for keyword in NEGATIVE_KEYWORDS if keyword in message_lower)

    # Count stress and burnout indicators
    stress_count = sum(1 for indicator in STRESS_INDICATORS if indicator in message_lower)
    burnout_count = sum(1 for indicator in BURNOUT_INDICATORS if indicator in message_lower)
    disengagement_count = sum(1 for indicator in DISENGAGEMENT_INDICATORS if indicator in message_lower)
    conflict_count = sum(1 for indicator in CONFLICT_INDICATORS if indicator in message_lower)

    # Calculate raw sentiment score
    total_keywords = positive_count + negative_count
    if total_keywords == 0:
        # No clear sentiment keywords - analyze other factors
        if stress_count + burnout_count + disengagement_count + conflict_count > 0:
            raw_score = -0.3  # Lean negative if stress indicators present
        else:
            raw_score = 0.0  # Neutral
    else:
        raw_score = (positive_count - negative_count) / total_keywords

    # Adjust for stress and burnout (strong negative signals)
    if burnout_count > 0:
        raw_score = min(raw_score - 0.5, -0.6)  # Strong negative adjustment
    elif stress_count >= 2:
        raw_score = min(raw_score - 0.3, -0.4)  # Moderate negative adjustment

    # Adjust for disengagement
    if disengagement_count > 0:
        raw_score = min(raw_score - 0.2, -0.3)

    # Adjust for conflict
    if conflict_count > 0:
        raw_score = min(raw_score - 0.2, -0.3)

    # Normalize score to -1.0 to 1.0 range
    sentiment_score = max(-1.0, min(1.0, raw_score))

    # Determine sentiment label
    if sentiment_score > 0.3:
        sentiment_label = "positive"
    elif sentiment_score < -0.3:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"

    # Identify specific indicators
    indicators = []
    if stress_count > 0:
        indicators.append("stress")
    if burnout_count > 0:
        indicators.append("burnout_risk")
    if disengagement_count > 0:
        indicators.append("disengagement")
    if conflict_count > 0:
        indicators.append("conflict")
    if positive_count > negative_count:
        indicators.append("positive_tone")

    return {
        "score": round(sentiment_score, 2),
        "label": sentiment_label,
        "confidence": min(0.95, 0.5 + (abs(sentiment_score) * 0.45)),  # Higher confidence with stronger sentiment
        "indicators": indicators,
        "keyword_counts": {
            "positive": positive_count,
            "negative": negative_count,
            "stress": stress_count,
            "burnout": burnout_count,
            "disengagement": disengagement_count,
            "conflict": conflict_count,
        },
    }


def assess_burnout_risk(messages: list[str]) -> dict:
    """Assess burnout risk based on conversation history.

    Multi-factor assessment:
    - Sentiment trend over time
    - Stress indicators frequency
    - Work-life balance signals
    - Engagement level
    - Help-seeking behavior

    Args:
        messages: List of recent messages from employee.

    Returns:
        Dictionary with risk level, score, and contributing factors.
    """
    if not messages:
        return {
            "risk_level": "unknown",
            "risk_score": 0,
            "factors": [],
            "recommendation": "Insufficient data for assessment",
        }

    # Analyze each message
    sentiment_scores = []
    total_stress = 0
    total_burnout = 0
    total_disengagement = 0
    total_conflict = 0

    for message in messages:
        analysis = analyze_employee_sentiment(message)
        sentiment_scores.append(analysis["score"])
        total_stress += analysis["keyword_counts"]["stress"]
        total_burnout += analysis["keyword_counts"]["burnout"]
        total_disengagement += analysis["keyword_counts"]["disengagement"]
        total_conflict += analysis["keyword_counts"]["conflict"]

    # Calculate risk factors
    risk_factors = []
    risk_score = 0

    # Factor 1: Overall sentiment trend (weight: 0.25)
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    if avg_sentiment < -0.4:
        risk_score += 3
        risk_factors.append("persistent_negative_sentiment")
    elif avg_sentiment < -0.2:
        risk_score += 2
        risk_factors.append("negative_sentiment")

    # Factor 2: Sentiment declining trend (weight: 0.20)
    if len(sentiment_scores) >= 3:
        recent_sentiment = sum(sentiment_scores[-3:]) / 3
        # Guard against division by zero when len == 3
        if len(sentiment_scores) > 3:
            earlier_sentiment = sum(sentiment_scores[:-3]) / (len(sentiment_scores) - 3)
            if recent_sentiment < earlier_sentiment - 0.3:
                risk_score += 2
                risk_factors.append("declining_sentiment")

    # Factor 3: Stress indicators (weight: 0.20)
    stress_frequency = total_stress / len(messages)
    if stress_frequency >= 1.5:
        risk_score += 3
        risk_factors.append("high_stress")
    elif stress_frequency >= 0.75:
        risk_score += 2
        risk_factors.append("moderate_stress")

    # Factor 4: Burnout indicators (weight: 0.20)
    if total_burnout > 0:
        risk_score += 3
        risk_factors.append("explicit_burnout_signals")

    # Factor 5: Disengagement (weight: 0.10)
    if total_disengagement >= 2:
        risk_score += 2
        risk_factors.append("disengagement")
    elif total_disengagement >= 1:
        risk_score += 1
        risk_factors.append("some_disengagement")

    # Factor 6: Conflict indicators (weight: 0.05)
    if total_conflict >= 2:
        risk_score += 1
        risk_factors.append("workplace_conflict")

    # Determine risk level
    if risk_score >= 7:
        risk_level = "high"
        recommendation = "URGENT: Immediate wellbeing check-in recommended. Consider escalation to HRBP or EAP referral."
    elif risk_score >= 4:
        risk_level = "medium"
        recommendation = "MODERATE: Proactive wellbeing resources should be offered. Schedule check-in within 1 week."
    else:
        risk_level = "low"
        recommendation = "LOW: Monitor sentiment in future interactions. Provide standard wellbeing resources."

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "max_score": 10,
        "factors": risk_factors,
        "recommendation": recommendation,
        "sentiment_trend": {
            "average": round(avg_sentiment, 2),
            "recent": round(sum(sentiment_scores[-3:]) / 3, 2) if len(sentiment_scores) >= 3 else round(avg_sentiment, 2),
            "trend": "declining" if len(sentiment_scores) > 3 and (sum(sentiment_scores[-3:]) / 3) < (sum(sentiment_scores[:-3]) / (len(sentiment_scores) - 3) - 0.2) else "stable",
        },
        "indicator_counts": {
            "stress": total_stress,
            "burnout": total_burnout,
            "disengagement": total_disengagement,
            "conflict": total_conflict,
        },
    }


def detect_escalation_triggers(message: str) -> dict:
    """Detect if message contains escalation triggers requiring immediate attention.

    Args:
        message: Employee message text.

    Returns:
        Dictionary with escalation required flag and trigger types.
    """
    message_lower = message.lower()

    # Critical escalation keywords
    HARASSMENT_KEYWORDS = [
        "harassment", "harassed", "harassing", "sexual harassment",
        "unwanted advances", "inappropriate touching", "sexually explicit",
    ]

    DISCRIMINATION_KEYWORDS = [
        "discrimination", "discriminated", "racist", "racism", "sexist", "sexism",
        "ageism", "homophobia", "transphobia", "disability discrimination",
    ]

    SAFETY_KEYWORDS = [
        "unsafe", "dangerous", "threat", "threatened", "afraid", "scared",
        "violence", "violent", "assault", "physical altercation",
    ]

    LEGAL_KEYWORDS = [
        "illegal", "unlawful", "lawsuit", "lawyer", "attorney",
        "legal action", "sue", "suing", "court",
    ]

    RETALIATION_KEYWORDS = [
        "retaliation", "retaliated", "retaliating", "punished for reporting",
        "fired for speaking up", "demoted for complaining",
    ]

    SELF_HARM_KEYWORDS = [
        "want to die", "kill myself", "suicide", "suicidal",
        "end it all", "not worth living", "self-harm",
    ]

    # Check for triggers
    triggers = []
    escalation_required = False

    if any(keyword in message_lower for keyword in HARASSMENT_KEYWORDS):
        triggers.append("harassment")
        escalation_required = True

    if any(keyword in message_lower for keyword in DISCRIMINATION_KEYWORDS):
        triggers.append("discrimination")
        escalation_required = True

    if any(keyword in message_lower for keyword in SAFETY_KEYWORDS):
        triggers.append("safety_concern")
        escalation_required = True

    if any(keyword in message_lower for keyword in LEGAL_KEYWORDS):
        triggers.append("legal_matter")
        escalation_required = True

    if any(keyword in message_lower for keyword in RETALIATION_KEYWORDS):
        triggers.append("retaliation")
        escalation_required = True

    if any(keyword in message_lower for keyword in SELF_HARM_KEYWORDS):
        triggers.append("crisis_self_harm")
        escalation_required = True

    return {
        "escalation_required": escalation_required,
        "triggers": triggers,
        "urgency": "critical" if escalation_required else "normal",
        "recommended_action": (
            "IMMEDIATE ESCALATION TO HRBP" if escalation_required
            else "Standard agent response"
        ),
    }


# =============================================================================
# Utility Functions
# =============================================================================


def get_sentiment_emoji(sentiment_score: float) -> str:
    """Get emoji representation of sentiment score.

    Args:
        sentiment_score: Sentiment score from -1.0 to 1.0.

    Returns:
        Emoji string representing sentiment.
    """
    if sentiment_score > 0.6:
        return "😊"  # Very positive
    elif sentiment_score > 0.3:
        return "🙂"  # Positive
    elif sentiment_score > -0.3:
        return "😐"  # Neutral
    elif sentiment_score > -0.6:
        return "😟"  # Negative
    else:
        return "😫"  # Very negative


def get_risk_emoji(risk_level: Literal["low", "medium", "high"]) -> str:
    """Get emoji representation of burnout risk level.

    Args:
        risk_level: Risk level (low, medium, high).

    Returns:
        Emoji string representing risk level.
    """
    risk_emojis = {
        "low": "🟢",
        "medium": "🟡",
        "high": "🔴",
    }
    return risk_emojis.get(risk_level, "⚪")


def format_sentiment_report(sentiment_result: dict, burnout_result: dict | None = None) -> str:
    """Format sentiment and burnout analysis into human-readable report.

    Args:
        sentiment_result: Result from analyze_employee_sentiment().
        burnout_result: Optional result from assess_burnout_risk().

    Returns:
        Formatted report string.
    """
    report = ["**Sentiment Analysis Report:**\n"]

    # Sentiment section
    emoji = get_sentiment_emoji(sentiment_result["score"])
    report.append(f"{emoji} **Overall Sentiment:** {sentiment_result['label'].title()} (score: {sentiment_result['score']})")
    report.append(f"**Confidence:** {sentiment_result['confidence']:.0%}")

    if sentiment_result["indicators"]:
        report.append(f"\n**Indicators Detected:** {', '.join(sentiment_result['indicators'])}")

    # Burnout section (if available)
    if burnout_result:
        report.append("\n\n**Burnout Risk Assessment:**\n")
        risk_emoji = get_risk_emoji(burnout_result["risk_level"])
        report.append(f"{risk_emoji} **Risk Level:** {burnout_result['risk_level'].upper()} (score: {burnout_result['risk_score']}/{burnout_result['max_score']})")
        report.append(f"\n**Recommendation:** {burnout_result['recommendation']}")

        if burnout_result["factors"]:
            report.append(f"\n**Contributing Factors:**")
            for factor in burnout_result["factors"]:
                report.append(f"  - {factor.replace('_', ' ').title()}")

    return "\n".join(report)


# =============================================================================
# For Production Use - LLM-Based Sentiment Analysis
# =============================================================================


def analyze_sentiment_with_llm(message: str, llm_client=None) -> dict:
    """Analyze sentiment using LLM for more nuanced understanding.

    This is a placeholder for production implementation using Claude or another LLM.
    The LLM can provide more context-aware sentiment analysis, detecting sarcasm,
    implicit stress, and emotional nuances that keyword matching might miss.

    Args:
        message: Employee message text.
        llm_client: LLM client (e.g., Anthropic Claude API client).

    Returns:
        Dictionary with sentiment analysis from LLM.
    """
    if llm_client is None:
        # Fall back to keyword-based analysis
        return analyze_employee_sentiment(message)

    # Placeholder for LLM-based analysis
    # In production, you would:
    # 1. Send message to Claude with sentiment analysis prompt
    # 2. Parse structured output with sentiment score, indicators, and reasoning
    # 3. Return result in same format as keyword-based analysis

    # This would call Claude API with a constructed prompt and parse the response.
    # For now, we fall back to keyword analysis to keep behavior consistent.
    return analyze_employee_sentiment(message)
