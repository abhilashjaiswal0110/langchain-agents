"""Employee Experience & HR Support Deep Agent Package.

Comprehensive HR support agent providing:
- HR policy Q&A and benefits information
- Career development and skills gap analysis
- Performance review preparation
- Employee sentiment detection and burnout risk assessment
- Wellbeing resources and support programs
- Learning and development recommendations
- Compensation insights and guidance
- Employee engagement and pulse surveys
- Escalation orchestration to HR business partners
"""

from app.agents.employee_experience.employee_experience_agent import (
    EmployeeExperienceAgent,
    EmployeeExperienceState,
    get_graph,
)
from app.agents.employee_experience.sentiment_analyzer import (
    analyze_employee_sentiment,
    assess_burnout_risk,
    detect_escalation_triggers,
    format_sentiment_report,
    get_risk_emoji,
    get_sentiment_emoji,
)

__all__ = [
    # Main agent
    "EmployeeExperienceAgent",
    "EmployeeExperienceState",
    "get_graph",
    # Sentiment analysis
    "analyze_employee_sentiment",
    "assess_burnout_risk",
    "detect_escalation_triggers",
    "format_sentiment_report",
    "get_sentiment_emoji",
    "get_risk_emoji",
]
