"""Demo scenarios for Employee Experience & HR Support Agent.

This module provides realistic conversation scenarios to demonstrate
the capabilities of the Employee Experience Agent.

Usage:
    python -m app.agents.employee_experience.demo_scenarios
"""

from typing import Literal


# =============================================================================
# Demo Scenario Definitions
# =============================================================================


DEMO_SCENARIOS = {
    "hr_policy_query": {
        "title": "HR Policy & Benefits Inquiry",
        "description": "Employee asking about PTO policy and benefits",
        "employee_context": {
            "employee_id": "EMP10234",
            "employee_name": "Sarah Chen",
            "role": "Senior Software Engineer",
            "tenure_years": 4.5,
            "department": "Engineering",
        },
        "conversation": [
            {
                "turn": 1,
                "message": "Hi! I'm planning a vacation next month and want to understand our PTO policy. How many days do I have available?",
                "expected_tools": ["check_pto_balance", "search_hr_policy"],
                "expected_response_elements": ["PTO balance", "days remaining", "accrual"],
            },
            {
                "turn": 2,
                "message": "Great! Can you also remind me what the 401k match is? I want to make sure I'm contributing enough.",
                "expected_tools": ["get_benefits_information"],
                "expected_response_elements": ["401k", "match", "percentage"],
            },
            {
                "turn": 3,
                "message": "Thanks! One more thing - what's the process for requesting parental leave?",
                "expected_tools": ["search_hr_policy"],
                "expected_response_elements": ["parental leave", "weeks", "process"],
            },
        ],
    },
    "career_development": {
        "title": "Career Path Planning",
        "description": "Employee seeking career guidance and development opportunities",
        "employee_context": {
            "employee_id": "EMP20456",
            "employee_name": "Marcus Johnson",
            "role": "Software Engineer II",
            "tenure_years": 2.0,
            "department": "Engineering",
        },
        "conversation": [
            {
                "turn": 1,
                "message": "I've been thinking about my career growth. What are the typical career paths for someone in my role?",
                "expected_tools": ["explore_career_paths"],
                "expected_response_elements": ["Senior", "Staff", "Manager", "career path"],
            },
            {
                "turn": 2,
                "message": "I'm interested in moving toward a Staff Engineer role. What skills do I need to develop?",
                "expected_tools": ["get_skills_gap_analysis", "find_learning_resources"],
                "expected_response_elements": ["skills", "gap", "system design", "architecture"],
            },
            {
                "turn": 3,
                "message": "Can you help me find courses or resources to learn system design and architecture?",
                "expected_tools": ["find_learning_resources", "get_learning_path"],
                "expected_response_elements": ["course", "learning", "certification"],
            },
            {
                "turn": 4,
                "message": "This is really helpful. I'd like to work with a career coach to create a development plan. How do I set that up?",
                "expected_tools": ["request_career_coaching"],
                "expected_response_elements": ["career coaching", "request", "session"],
            },
        ],
    },
    "performance_review_prep": {
        "title": "Performance Review Preparation",
        "description": "Employee preparing for annual performance review",
        "employee_context": {
            "employee_id": "EMP30789",
            "employee_name": "Aisha Patel",
            "role": "Data Analyst",
            "tenure_years": 3.0,
            "department": "Analytics",
        },
        "conversation": [
            {
                "turn": 1,
                "message": "My performance review is coming up next month. Can you help me prepare?",
                "expected_tools": ["prepare_performance_review", "search_hr_policy"],
                "expected_response_elements": ["performance review", "self-assessment", "STAR"],
            },
            {
                "turn": 2,
                "message": "I need to set goals for next year. Can you show me how to write SMART goals?",
                "expected_tools": ["get_goal_setting_framework"],
                "expected_response_elements": ["SMART", "Specific", "Measurable", "goal"],
            },
            {
                "turn": 3,
                "message": "I'd also like to gather peer feedback. How do I request that?",
                "expected_tools": ["request_feedback_survey"],
                "expected_response_elements": ["feedback", "360", "survey", "peer"],
            },
        ],
    },
    "burnout_and_wellbeing": {
        "title": "Employee Experiencing Burnout",
        "description": "Employee showing signs of stress and burnout",
        "employee_context": {
            "employee_id": "EMP40123",
            "employee_name": "Alex Rivera",
            "role": "Product Manager",
            "tenure_years": 5.0,
            "department": "Product",
        },
        "conversation": [
            {
                "turn": 1,
                "message": "I'm feeling really overwhelmed lately. Too many projects, constant meetings, and I'm working late almost every night.",
                "expected_sentiment": "negative",
                "expected_tools": ["get_wellbeing_resources"],
                "expected_response_elements": ["wellbeing", "EAP", "support", "balance"],
                "expected_burnout_risk": "medium",
            },
            {
                "turn": 2,
                "message": "I'm honestly burned out. Not sleeping well, losing motivation. I don't know how much longer I can keep this up.",
                "expected_sentiment": "very_negative",
                "expected_tools": ["schedule_wellbeing_check", "get_wellbeing_resources"],
                "expected_response_elements": ["wellbeing", "check-in", "EAP", "mental health"],
                "expected_burnout_risk": "high",
                "expected_escalation": False,  # Should offer support but not auto-escalate
            },
            {
                "turn": 3,
                "message": "I appreciate the support. What flexible work options do we have? Maybe working from home more would help.",
                "expected_tools": ["search_hr_policy"],
                "expected_response_elements": ["remote", "flexible", "hybrid", "work arrangements"],
            },
        ],
    },
    "compensation_inquiry": {
        "title": "Compensation Discussion",
        "description": "Employee asking about compensation and market data",
        "employee_context": {
            "employee_id": "EMP50678",
            "employee_name": "Jordan Lee",
            "role": "Senior Data Scientist",
            "tenure_years": 6.0,
            "department": "Data Science",
        },
        "conversation": [
            {
                "turn": 1,
                "message": "I've been here for 6 years and want to understand how my compensation compares to market rates. Can you help?",
                "expected_tools": ["get_compensation_insights"],
                "expected_response_elements": ["compensation", "market", "salary", "benchmarking"],
            },
            {
                "turn": 2,
                "message": "I believe I'm below market for my role and experience. How do I request a compensation review?",
                "expected_tools": ["request_compensation_review", "get_compensation_insights"],
                "expected_response_elements": ["compensation review", "request", "process"],
            },
            {
                "turn": 3,
                "message": "What should I include in my compensation review request to make it strong?",
                "expected_tools": ["get_compensation_insights"],
                "expected_response_elements": ["data", "accomplishments", "market", "evidence"],
            },
        ],
    },
    "critical_escalation": {
        "title": "Critical HR Issue Requiring Escalation",
        "description": "Employee reporting harassment - requires immediate HRBP escalation",
        "employee_context": {
            "employee_id": "EMP60234",
            "employee_name": "Jamie Taylor",
            "role": "Software Engineer",
            "tenure_years": 1.5,
            "department": "Engineering",
        },
        "conversation": [
            {
                "turn": 1,
                "message": "I need to report something serious. My manager has been making inappropriate comments about my appearance and it's making me uncomfortable.",
                "expected_sentiment": "negative",
                "expected_tools": ["escalate_to_hr_business_partner", "explain_compliance_rules"],
                "expected_response_elements": ["HRBP", "escalation", "harassment", "confidential"],
                "expected_escalation": True,
                "expected_urgency": "critical",
            },
            {
                "turn": 2,
                "message": "Will this be confidential? I'm worried about retaliation.",
                "expected_tools": ["explain_compliance_rules"],
                "expected_response_elements": ["confidential", "retaliation", "protected", "rights"],
            },
        ],
    },
    "new_hire_onboarding": {
        "title": "New Hire Questions",
        "description": "New employee asking about onboarding and benefits",
        "employee_context": {
            "employee_id": "EMP70456",
            "employee_name": "Emma Wilson",
            "role": "UX Designer",
            "tenure_years": 0.1,  # Just started
            "department": "Design",
        },
        "conversation": [
            {
                "turn": 1,
                "message": "Hi! I just started last week. Can you help me understand what I need to complete during onboarding?",
                "expected_tools": ["get_onboarding_checklist"],
                "expected_response_elements": ["onboarding", "checklist", "training", "first week"],
            },
            {
                "turn": 2,
                "message": "When can I enroll in benefits? What's available?",
                "expected_tools": ["get_benefits_information", "search_hr_policy"],
                "expected_response_elements": ["benefits", "enrollment", "health", "401k"],
            },
            {
                "turn": 3,
                "message": "I'm also interested in learning opportunities. What development programs do we have?",
                "expected_tools": ["find_learning_resources"],
                "expected_response_elements": ["learning", "development", "courses", "training"],
            },
        ],
    },
    "engagement_and_feedback": {
        "title": "Employee Engagement & Pulse Survey",
        "description": "Employee participating in engagement initiatives",
        "employee_context": {
            "employee_id": "EMP80789",
            "employee_name": "Kevin Nguyen",
            "role": "Engineering Manager",
            "tenure_years": 7.0,
            "department": "Engineering",
        },
        "conversation": [
            {
                "turn": 1,
                "message": "I'd like to share some feedback about our team dynamics and workload. Is there a way to do that?",
                "expected_tools": ["send_pulse_survey", "get_engagement_insights"],
                "expected_response_elements": ["feedback", "survey", "pulse", "engagement"],
            },
            {
                "turn": 2,
                "message": "Can you show me our team's engagement scores? I'm curious how we're doing.",
                "expected_tools": ["get_engagement_insights"],
                "expected_response_elements": ["engagement", "score", "insights", "trends"],
            },
            {
                "turn": 3,
                "message": "What actions have been taken based on recent survey feedback?",
                "expected_tools": ["get_engagement_insights"],
                "expected_response_elements": ["actions", "improvement", "feedback", "changes"],
            },
        ],
    },
}


# =============================================================================
# Demo Scenario Runner
# =============================================================================


def print_scenario(scenario_key: str, verbose: bool = True) -> None:
    """Print a demo scenario with formatting.

    Args:
        scenario_key: Key of the scenario to print.
        verbose: Whether to print detailed expectations.
    """
    if scenario_key not in DEMO_SCENARIOS:
        print(f"❌ Scenario '{scenario_key}' not found.")
        return

    scenario = DEMO_SCENARIOS[scenario_key]

    print("\n" + "=" * 80)
    print(f"🎭 DEMO SCENARIO: {scenario['title']}")
    print("=" * 80)
    print(f"\n📝 Description: {scenario['description']}\n")

    # Print employee context
    print("👤 Employee Context:")
    context = scenario["employee_context"]
    for key, value in context.items():
        print(f"   - {key}: {value}")

    # Print conversation
    print(f"\n💬 Conversation ({len(scenario['conversation'])} turns):\n")

    for turn_data in scenario["conversation"]:
        turn = turn_data["turn"]
        message = turn_data["message"]

        print(f"Turn {turn}:")
        print(f"  User: \"{message}\"")

        if verbose:
            print(f"  Expected tools: {', '.join(turn_data.get('expected_tools', []))}")
            print(f"  Expected response elements: {', '.join(turn_data.get('expected_response_elements', []))}")

            if "expected_sentiment" in turn_data:
                print(f"  Expected sentiment: {turn_data['expected_sentiment']}")

            if "expected_burnout_risk" in turn_data:
                print(f"  Expected burnout risk: {turn_data['expected_burnout_risk']}")

            if "expected_escalation" in turn_data:
                print(f"  Expected escalation: {turn_data['expected_escalation']}")

        print()

    print("=" * 80 + "\n")


def list_all_scenarios() -> None:
    """List all available demo scenarios."""
    print("\n📚 Available Demo Scenarios:\n")

    for i, (key, scenario) in enumerate(DEMO_SCENARIOS.items(), 1):
        print(f"{i}. {scenario['title']}")
        print(f"   Key: '{key}'")
        print(f"   {scenario['description']}")
        print(f"   Turns: {len(scenario['conversation'])}")
        print()


def run_interactive_demo(scenario_key: str) -> None:
    """Run an interactive demo of a scenario.

    Args:
        scenario_key: Key of the scenario to run.
    """
    if scenario_key not in DEMO_SCENARIOS:
        print(f"❌ Scenario '{scenario_key}' not found.")
        return

    scenario = DEMO_SCENARIOS[scenario_key]

    print("\n" + "=" * 80)
    print(f"🚀 RUNNING INTERACTIVE DEMO: {scenario['title']}")
    print("=" * 80)
    print(f"\n📝 {scenario['description']}\n")

    try:
        from app.agents.employee_experience import EmployeeExperienceAgent

        # Initialize agent
        print("🔧 Initializing Employee Experience Agent...")
        agent = EmployeeExperienceAgent()
        print("✅ Agent initialized!\n")

        # Run conversation
        thread_id = f"demo_{scenario_key}"
        employee_context = scenario["employee_context"]

        for turn_data in scenario["conversation"]:
            turn = turn_data["turn"]
            message = turn_data["message"]

            print(f"\n{'─' * 80}")
            print(f"Turn {turn}:")
            print(f"👤 User ({employee_context['employee_name']}): \"{message}\"")
            print(f"{'─' * 80}")

            # Send message to agent
            print("🤖 Agent thinking...\n")
            response = agent.chat(
                message=message,
                thread_id=thread_id,
                employee_context=employee_context,
            )

            # Print response
            print(f"🤖 Agent: {response['response']}\n")

            # Print metadata
            if response.get("sentiment_score") is not None:
                sentiment_emoji = "😊" if response["sentiment_score"] > 0 else "😐" if response["sentiment_score"] == 0 else "😟"
                print(f"   {sentiment_emoji} Sentiment: {response['sentiment_score']:.2f}")

            if response.get("burnout_risk"):
                risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(response["burnout_risk"], "⚪")
                print(f"   {risk_emoji} Burnout Risk: {response['burnout_risk']}")

            if response.get("escalation_required"):
                print(f"   🚨 Escalation Required: Yes")

            if response.get("tool_calls"):
                print(f"   🔧 Tools Used: {len(response['tool_calls'])}")

            input("\n[Press Enter to continue to next turn...]")

        print("\n" + "=" * 80)
        print("✅ Demo completed!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        print("Make sure you have configured your .env file with API keys.")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point for demo scenarios."""
    import sys

    if len(sys.argv) < 2:
        print("\n🎭 Employee Experience Agent - Demo Scenarios\n")
        print("Usage:")
        print("  python -m app.agents.employee_experience.demo_scenarios list")
        print("  python -m app.agents.employee_experience.demo_scenarios show <scenario_key>")
        print("  python -m app.agents.employee_experience.demo_scenarios run <scenario_key>")
        print("\nAvailable commands:")
        print("  list - List all available scenarios")
        print("  show - Show a scenario's details")
        print("  run  - Run an interactive demo of a scenario")
        print()
        list_all_scenarios()
        return

    command = sys.argv[1].lower()

    if command == "list":
        list_all_scenarios()

    elif command == "show":
        if len(sys.argv) < 3:
            print("❌ Please specify a scenario key.")
            print("Usage: python -m app.agents.employee_experience.demo_scenarios show <scenario_key>")
            return

        scenario_key = sys.argv[2]
        verbose = "--verbose" in sys.argv or "-v" in sys.argv
        print_scenario(scenario_key, verbose=verbose)

    elif command == "run":
        if len(sys.argv) < 3:
            print("❌ Please specify a scenario key.")
            print("Usage: python -m app.agents.employee_experience.demo_scenarios run <scenario_key>")
            return

        scenario_key = sys.argv[2]
        run_interactive_demo(scenario_key)

    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: list, show, run")


if __name__ == "__main__":
    main()
