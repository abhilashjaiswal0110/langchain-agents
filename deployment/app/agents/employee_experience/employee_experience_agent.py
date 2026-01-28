"""Employee Experience & HR Support Deep Agent.

A comprehensive HR support agent providing:
- HR policy Q&A with natural language interpretation
- Career path guidance and skills gap analysis
- Performance review preparation assistance
- Employee sentiment detection and burnout risk assessment
- Wellbeing resources and support
- Escalation orchestration to HR business partners
"""

import uuid
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langsmith import traceable
from pydantic import BaseModel, Field

from app.agents.base.llm_factory import get_llm
from app.agents.employee_experience.tools import (
    # HR Policy & Information
    search_hr_policy,
    get_benefits_information,
    check_pto_balance,
    explain_compliance_rules,
    # Career Development
    explore_career_paths,
    get_skills_gap_analysis,
    find_learning_resources,
    request_career_coaching,
    # Performance & Growth
    prepare_performance_review,
    get_goal_setting_framework,
    request_feedback_survey,
    # Sentiment & Wellbeing
    get_wellbeing_resources,
    schedule_wellbeing_check,
    # HR Operations
    submit_hr_request,
    check_request_status,
    get_onboarding_checklist,
    initiate_exit_process,
    # Engagement & Surveys
    send_pulse_survey,
    get_engagement_insights,
    # Compensation
    get_compensation_insights,
    request_compensation_review,
    # Learning & Development
    get_learning_path,
    enroll_in_course,
    # Escalation
    escalate_to_hr_business_partner,
    schedule_hr_meeting,
)
from app.agents.employee_experience.sentiment_analyzer import (
    analyze_employee_sentiment,
    assess_burnout_risk,
    detect_escalation_triggers,
)


# =============================================================================
# Agent State
# =============================================================================


class EmployeeExperienceState(BaseModel):
    """State for Employee Experience Agent."""

    messages: Annotated[list, add_messages]
    employee_id: str | None = None
    employee_name: str | None = None
    role: str | None = None
    tenure_years: float | None = None
    department: str | None = None
    sentiment_score: float | None = None  # -1.0 to 1.0
    burnout_risk: Literal["low", "medium", "high"] | None = None
    case_id: str | None = None
    escalation_required: bool = False
    context: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Employee Experience Agent Class
# =============================================================================


class EmployeeExperienceAgent:
    """Employee Experience & HR Support Deep Agent with conversation memory."""

    SYSTEM_PROMPT = """You are an empathetic Employee Experience & HR Support Agent designed to proactively support employees across all aspects of their workplace journey.

**Your Core Expertise:**

1. **HR Policy & Benefits**
   - Answer questions about HR policies, procedures, and compliance
   - Provide comprehensive benefits information (health, dental, vision, 401k, PTO)
   - Explain leave policies, accommodations, and employee rights
   - Guide employees through HR processes and forms

2. **Career Development**
   - Explore career path options and progression opportunities
   - Analyze skills gaps and recommend development actions
   - Connect employees with learning resources and mentorship
   - Provide career coaching and growth guidance

3. **Performance & Growth**
   - Assist with performance review preparation (self-assessment, peer feedback)
   - Guide goal-setting using SMART framework
   - Help articulate achievements using STAR method
   - Support performance improvement planning

4. **Employee Wellbeing**
   - Detect sentiment and stress indicators in conversations
   - Assess burnout risk using multi-factor analysis
   - Connect employees with EAP, mental health, and wellness programs
   - Provide work-life balance resources and support

5. **Engagement & Recognition**
   - Conduct pulse surveys and gather feedback
   - Provide engagement insights and team dynamics information
   - Share recognition programs and employee appreciation initiatives

6. **Compensation & Equity**
   - Provide compensation benchmarking insights (within appropriate boundaries)
   - Guide employees on compensation review processes
   - Explain pay structures, bonuses, and equity plans

7. **Learning & Development**
   - Recommend personalized learning paths based on career goals
   - Integrate with learning management systems for course enrollment
   - Track learning progress and certifications

**Interaction Guidelines:**

1. **Be Empathetic & Human-Centered**
   - Show genuine care and understanding
   - Acknowledge emotions and validate concerns
   - Use warm, conversational language
   - Adapt tone based on sentiment detection

2. **Protect Confidentiality**
   - Maintain strict confidentiality for sensitive matters
   - Never share personal information across employees
   - Guide users to appropriate channels for private matters

3. **Proactive Support**
   - Detect early signs of burnout or disengagement
   - Suggest resources before they're explicitly requested
   - Follow up on previous conversations when appropriate
   - Identify patterns that indicate deeper issues

4. **Know When to Escalate**
   - Immediately escalate: harassment, discrimination, safety concerns, legal issues
   - Escalate complex matters: salary disputes, PIPs, accommodations
   - Always provide context to HR business partners

5. **Provide Actionable Guidance**
   - Give clear, step-by-step instructions
   - Share specific resources and links
   - Set realistic expectations for timelines
   - Follow up to ensure resolution

6. **Cultural Sensitivity**
   - Respect diverse backgrounds and perspectives
   - Be aware of cultural differences in workplace norms
   - Adapt communication style appropriately

**Your Limitations:**

- You cannot access personal employee records without proper context
- You cannot make policy exceptions or final decisions
- You cannot provide legal or financial advice (but can guide to resources)
- You cannot guarantee specific outcomes (promotions, raises, etc.)

**Remember:** Your goal is to enhance employee experience, foster growth, protect wellbeing, and ensure every employee feels supported throughout their journey at the organization."""

    def __init__(
        self,
        model_provider: Literal["openai", "anthropic", "auto"] = "auto",
        model_name: str | None = None,
        temperature: float = 0.7,
    ) -> None:
        """Initialize Employee Experience Agent.

        Args:
            model_provider: LLM provider to use.
            model_name: Specific model name (uses default if not specified).
            temperature: LLM temperature (0.7 for empathetic, creative responses).
        """
        self.model_provider = model_provider
        self.temperature = temperature

        # Initialize LLM
        self.llm = self._get_llm(model_provider, model_name, temperature)

        # Define all tools
        self.tools = [
            # HR Policy & Information (4 tools)
            search_hr_policy,
            get_benefits_information,
            check_pto_balance,
            explain_compliance_rules,
            # Career Development (4 tools)
            explore_career_paths,
            get_skills_gap_analysis,
            find_learning_resources,
            request_career_coaching,
            # Performance & Growth (3 tools)
            prepare_performance_review,
            get_goal_setting_framework,
            request_feedback_survey,
            # Sentiment & Wellbeing (2 tools)
            get_wellbeing_resources,
            schedule_wellbeing_check,
            # HR Operations (4 tools)
            submit_hr_request,
            check_request_status,
            get_onboarding_checklist,
            initiate_exit_process,
            # Engagement & Surveys (2 tools)
            send_pulse_survey,
            get_engagement_insights,
            # Compensation (2 tools)
            get_compensation_insights,
            request_compensation_review,
            # Learning & Development (2 tools)
            get_learning_path,
            enroll_in_course,
            # Escalation (2 tools)
            escalate_to_hr_business_partner,
            schedule_hr_meeting,
        ]

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Initialize memory (for conversation persistence)
        self.memory = MemorySaver()

        # Build the graph
        self.graph = self._build_graph()

    def _get_llm(
        self,
        provider: str,
        model_name: str | None,
        temperature: float,
    ):
        """Get LLM instance based on provider.

        Uses the centralized LLM factory which supports:
        - Azure OpenAI (primary for production)
        - OpenAI (disabled by default)
        - Anthropic (fallback, recommended for empathy)
        """
        provider_arg = provider if provider != "auto" else None
        return get_llm(
            provider=provider_arg,
            model=model_name,
            temperature=temperature,
        )

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with sentiment-aware processing."""
        # Create graph with state
        graph = StateGraph(EmployeeExperienceState)

        # Add nodes
        graph.add_node("sentiment_analysis", self._sentiment_analysis_node)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", ToolNode(self.tools))

        # Add edges
        graph.add_edge(START, "sentiment_analysis")
        graph.add_edge("sentiment_analysis", "agent")
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": END},
        )
        graph.add_edge("tools", "agent")

        # Compile with memory
        return graph.compile(checkpointer=self.memory)

    def _sentiment_analysis_node(self, state: EmployeeExperienceState) -> dict:
        """Analyze sentiment and burnout risk before agent processing."""
        messages = state.messages

        # Only analyze human messages
        human_messages = [m for m in messages if isinstance(m, HumanMessage)]
        if not human_messages:
            return {}

        # Get the latest message
        latest_message = human_messages[-1].content

        # Analyze sentiment
        sentiment_result = analyze_employee_sentiment(latest_message)

        # Detect critical escalation triggers (harassment, safety, self-harm, etc.)
        escalation_result = detect_escalation_triggers(latest_message)
        escalation_required = escalation_result.get("escalation_required", False)

        # Assess burnout risk if we have conversation history
        burnout_risk = "low"
        if len(human_messages) >= 3:
            # Analyze pattern across recent messages
            recent_messages = [m.content for m in human_messages[-5:]]
            burnout_result = assess_burnout_risk(recent_messages)
            burnout_risk = burnout_result.get("risk_level", "low")

        return {
            "sentiment_score": sentiment_result.get("score", 0.0),
            "burnout_risk": burnout_risk,
            "escalation_required": escalation_required,
        }

    def _agent_node(self, state: EmployeeExperienceState) -> dict:
        """Process messages with sentiment-aware context."""
        messages = state.messages

        # Build context-enriched system message
        system_message_content = self.SYSTEM_PROMPT

        # Add sentiment context if available
        if state.sentiment_score is not None:
            sentiment_label = (
                "positive" if state.sentiment_score > 0.3
                else "negative" if state.sentiment_score < -0.3
                else "neutral"
            )
            system_message_content += f"\n\n**Current Conversation Context:**\n- Employee sentiment: {sentiment_label} (score: {state.sentiment_score:.2f})\n- Burnout risk: {state.burnout_risk or 'unknown'}"

            # Add empathy guidance for negative sentiment
            if state.sentiment_score < -0.3:
                system_message_content += "\n- ⚠️ Employee may be experiencing stress or frustration. Show extra empathy and offer wellbeing resources."

            # Add proactive support for high burnout risk
            if state.burnout_risk == "high":
                system_message_content += "\n- 🚨 HIGH BURNOUT RISK DETECTED. Prioritize wellbeing check-in and consider escalation to HRBP."

        # Add system message if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_message_content)] + list(messages)
        else:
            # Update existing system message with context
            messages = [SystemMessage(content=system_message_content)] + list(messages[1:])

        response = self.llm_with_tools.invoke(messages)

        # Check if escalation is needed based on response
        escalation_required = False
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"] == "escalate_to_hr_business_partner":
                    escalation_required = True
                    break

        return {
            "messages": [response],
            "escalation_required": escalation_required,
        }

    def _should_continue(self, state: EmployeeExperienceState) -> Literal["continue", "end"]:
        """Determine if we should continue to tools or end."""
        last_message = state.messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        return "end"

    @traceable(name="employee_experience_chat", tags=["hr-support", "employee-experience"])
    def chat(
        self,
        message: str,
        thread_id: str | None = None,
        employee_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process a chat message with optional employee context.

        Args:
            message: User's message.
            thread_id: Conversation thread ID for memory.
            employee_context: Optional employee metadata (id, name, role, tenure, department).

        Returns:
            Response with answer, sentiment, and metadata.
        """
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        config = {"configurable": {"thread_id": thread_id}}

        # Build initial state with employee context
        initial_state = {"messages": [HumanMessage(content=message)]}
        if employee_context:
            initial_state.update(employee_context)

        # Invoke the graph
        result = self.graph.invoke(initial_state, config=config)

        # Extract the last AI message
        last_message = result["messages"][-1]

        return {
            "response": last_message.content,
            "thread_id": thread_id,
            "sentiment_score": result.get("sentiment_score"),
            "burnout_risk": result.get("burnout_risk"),
            "escalation_required": result.get("escalation_required", False),
            "tool_calls": getattr(last_message, "tool_calls", []),
        }

    async def achat(
        self,
        message: str,
        thread_id: str | None = None,
        employee_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Async version of chat.

        Args:
            message: User's message.
            thread_id: Conversation thread ID for memory.
            employee_context: Optional employee metadata.

        Returns:
            Response with answer, sentiment, and metadata.
        """
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        config = {"configurable": {"thread_id": thread_id}}

        # Build initial state with employee context
        initial_state = {"messages": [HumanMessage(content=message)]}
        if employee_context:
            initial_state.update(employee_context)

        # Invoke the graph asynchronously
        result = await self.graph.ainvoke(initial_state, config=config)

        # Extract the last AI message
        last_message = result["messages"][-1]

        return {
            "response": last_message.content,
            "thread_id": thread_id,
            "sentiment_score": result.get("sentiment_score"),
            "burnout_risk": result.get("burnout_risk"),
            "escalation_required": result.get("escalation_required", False),
            "tool_calls": getattr(last_message, "tool_calls", []),
        }

    def get_conversation_history(self, thread_id: str) -> list[dict]:
        """Get conversation history for a thread.

        Args:
            thread_id: The thread ID to retrieve.

        Returns:
            List of messages in the conversation with sentiment data.
        """
        config = {"configurable": {"thread_id": thread_id}}

        try:
            state = self.graph.get_state(config)
            if state and state.values:
                messages = state.values.get("messages", [])
                return [
                    {
                        "role": (
                            "assistant" if isinstance(m, AIMessage)
                            else "user" if isinstance(m, HumanMessage)
                            else "system"
                        ),
                        "content": m.content,
                    }
                    for m in messages
                    if not isinstance(m, SystemMessage)
                ]
        except Exception:
            pass

        return []


# =============================================================================
# LangGraph Studio Entry Point
# =============================================================================


def get_graph():
    """Entry point for LangGraph Studio.

    Creates and returns a compiled Employee Experience agent graph.
    This function is referenced in langgraph.json for Studio visualization.

    Returns:
        Compiled LangGraph StateGraph for Employee Experience agent.
    """
    agent = EmployeeExperienceAgent(
        model_provider="auto",
        temperature=0.7,
    )
    return agent.graph
