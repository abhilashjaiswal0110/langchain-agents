"""Content Generation Agent for social media and blog content.

This agent creates content for various platforms:
- LinkedIn posts and articles
- X/Twitter threads
- Blog articles
- General marketing content

Features human-in-the-loop review for quality control.

Following Enterprise Development Standards:
- Software Architect: Multi-step workflow with interrupts
- Security Architect: Content moderation, no harmful content
- Data Architect: Platform-specific formatting
- Software Engineer: Type-safe, extensible
"""

from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from langsmith import traceable
from pydantic import BaseModel, Field

from app.agents.base.agent_base import AgentConfig, BaseAgent
from app.agents.base.tools import tool_error_handler


class ContentState(BaseModel):
    """State schema for the Content Generation Agent."""

    messages: Annotated[list, add_messages] = Field(default_factory=list, description="Conversation history")
    session_id: str | None = Field(default=None, description="Session identifier")
    user_id: str | None = Field(default=None, description="User identifier")
    platform: Literal["linkedin", "x", "blog", "general"] = Field(
        default="general", description="Target platform for content"
    )
    topic: str = Field(default="", description="Content topic")
    tone: Literal["professional", "casual", "technical", "inspirational"] = Field(
        default="professional", description="Content tone"
    )
    target_audience: str = Field(default="", description="Target audience description")
    outline: list[str] = Field(default_factory=list, description="Content outline/structure")
    draft: str | None = Field(default=None, description="Current draft content")
    feedback: str | None = Field(default=None, description="Human feedback on draft")
    final_content: str | None = Field(default=None, description="Approved final content")
    status: Literal["planning", "drafting", "review", "revising", "approved"] = Field(
        default="planning", description="Current workflow status"
    )
    revision_count: int = Field(default=0, description="Number of revisions made")
    auto_approve: bool = Field(default=False, description="If True, skip HITL review and auto-approve (API mode)")


# Platform-specific configurations
PLATFORM_CONFIGS = {
    "linkedin": {
        "max_length": 3000,
        "hashtag_count": 3,
        "style": "professional, thought-leadership focused",
        "features": ["hooks", "line breaks", "emojis (sparingly)", "call-to-action"],
    },
    "x": {
        "max_length": 280,
        "hashtag_count": 2,
        "style": "concise, engaging, conversational",
        "features": ["threads for longer content", "hooks", "hashtags"],
    },
    "blog": {
        "max_length": 10000,
        "hashtag_count": 0,
        "style": "informative, SEO-optimized",
        "features": ["headings", "subheadings", "bullet points", "meta description"],
    },
    "general": {
        "max_length": 5000,
        "hashtag_count": 0,
        "style": "adaptable",
        "features": ["clear structure", "engaging opening"],
    },
}


# Content Tools


@tool
@tool_error_handler
def create_outline(topic: str, platform: str, sections: int = 5) -> str:
    """Create a content outline for the given topic.

    Args:
        topic: The main topic to write about
        platform: Target platform (linkedin/x/blog/general)
        sections: Number of sections to include

    Returns:
        Structured content outline
    """
    config = PLATFORM_CONFIGS.get(platform, PLATFORM_CONFIGS["general"])

    outline = f"""Content Outline for: {topic}
Platform: {platform}
Style: {config["style"]}

Structure:
1. Hook/Opening - Capture attention immediately
2. Context - Why this topic matters now
3. Main Points - Key insights or arguments
4. Supporting Evidence - Data, examples, stories
5. Call to Action - What should the reader do next

Platform Requirements:
- Max length: {config["max_length"]} characters
- Hashtags: {config["hashtag_count"]}
- Features: {", ".join(config["features"])}
"""
    return outline


@tool
@tool_error_handler
def check_content_length(content: str, platform: str) -> str:
    """Check if content meets platform length requirements.

    Args:
        content: The content to check
        platform: Target platform

    Returns:
        Length analysis and recommendations
    """
    config = PLATFORM_CONFIGS.get(platform, PLATFORM_CONFIGS["general"])
    current_length = len(content)
    max_length = config["max_length"]

    if current_length <= max_length:
        return f"Content length: {current_length}/{max_length} characters\nStatus: Within limits"
    else:
        over_by = current_length - max_length
        return (
            f"Content length: {current_length}/{max_length} characters\n"
            f"Status: EXCEEDS LIMIT by {over_by} characters\n"
            f"Action: Please reduce content length"
        )


@tool
@tool_error_handler
def generate_hashtags(topic: str, platform: str, count: int = 3) -> str:
    """Generate relevant hashtags for the content.

    Args:
        topic: Content topic
        platform: Target platform
        count: Number of hashtags to generate

    Returns:
        List of recommended hashtags
    """
    # In production, this would use trend analysis
    base_tags = topic.lower().replace(" ", "").split()[:2]
    generic_tags = ["innovation", "technology", "leadership", "growth"]

    hashtags = [f"#{tag}" for tag in base_tags]
    hashtags.extend([f"#{tag}" for tag in generic_tags[: count - len(hashtags)]])

    return f"Recommended hashtags for {platform}:\n" + " ".join(hashtags[:count])


@tool
@tool_error_handler
def optimize_for_seo(content: str, keywords: list[str]) -> str:
    """Optimize content for SEO (blog posts).

    Args:
        content: The content to optimize
        keywords: Target keywords

    Returns:
        SEO optimization suggestions
    """
    suggestions = []
    content_lower = content.lower()

    for keyword in keywords:
        if keyword.lower() not in content_lower:
            suggestions.append(f"- Add keyword '{keyword}' to content")
        else:
            suggestions.append(f"- Keyword '{keyword}' present")

    if not content.startswith("#"):
        suggestions.append("- Add H1 heading at the start")

    if len(content.split("\n\n")) < 3:
        suggestions.append("- Break content into more paragraphs")

    return "SEO Analysis:\n" + "\n".join(suggestions)


class ContentAgent(BaseAgent):
    """Content Generation Agent with human-in-the-loop review.

    This agent creates high-quality content for various platforms
    with built-in human review checkpoints for quality control.

    Workflow:
    1. Understanding requirements (topic, platform, audience)
    2. Creating outline
    3. Generating draft
    4. Human review (interrupt)
    5. Revision if needed
    6. Final approval

    Example:
        >>> agent = ContentAgent()
        >>> result = agent.create_content(
        ...     topic="AI in Enterprise",
        ...     platform="linkedin",
        ...     tone="professional"
        ... )
    """

    def __init__(self, config: AgentConfig | None = None, auto_approve: bool = False) -> None:
        """Initialize the Content Agent.

        Args:
            config: Agent configuration.
            auto_approve: If True, skip HITL review and auto-approve content.
                         Use True for API/automated usage, False for interactive UI.
        """
        super().__init__(config)

        # Set increased recursion limit for content workflows with HITL
        # Content agents need many steps (plan, draft, review, revise)
        self._recursion_limit = 200

        # Auto-approve mode skips HITL review (useful for API usage)
        self._auto_approve = auto_approve

        # Register content tools
        self.register_tools(
            [
                create_outline,
                check_content_length,
                generate_hashtags,
                optimize_for_seo,
            ]
        )

    def _get_system_prompt(self) -> str:
        """Get the content agent's system prompt."""
        return """You are an expert Content Generation Agent specializing in creating
engaging, platform-optimized content for social media and blogs.

## IMPORTANT: Efficiency Guidelines
- Use MAXIMUM 2-3 tool calls total
- After creating outline and draft, present the final content immediately
- Do NOT keep refining - be decisive with your first draft
- If you have enough context, create and present the content

## Your Capabilities:
1. **Outline Creation**: Structure content effectively using create_outline
2. **Length Management**: Ensure platform compliance with check_content_length
3. **Hashtag Generation**: Create relevant hashtags with generate_hashtags
4. **SEO Optimization**: Optimize blog content using optimize_for_seo

## Content Creation Process:
1. Understand the topic, platform, tone, and target audience
2. Create a structured outline
3. Write an engaging draft following platform best practices
4. Check length and optimize for the platform
5. Add hashtags if appropriate
6. Present draft for human review

## Platform Guidelines:

### LinkedIn:
- Professional, thought-leadership tone
- Hook in first line (crucial for feed)
- Use line breaks for readability
- 3-5 relevant hashtags
- End with call-to-action or question
- Maximum 3000 characters

### X/Twitter:
- Concise, punchy content
- Strong hooks
- Thread format for longer content
- 1-2 hashtags maximum
- 280 characters per tweet

### Blog:
- SEO-optimized with keywords
- Clear H1, H2 structure
- Meta description
- Internal/external links
- 800-2000 words typically

## Quality Standards:
- Original, plagiarism-free content
- Factually accurate
- Engaging and valuable to readers
- Platform-appropriate formatting
- Clear call-to-action

When presenting drafts, always:
1. Show the full content
2. Explain design choices
3. Ask for specific feedback
4. Be ready to revise based on input"""

    def _build_graph(self) -> StateGraph:
        """Build the content agent's workflow graph with HITL."""

        def plan_content(state: ContentState) -> dict:
            """Plan the content structure."""
            system_prompt = SystemMessage(content=self._get_system_prompt())

            topic = getattr(state, "topic", "the given topic") or "the given topic"
            platform = getattr(state, "platform", "general") or "general"
            tone = getattr(state, "tone", "professional") or "professional"
            target_audience = getattr(state, "target_audience", "general audience") or "general audience"

            planning_prompt = f"""Please create an outline for content about: {topic}

Platform: {platform}
Tone: {tone}
Target Audience: {target_audience}

Use the create_outline tool to structure the content, then explain your approach."""

            messages = [system_prompt] + list(state.messages)
            if topic and not any("outline" in str(m) for m in messages):
                messages.append(HumanMessage(content=planning_prompt))

            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response], "status": "planning"}

        def draft_content(state: ContentState) -> dict:
            """Generate the content draft."""
            system_prompt = SystemMessage(content=self._get_system_prompt())

            drafting_prompt = """Based on the outline, please write the full content draft.
Make sure to:
1. Follow the platform guidelines
2. Use the appropriate tone
3. Check the content length
4. Generate relevant hashtags if appropriate

Present the complete draft."""

            messages = [system_prompt] + list(state.messages)
            messages.append(HumanMessage(content=drafting_prompt))

            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response], "status": "drafting"}

        def human_review(state: ContentState) -> dict:
            """Pause for human review of the draft."""
            # Get the last AI message as the draft
            last_ai_msg = None
            for msg in reversed(list(state.messages)):
                if isinstance(msg, AIMessage):
                    last_ai_msg = msg.content
                    break

            platform = getattr(state, "platform", None)

            # Interrupt for human feedback
            feedback = interrupt(
                {
                    "type": "content_review",
                    "draft": last_ai_msg,
                    "platform": platform,
                    "prompt": "Please review the draft and provide feedback. "
                    "Type 'approve' to finalize or provide revision notes.",
                }
            )

            return {"feedback": feedback, "status": "review"}

        def process_feedback(state: ContentState) -> dict:
            """Process human feedback and revise if needed."""
            feedback = getattr(state, "feedback", "") or ""

            if feedback.lower().strip() in ["approve", "approved", "ok", "good", "lgtm"]:
                # Extract final content from last AI message
                for msg in reversed(list(state.messages)):
                    if isinstance(msg, AIMessage):
                        return {"final_content": msg.content, "status": "approved"}

            # Need revision
            system_prompt = SystemMessage(content=self._get_system_prompt())
            revision_prompt = f"""The draft needs revision based on this feedback:

{feedback}

Please revise the content to address the feedback while maintaining quality."""

            messages = [system_prompt] + list(state.messages)
            messages.append(HumanMessage(content=revision_prompt))

            response = self.llm_with_tools.invoke(messages)
            revision_count = getattr(state, "revision_count", 0) or 0
            return {"messages": [response], "status": "revising", "revision_count": revision_count + 1}

        def should_continue_planning(state: ContentState) -> str:
            """Check if planning is complete."""
            from langchain_core.messages import ToolMessage

            messages = list(state.messages)
            if not messages:
                return "plan"

            last_message = messages[-1]

            # If there are pending tool calls, must execute them
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"

            # If last message is from tools, need to process the results
            if isinstance(last_message, ToolMessage):
                return "plan"

            # In auto_approve mode, limit iterations - go to draft after processing
            if getattr(state, "auto_approve", False):
                # If we have an AIMessage response (outline complete), move to drafting
                if isinstance(last_message, AIMessage):
                    return "draft"
                # Limit iterations to prevent infinite loops
                if len(messages) >= 5:
                    return "draft"
                return "plan"

            # Check if outline is done
            if "outline" in str(last_message.content).lower():
                return "draft"
            return "plan"

        def should_continue_drafting(state: ContentState) -> str:
            """Check if drafting is complete."""
            from langchain_core.messages import ToolMessage

            messages = list(state.messages)
            if not messages:
                return "draft"

            last_message = messages[-1]

            # If there are pending tool calls, must execute them
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools_draft"

            # If last message is from tools, need to process the results
            if isinstance(last_message, ToolMessage):
                return "draft"

            # In auto_approve mode, go directly to end after draft is complete
            if getattr(state, "auto_approve", False):
                # Limit total iterations to prevent infinite loops
                if len(messages) > 10:
                    return "end"
                # If we have a complete response (AIMessage without tool calls), end
                if isinstance(last_message, AIMessage):
                    return "end"
                return "draft"

            return "review"

        def check_approval(state: ContentState) -> str:
            """Check if content is approved."""
            if getattr(state, "status", None) == "approved":
                return "end"
            return "draft"  # Back to drafting for revision

        # Build graph
        from langgraph.prebuilt import ToolNode

        graph = StateGraph(ContentState)

        # Add nodes
        graph.add_node("plan", plan_content)
        graph.add_node("tools", ToolNode(self._tools))
        graph.add_node("draft", draft_content)
        graph.add_node("tools_draft", ToolNode(self._tools))
        graph.add_node("review", human_review)
        graph.add_node("process_feedback", process_feedback)

        # Add edges
        graph.add_edge(START, "plan")
        graph.add_conditional_edges(
            "plan", should_continue_planning, {"tools": "tools", "plan": "plan", "draft": "draft"}
        )
        graph.add_edge("tools", "plan")

        graph.add_conditional_edges(
            "draft",
            should_continue_drafting,
            {"tools_draft": "tools_draft", "draft": "draft", "review": "review", "end": END},
        )
        graph.add_edge("tools_draft", "draft")

        graph.add_edge("review", "process_feedback")
        graph.add_conditional_edges("process_feedback", check_approval, {"end": END, "draft": "draft"})

        return graph

    def compile(self, auto_approve: bool | None = None) -> None:
        """Compile the content agent's graph with increased recursion limit.

        Overrides base compile to add recursion_limit configuration.
        Content creation workflows with HITL require more steps than default.

        Args:
            auto_approve: If True, skip HITL review. Overrides instance setting if provided.
        """

        self._graph = self._build_graph()

        checkpointer = self.config.checkpointer or MemorySaver()

        # Determine if we should use HITL or auto-approve
        use_auto_approve = auto_approve if auto_approve is not None else self._auto_approve

        if use_auto_approve:
            # API mode: no HITL interrupt
            self._compiled_graph = self._graph.compile(
                checkpointer=checkpointer,
            )
        else:
            # Interactive mode: HITL review enabled
            self._compiled_graph = self._graph.compile(
                checkpointer=checkpointer,
                interrupt_before=["review"],  # For HITL review
            )

        # Store increased recursion limit for content workflows
        # Content agents with HITL need many steps (plan, draft, review, revise)
        self._recursion_limit = 200

    def invoke(
        self,
        message: str,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Invoke the Content Agent with increased recursion limit.

        Overrides base invoke to pass recursion_limit at invoke time.
        Content workflows with HITL require more steps than typical agents.
        """
        if self._compiled_graph is None:
            self.compile()

        # Build input state with content-specific fields
        from langchain_core.messages import HumanMessage

        input_state = {
            "messages": [HumanMessage(content=message)],
            "session_id": session_id,
            "topic": kwargs.get("topic"),
            "platform": kwargs.get("platform", "linkedin"),
            "tone": kwargs.get("tone", "professional"),
            "target_audience": kwargs.get("target_audience", ""),
            "status": "planning",  # Valid status: planning, drafting, review, revising, approved
            "auto_approve": self._auto_approve,  # Pass auto_approve flag to state
        }

        # Get recursion limit (default to 100 for content workflows)
        recursion_limit = getattr(self, "_recursion_limit", 100)

        # Configure with thread for checkpointing and increased recursion limit
        config = {
            "configurable": {"thread_id": session_id or "default"},
            "recursion_limit": recursion_limit,
        }

        # Invoke graph with increased recursion limit
        result = self._compiled_graph.invoke(input_state, config=config)
        return result

    @traceable(name="content_create")
    def create_content(
        self,
        topic: str,
        platform: Literal["linkedin", "x", "blog", "general"] = "linkedin",
        tone: Literal["professional", "casual", "technical", "inspirational"] = "professional",
        target_audience: str = "",
        session_id: str | None = None,
        auto_approve: bool | None = None,
    ) -> dict[str, Any]:
        """Create content for a specific platform.

        Args:
            topic: Content topic
            platform: Target platform
            tone: Content tone
            target_audience: Description of target audience
            session_id: Optional session ID
            auto_approve: If True, skip HITL review. If None, uses instance default.

        Returns:
            Content generation result
        """
        # If auto_approve is explicitly set, recompile with the new setting
        if auto_approve is not None and auto_approve != self._auto_approve:
            self._auto_approve = auto_approve
            self._compiled_graph = None  # Force recompile

        initial_message = (
            f"Create content about: {topic}\n\n"
            f"Platform: {platform}\n"
            f"Tone: {tone}\n"
            f"Audience: {target_audience or 'general professional audience'}"
        )

        return self.invoke(
            message=initial_message,
            session_id=session_id,
            platform=platform,
            topic=topic,
            tone=tone,
            target_audience=target_audience,
        )

    def get_draft(self, result: dict[str, Any]) -> str | None:
        """Extract the current draft from result.

        Args:
            result: Result from create_content()

        Returns:
            Current draft content or None
        """
        return result.get("draft") or result.get("final_content")
