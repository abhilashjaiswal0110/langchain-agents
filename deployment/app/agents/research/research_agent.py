"""AI Research Agent for comprehensive information gathering and analysis.

This agent performs multi-step research workflows:
1. Query understanding and planning
2. Web search and information gathering
3. Source analysis and fact extraction
4. Synthesis and summary generation

Following Enterprise Development Standards:
- Software Architect: Multi-step workflow with state management
- Security Architect: No PII exposure, sanitized outputs
- Data Architect: Structured source tracking
- Software Engineer: Type-safe, well-documented
"""

import os
from typing import Annotated, Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langsmith import traceable
from pydantic import BaseModel, Field

from app.agents.base.agent_base import AgentConfig, BaseAgent
from app.agents.base.tools import tool_error_handler


class ResearchState(BaseModel):
    """State schema for the Research Agent.

    Tracks the research workflow including sources,
    findings, and synthesis.
    """

    messages: Annotated[list, add_messages] = Field(default_factory=list, description="Conversation history")
    session_id: str | None = Field(default=None, description="Session identifier")
    user_id: str | None = Field(default=None, description="User identifier")
    query: str = Field(default="", description="Current research query")
    research_plan: list[str] = Field(default_factory=list, description="Planned research steps")
    sources: list[dict[str, Any]] = Field(default_factory=list, description="Collected sources with metadata")
    findings: list[str] = Field(default_factory=list, description="Key findings from research")
    summary: str | None = Field(default=None, description="Final research summary")
    status: Literal["planning", "researching", "analyzing", "complete"] = Field(
        default="planning", description="Current research status"
    )


# Research Tools


@tool
@tool_error_handler
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information on a topic.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        Formatted search results with titles, snippets, and URLs
    """
    # Try to use Tavily if available
    tavily_key = os.getenv("TAVILY_API_KEY")

    if tavily_key:
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=tavily_key)
            response = client.search(query, max_results=max_results)

            results = []
            for item in response.get("results", []):
                results.append(
                    f"**{item.get('title', 'No title')}**\n"
                    f"URL: {item.get('url', 'N/A')}\n"
                    f"Summary: {item.get('content', 'No content')[:500]}\n"
                )

            return "\n---\n".join(results) if results else "No results found."

        except Exception as e:
            return f"Search error: {e}"

    # Fallback: simulated search results for demo
    return f"""Simulated search results for: "{query}"

**Result 1: Overview of {query}**
URL: https://example.com/overview
Summary: This is a comprehensive overview of the topic covering key aspects and recent developments.

**Result 2: {query} - Latest Research**
URL: https://research.example.com/latest
Summary: Recent academic research and industry reports on the subject with data from 2024.

**Result 3: Practical Guide to {query}**
URL: https://guide.example.com/practical
Summary: A hands-on guide with best practices, case studies, and implementation strategies.

Note: For real search results, configure TAVILY_API_KEY in your environment."""


@tool
@tool_error_handler
def extract_key_points(text: str) -> str:
    """Extract key points from a text passage.

    Args:
        text: Text to analyze

    Returns:
        Bullet-pointed list of key points
    """
    # This would use NLP in production; simple extraction for demo
    sentences = text.split(". ")
    key_points = []

    for i, sentence in enumerate(sentences[:5]):
        if len(sentence) > 20:
            key_points.append(f"- {sentence.strip()}")

    return "\n".join(key_points) if key_points else "No key points extracted."


@tool
@tool_error_handler
def add_source(title: str, url: str, content_summary: str, reliability: str = "medium") -> str:
    """Add a source to the research collection.

    Args:
        title: Source title
        url: Source URL
        content_summary: Brief summary of the content
        reliability: Source reliability rating (low/medium/high)

    Returns:
        Confirmation message
    """
    return (
        f"Source added:\n"
        f"- Title: {title}\n"
        f"- URL: {url}\n"
        f"- Reliability: {reliability}\n"
        f"- Summary: {content_summary[:200]}..."
    )


@tool
@tool_error_handler
def synthesize_findings(findings: list[str], format_type: str = "summary") -> str:
    """Synthesize multiple findings into a coherent output.

    Args:
        findings: List of research findings
        format_type: Output format (summary/bullets/detailed)

    Returns:
        Synthesized content
    """
    if not findings:
        return "No findings to synthesize."

    if format_type == "bullets":
        return "Key Findings:\n" + "\n".join(f"- {f}" for f in findings)
    elif format_type == "detailed":
        return "Detailed Analysis:\n\n" + "\n\n".join(f"Finding {i + 1}:\n{f}" for i, f in enumerate(findings))
    else:
        return "Research Summary:\n" + " ".join(findings)


class ResearchAgent(BaseAgent):
    """AI Research Agent for comprehensive information gathering.

    This agent excels at:
    - Breaking down complex research questions
    - Gathering information from multiple sources
    - Analyzing and synthesizing findings
    - Producing well-structured research summaries

    Example:
        >>> agent = ResearchAgent()
        >>> result = agent.invoke("What are the latest trends in AI agents?")
        >>> print(agent.get_last_response(result))
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize the Research Agent."""
        super().__init__(config)

        # Register research tools
        self.register_tools(
            [
                web_search,
                extract_key_points,
                add_source,
                synthesize_findings,
            ]
        )

    def compile(self) -> None:
        """Compile the research agent's graph with increased recursion limit.

        Overrides base compile to add recursion_limit configuration.
        Research workflows require more steps than the default 25.
        """

        self._graph = self._build_graph()

        checkpointer = self.config.checkpointer or MemorySaver()
        self._compiled_graph = self._graph.compile(checkpointer=checkpointer)

        # Store increased recursion limit for deep research workflows
        # Research agents often need many tool calls (search, analyze, synthesize)
        # Set high limit to handle complex multi-step research workflows
        self._recursion_limit = 200

    def _get_system_prompt(self) -> str:
        """Get the research agent's system prompt."""
        return """You are an expert AI Research Agent. Research topics and provide clear summaries.

## IMPORTANT: Efficiency Guidelines
- Use MAXIMUM 2-3 web searches per query
- After gathering information, provide your final answer immediately
- Do NOT keep searching for more information - be decisive
- If you have enough information, synthesize and respond

## Tools Available:
1. **web_search**: Search the web (use sparingly - max 2-3 searches)
2. **synthesize_findings**: Combine findings into a summary

## Research Process:
1. Perform 1-2 targeted web searches
2. Synthesize the findings you have
3. Provide your final answer with sources

## Output Format:
- **Summary**: Brief overview (2-3 sentences)
- **Key Points**: Main findings (3-5 bullet points)
- **Sources**: URLs of sources used

Be concise and efficient. Provide your answer quickly."""

    def _build_graph(self) -> StateGraph:
        """Build the research agent's workflow graph."""

        def call_model(state: ResearchState) -> dict:
            """Call the LLM to process the current state."""
            system_prompt = SystemMessage(content=self._get_system_prompt())
            # Use attribute access for Pydantic model
            messages = [system_prompt] + list(state.messages)
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def should_continue(state: ResearchState) -> str:
            """Determine if we should continue with tools or end."""
            # Use attribute access for Pydantic model
            messages = list(state.messages)
            last_message = messages[-1] if messages else None

            if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "end"

        # Build graph
        graph = StateGraph(ResearchState)

        # Add nodes
        graph.add_node("agent", call_model)
        graph.add_node("tools", ToolNode(self._tools))

        # Add edges
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
        graph.add_edge("tools", "agent")

        return graph

    def invoke(
        self,
        message: str,
        session_id: str | None = None,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Invoke the Research Agent with increased recursion limit.

        Overrides base invoke to pass recursion_limit at invoke time.
        Research workflows require more steps than typical agents.
        """
        if self._compiled_graph is None:
            self.compile()

        # Build input state
        input_state = {
            "messages": [HumanMessage(content=message)],
            "session_id": session_id,
            "user_id": user_id,
            **kwargs,
        }

        # Get recursion limit (default to 100 for research workflows)
        recursion_limit = getattr(self, "_recursion_limit", 100)

        # Configure with thread for checkpointing and increased recursion limit
        config = {
            "configurable": {"thread_id": session_id or "default"},
            "recursion_limit": recursion_limit,
        }

        # Invoke graph with increased recursion limit
        result = self._compiled_graph.invoke(input_state, config=config)
        return result

    @traceable(name="research_invoke")
    def research(
        self,
        query: str,
        session_id: str | None = None,
        depth: Literal["quick", "standard", "comprehensive"] = "standard",
    ) -> dict[str, Any]:
        """Perform research on a topic.

        Args:
            query: Research question or topic
            session_id: Optional session ID for continuity
            depth: Research depth level

        Returns:
            Research results including findings and summary
        """
        # Adjust prompt based on depth
        depth_instructions = {
            "quick": "Provide a brief overview with 2-3 key points.",
            "standard": "Provide a thorough analysis with multiple sources.",
            "comprehensive": "Provide an exhaustive analysis covering all aspects, "
            "with detailed source citations and cross-referencing.",
        }

        enhanced_query = f"{query}\n\nResearch depth: {depth}. {depth_instructions[depth]}"

        # Use our custom invoke with increased recursion limit
        return self.invoke(
            message=enhanced_query,
            session_id=session_id,
            query=query,
            status="researching",
        )

    def get_sources(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract sources from research result.

        Args:
            result: Result from research()

        Returns:
            List of source dictionaries
        """
        return result.get("sources", [])
