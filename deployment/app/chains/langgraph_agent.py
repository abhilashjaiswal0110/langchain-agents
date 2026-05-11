"""LangGraph Agent Integration.

This module provides LangGraph-based agents that can be used alongside
LangChain chains. Supports Azure OpenAI, OpenAI, and Anthropic models
with LangSmith tracing enabled.
"""

import os
from collections.abc import Sequence
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from app.agents.base.llm_factory import get_llm

# ============================================================================
# Custom Tools for the Agent
# ============================================================================


@tool
def web_search(query: str) -> str:
    """Search the web for information.

    Args:
        query: The search query.

    Returns:
        Search results as a string.
    """
    # Simulated web search - replace with Tavily if API key available
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults

            tavily = TavilySearchResults(max_results=3)
            results = tavily.invoke(query)
            return str(results)
        except Exception as e:
            return f"Search error: {e}"

    # Fallback simulated results
    return f"Simulated search results for: {query}. (Set TAVILY_API_KEY for real search)"


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate.

    Returns:
        The result of the calculation.
    """
    try:
        allowed_chars = set("0123456789+-*/.(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def get_system_info() -> str:
    """Get current system information.

    Returns:
        System information including date/time.
    """
    from datetime import datetime

    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


# ============================================================================
# Agent State and Context
# ============================================================================


class AgentContext(TypedDict):
    """Context for selecting model provider."""

    model: Literal["azure_openai", "anthropic", "openai"]


class AgentState(TypedDict):
    """State for the LangGraph agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


# ============================================================================
# Agent Graph Construction
# ============================================================================


def create_langgraph_agent(
    model_provider: str = "auto",
) -> StateGraph | None:
    """Create a LangGraph agent with tools.

    Uses the centralized LLM factory which supports:
    - Azure OpenAI (primary for production)
    - OpenAI (disabled by default)
    - Anthropic (fallback)

    Args:
        model_provider: The model provider to use ('azure_openai', 'openai', 'anthropic', or 'auto').

    Returns:
        Compiled LangGraph agent or None if no providers available.
    """
    # Initialize tools
    tools = [web_search, calculator, get_system_info]

    # Use factory to get LLM (handles Azure OpenAI, OpenAI, and Anthropic)
    try:
        provider_arg = model_provider if model_provider != "auto" else None
        llm = get_llm(provider=provider_arg, temperature=0)
        model = llm.bind_tools(tools)
    except ValueError:
        # No LLM provider configured
        return None

    # Define routing function
    def should_continue(state: AgentState) -> str:
        """Determine whether to continue with tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        return "continue"

    # Define model call function
    def call_model(state: AgentState) -> dict:
        """Call the model with current state."""
        messages = state["messages"]
        response = model.invoke(messages)
        return {"messages": [response]}

    # Build the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )

    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")

    # Compile and return
    return workflow.compile()


# ============================================================================
# Runnable Wrapper for LangServe
# ============================================================================


class LangGraphAgentRunnable:
    """Wrapper to make LangGraph agent compatible with LangServe."""

    def __init__(self, model_provider: str = "auto") -> None:
        """Initialize the LangGraph agent runnable.

        Args:
            model_provider: The model provider to use.
        """
        self.agent = create_langgraph_agent(model_provider)
        self.model_provider = model_provider

    def invoke(self, input_data: dict) -> dict:
        """Invoke the agent with input.

        Args:
            input_data: Input containing 'input' key with the user message.

        Returns:
            Response dict with 'output' key.
        """
        if self.agent is None:
            return {
                "output": "Error: No LLM provider configured. "
                "Configure Azure OpenAI, OpenAI (with OPENAI_ENABLED=true), or Anthropic.",
            }

        user_input = input_data.get("input", "")
        messages = [HumanMessage(content=user_input)]

        result = self.agent.invoke({"messages": messages})

        # Extract final response
        final_message = result["messages"][-1]
        return {"output": final_message.content}

    async def ainvoke(self, input_data: dict) -> dict:
        """Async invoke the agent.

        Args:
            input_data: Input containing 'input' key with the user message.

        Returns:
            Response dict with 'output' key.
        """
        if self.agent is None:
            return {
                "output": "Error: No LLM provider configured. "
                "Configure Azure OpenAI, OpenAI (with OPENAI_ENABLED=true), or Anthropic.",
            }

        user_input = input_data.get("input", "")
        messages = [HumanMessage(content=user_input)]

        result = await self.agent.ainvoke({"messages": messages})

        final_message = result["messages"][-1]
        return {"output": final_message.content}


# Create default instance
langgraph_agent = LangGraphAgentRunnable(model_provider="auto")
