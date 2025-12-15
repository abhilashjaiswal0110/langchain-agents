"""LangChain Agent with tools."""

from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g., '2 + 2', '10 * 5').
    """
    try:
        # Safe evaluation of mathematical expressions
        allowed_chars = set("0123456789+-*/.(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information.

    Args:
        query: The search query.
    """
    # Simulated knowledge base responses
    knowledge = {
        "langchain": "LangChain is a framework for building LLM-powered applications.",
        "agents": "Agents use LLMs to determine which actions to take and in what order.",
        "rag": "RAG combines retrieval systems with generative models for better responses.",
        "default": "I don't have specific information about that topic.",
    }

    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return knowledge["default"]


# Define available tools
tools = [get_current_time, calculate, search_knowledge_base]

# System prompt for the agent
SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.
Use the available tools to help answer questions.
Always provide clear and helpful responses."""

# Initialize LLM with tool calling capability
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create the agent using LangGraph's create_react_agent
agent_executor = create_react_agent(
    llm,
    tools,
    prompt=SYSTEM_PROMPT,
)
