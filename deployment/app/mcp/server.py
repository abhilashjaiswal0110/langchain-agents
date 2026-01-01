"""MCP server exposing LangChain agents as tools.

This module provides an MCP (Model Context Protocol) server that exposes
enterprise agents as tools that can be used by MCP-compatible clients
like Claude Desktop, VS Code extensions, or other AI applications.

Usage:
    # Run standalone
    python -m app.mcp.server

    # Or import and integrate with FastAPI
    from app.mcp.server import get_mcp_server, mount_mcp_routes
"""

import asyncio
import os
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from app.mcp.gateway import MCPGateway, get_mcp_gateway


# Create MCP server instance
mcp = FastMCP(
    "LangChain-Agents",
    instructions="""
    This MCP server provides access to enterprise IT agents including:
    - Research Agent: Web research with source citations
    - ServiceNow Agent: IT service management (incidents, changes, CMDB)
    - Document Agent: Document generation (SOPs, policies, WLIs)
    - IT Support Agent: General IT helpdesk support
    - Code Assistant: Code review and generation

    Use these tools to help with IT operations, research, and document management.
    """,
)


# ============== Research Tools ==============


@mcp.tool()
async def research_topic(
    query: str,
    depth: int = 2,
    max_sources: int = 5,
    ctx: Context | None = None,
) -> str:
    """Research a topic using web search and analysis.

    Args:
        query: The research question or topic to investigate.
        depth: How deep to research (1=quick, 2=standard, 3=comprehensive).
        max_sources: Maximum number of sources to cite.
        ctx: MCP context (injected automatically).

    Returns:
        Research findings with source citations.
    """
    gateway = get_mcp_gateway()
    await gateway.check_tool_permission("research_topic", ctx)

    try:
        # Import agent lazily to avoid circular imports
        from app.agents import get_agent

        agent = get_agent("research")
        if not agent:
            return "Research agent is not available. Please check configuration."

        result = await agent.ainvoke({
            "input": query,
            "config": {
                "depth": depth,
                "max_sources": max_sources,
            },
        })

        # Extract response from LangGraph state
        return _extract_response(result)

    except Exception as e:
        return f"Research failed: {e}"


@mcp.tool()
async def quick_search(query: str, ctx: Context | None = None) -> str:
    """Perform a quick web search for simple queries.

    Args:
        query: Search query.
        ctx: MCP context.

    Returns:
        Search results summary.
    """
    return await research_topic(query, depth=1, max_sources=3, ctx=ctx)


# ============== ServiceNow Tools ==============


@mcp.tool()
async def create_incident(
    short_description: str,
    description: str,
    priority: str = "3",
    category: str = "inquiry",
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Create a ServiceNow incident ticket.

    Args:
        short_description: Brief summary of the issue.
        description: Detailed description of the problem.
        priority: Priority level (1=Critical, 2=High, 3=Medium, 4=Low).
        category: Incident category (e.g., "inquiry", "software", "hardware").
        ctx: MCP context.

    Returns:
        Created incident details including ticket number.
    """
    gateway = get_mcp_gateway()
    await gateway.check_tool_permission("create_incident", ctx)

    try:
        from app.mcp.servicenow_client import get_servicenow_client

        client = get_servicenow_client()
        incident = await client.create_incident(
            short_description=short_description,
            description=description,
            priority=priority,
            category=category,
        )
        return incident

    except Exception as e:
        return {"error": str(e), "status": "failed"}


@mcp.tool()
async def search_incidents(
    query: str,
    status: str | None = None,
    limit: int = 10,
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    """Search for ServiceNow incidents.

    Args:
        query: Search query (searches short_description and description).
        status: Filter by status (e.g., "new", "in_progress", "resolved").
        limit: Maximum number of results.
        ctx: MCP context.

    Returns:
        List of matching incidents.
    """
    gateway = get_mcp_gateway()
    await gateway.check_tool_permission("search_incidents", ctx)

    try:
        from app.mcp.servicenow_client import get_servicenow_client

        client = get_servicenow_client()
        incidents = await client.search_incidents(
            query=query,
            status=status,
            limit=limit,
        )
        return incidents

    except Exception as e:
        return [{"error": str(e), "status": "failed"}]


@mcp.tool()
async def get_incident(
    incident_number: str,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Get details of a specific ServiceNow incident.

    Args:
        incident_number: The incident number (e.g., "INC0010001").
        ctx: MCP context.

    Returns:
        Incident details.
    """
    gateway = get_mcp_gateway()
    await gateway.check_tool_permission("get_incident", ctx)

    try:
        from app.mcp.servicenow_client import get_servicenow_client

        client = get_servicenow_client()
        incident = await client.get_incident(incident_number)
        return incident

    except Exception as e:
        return {"error": str(e), "status": "failed"}


@mcp.tool()
async def update_incident(
    incident_number: str,
    work_notes: str | None = None,
    state: str | None = None,
    assigned_to: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Update a ServiceNow incident.

    Args:
        incident_number: The incident number to update.
        work_notes: Work notes to add.
        state: New state (e.g., "2" for In Progress, "6" for Resolved).
        assigned_to: User to assign the incident to.
        ctx: MCP context.

    Returns:
        Updated incident details.
    """
    gateway = get_mcp_gateway()
    await gateway.check_tool_permission("update_incident", ctx)

    try:
        from app.mcp.servicenow_client import get_servicenow_client

        client = get_servicenow_client()
        updates: dict[str, Any] = {}
        if work_notes:
            updates["work_notes"] = work_notes
        if state:
            updates["state"] = state
        if assigned_to:
            updates["assigned_to"] = assigned_to

        incident = await client.update_incident(incident_number, updates)
        return incident

    except Exception as e:
        return {"error": str(e), "status": "failed"}


@mcp.tool()
async def query_cmdb(
    ci_type: str,
    query: str,
    limit: int = 10,
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    """Query the ServiceNow CMDB for configuration items.

    Args:
        ci_type: Type of configuration item (e.g., "cmdb_ci_server", "cmdb_ci_computer").
        query: Search query for CI name or attributes.
        limit: Maximum results.
        ctx: MCP context.

    Returns:
        List of matching configuration items.
    """
    gateway = get_mcp_gateway()
    await gateway.check_tool_permission("query_cmdb", ctx)

    try:
        from app.mcp.servicenow_client import get_servicenow_client

        client = get_servicenow_client()
        items = await client.query_cmdb(ci_type, query, limit)
        return items

    except Exception as e:
        return [{"error": str(e), "status": "failed"}]


# ============== Document Tools ==============


@mcp.tool()
async def generate_document(
    document_type: str,
    title: str,
    content_brief: str,
    sections: list[str] | None = None,
    ctx: Context | None = None,
) -> str:
    """Generate a document using the Document Generation Agent.

    Args:
        document_type: Type of document ("sop", "policy", "wli", "runbook").
        title: Document title.
        content_brief: Brief description of what the document should contain.
        sections: Optional list of section titles to include.
        ctx: MCP context.

    Returns:
        Generated document content in markdown format.
    """
    gateway = get_mcp_gateway()
    await gateway.check_tool_permission("generate_document", ctx)

    try:
        from app.agents import get_agent

        agent = get_agent("document")
        if not agent:
            return "Document agent is not available."

        result = await agent.ainvoke({
            "input": content_brief,
            "document_type": document_type,
            "title": title,
            "sections": sections or [],
        })

        return _extract_response(result)

    except Exception as e:
        return f"Document generation failed: {e}"


@mcp.tool()
async def generate_sop(
    title: str,
    purpose: str,
    steps: list[str],
    ctx: Context | None = None,
) -> str:
    """Generate a Standard Operating Procedure (SOP).

    Args:
        title: SOP title.
        purpose: Purpose/objective of the procedure.
        steps: List of high-level steps to include.
        ctx: MCP context.

    Returns:
        Generated SOP document.
    """
    content_brief = f"Purpose: {purpose}\nKey steps: {', '.join(steps)}"
    return await generate_document(
        document_type="sop",
        title=title,
        content_brief=content_brief,
        sections=["Purpose", "Scope", "Procedure", "Responsibilities"],
        ctx=ctx,
    )


# ============== IT Support Tools ==============


@mcp.tool()
async def it_support_query(
    question: str,
    context: str | None = None,
    ctx: Context | None = None,
) -> str:
    """Ask the IT Support agent a question.

    Args:
        question: The IT support question.
        context: Optional additional context about the issue.
        ctx: MCP context.

    Returns:
        IT support response with recommendations.
    """
    gateway = get_mcp_gateway()
    await gateway.check_tool_permission("it_support_query", ctx)

    try:
        from app.agents import get_agent

        agent = get_agent("helpdesk")
        if not agent:
            return "IT Support agent is not available."

        full_query = question
        if context:
            full_query = f"{question}\n\nAdditional context: {context}"

        result = await agent.ainvoke({"input": full_query})
        return _extract_response(result)

    except Exception as e:
        return f"IT support query failed: {e}"


@mcp.tool()
async def troubleshoot_issue(
    symptoms: str,
    system: str | None = None,
    error_message: str | None = None,
    ctx: Context | None = None,
) -> str:
    """Get troubleshooting guidance for an IT issue.

    Args:
        symptoms: Description of the symptoms or problem.
        system: Affected system or application (optional).
        error_message: Any error message received (optional).
        ctx: MCP context.

    Returns:
        Troubleshooting steps and recommendations.
    """
    question = f"Troubleshoot: {symptoms}"
    context_parts = []
    if system:
        context_parts.append(f"System: {system}")
    if error_message:
        context_parts.append(f"Error: {error_message}")

    context = "\n".join(context_parts) if context_parts else None
    return await it_support_query(question, context, ctx)


# ============== Code Assistant Tools ==============


@mcp.tool()
async def review_code(
    code: str,
    language: str = "python",
    focus: str | None = None,
    ctx: Context | None = None,
) -> str:
    """Review code for issues and improvements.

    Args:
        code: The code to review.
        language: Programming language.
        focus: Specific area to focus on (e.g., "security", "performance").
        ctx: MCP context.

    Returns:
        Code review with suggestions.
    """
    gateway = get_mcp_gateway()
    await gateway.check_tool_permission("review_code", ctx)

    try:
        from app.agents import get_agent

        agent = get_agent("code-assistant")
        if not agent:
            return "Code Assistant agent is not available."

        query = f"Review this {language} code"
        if focus:
            query += f" with focus on {focus}"
        query += f":\n\n```{language}\n{code}\n```"

        result = await agent.ainvoke({"input": query})
        return _extract_response(result)

    except Exception as e:
        return f"Code review failed: {e}"


@mcp.tool()
async def explain_code(
    code: str,
    language: str = "python",
    ctx: Context | None = None,
) -> str:
    """Explain what a piece of code does.

    Args:
        code: The code to explain.
        language: Programming language.
        ctx: MCP context.

    Returns:
        Explanation of the code.
    """
    gateway = get_mcp_gateway()
    await gateway.check_tool_permission("explain_code", ctx)

    try:
        from app.agents import get_agent

        agent = get_agent("code-assistant")
        if not agent:
            return "Code Assistant agent is not available."

        query = f"Explain this {language} code:\n\n```{language}\n{code}\n```"
        result = await agent.ainvoke({"input": query})
        return _extract_response(result)

    except Exception as e:
        return f"Code explanation failed: {e}"


# ============== Helper Functions ==============


def _extract_response(result: dict[str, Any]) -> str:
    """Extract text response from LangGraph agent result.

    Args:
        result: The agent result dictionary.

    Returns:
        Extracted text response.
    """
    # Try different response formats
    if isinstance(result, str):
        return result

    # LangGraph format with messages
    if "messages" in result:
        messages = result["messages"]
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                return str(last_msg.content)
            if isinstance(last_msg, dict) and "content" in last_msg:
                return str(last_msg["content"])

    # Direct output format
    if "output" in result:
        return str(result["output"])

    # Response format
    if "response" in result:
        return str(result["response"])

    return str(result)


def get_mcp_server() -> FastMCP:
    """Get the MCP server instance.

    Returns:
        FastMCP server instance.
    """
    return mcp


def run_mcp_server(transport: str = "stdio") -> None:
    """Run the MCP server.

    Args:
        transport: Transport type ("stdio" or "http").
    """
    mcp.run(transport=transport)


async def handle_mcp_http_request(request_body: bytes) -> bytes:
    """Handle an HTTP MCP request.

    This can be used to integrate MCP with FastAPI.

    Args:
        request_body: Raw request body.

    Returns:
        Response body.
    """
    # This is a placeholder - actual implementation depends on
    # how FastMCP exposes HTTP handling
    return b""


if __name__ == "__main__":
    # Run standalone MCP server
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    run_mcp_server(transport)
