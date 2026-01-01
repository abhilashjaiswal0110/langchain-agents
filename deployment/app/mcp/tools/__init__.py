"""MCP tools organized by category.

Tools are defined in app/mcp/server.py using the @mcp.tool() decorator.
This package provides additional helper utilities for tool development.

Tool Categories:
- Research tools: research_topic, quick_search
- ServiceNow tools: create_incident, search_incidents, get_incident, update_incident, query_cmdb
- Document tools: generate_document, generate_sop
- IT Support tools: it_support_query, troubleshoot_issue
- Code Assistant tools: review_code, explain_code
"""

__all__: list[str] = []
