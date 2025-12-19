"""IT Document Generator Agent for SOP, WLI, and Policy documents.

This agent generates IT documentation:
- Standard Operating Procedures (SOP)
- Work Level Instructions (WLI)
- Policy Documents
- Using templates and best practices

Following Enterprise Development Standards:
- Software Architect: Template-based generation
- Security Architect: Compliance-aware content
- Data Architect: Structured document output
- Software Engineer: Type-safe, maintainable
"""

from datetime import datetime
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langsmith import traceable
from pydantic import BaseModel, Field

from app.agents.base.agent_base import BaseAgent, AgentConfig
from app.agents.base.tools import tool_error_handler


class DocumentState(BaseModel):
    """State schema for the Document Generator Agent."""

    messages: Annotated[list, add_messages] = Field(
        default_factory=list,
        description="Conversation history"
    )
    session_id: str | None = Field(default=None)
    user_id: str | None = Field(default=None)
    doc_type: Literal["sop", "wli", "policy", "general"] = Field(
        default="general",
        description="Type of document to generate"
    )
    title: str = Field(default="", description="Document title")
    department: str = Field(default="", description="Department/team")
    purpose: str = Field(default="", description="Document purpose")
    sections: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Document sections"
    )
    draft: str | None = Field(default=None, description="Current draft")
    version: str = Field(default="1.0", description="Document version")
    status: Literal["drafting", "review", "approved"] = Field(default="drafting")


# Document Templates
TEMPLATES = {
    "sop": """# Standard Operating Procedure (SOP)

## Document Control
- **Title**: {title}
- **Document ID**: SOP-{doc_id}
- **Version**: {version}
- **Effective Date**: {date}
- **Department**: {department}
- **Author**: {author}
- **Approver**: [Pending]

---

## 1. Purpose
{purpose}

## 2. Scope
{scope}

## 3. Definitions
{definitions}

## 4. Responsibilities
{responsibilities}

## 5. Procedure
{procedure}

## 6. Related Documents
{related_docs}

## 7. Revision History
| Version | Date | Description | Author |
|---------|------|-------------|--------|
| 1.0 | {date} | Initial release | {author} |

---
*This document is confidential and intended for internal use only.*
""",

    "wli": """# Work Level Instruction (WLI)

## Document Information
- **Title**: {title}
- **WLI ID**: WLI-{doc_id}
- **Version**: {version}
- **Date**: {date}
- **Department**: {department}

---

## Overview
{overview}

## Prerequisites
{prerequisites}

## Step-by-Step Instructions

{steps}

## Expected Outcomes
{outcomes}

## Troubleshooting
{troubleshooting}

## Support
For assistance, contact: {support_contact}

---
*Last Updated: {date}*
""",

    "policy": """# {title}

## Policy Document

**Policy ID**: POL-{doc_id}
**Version**: {version}
**Effective Date**: {date}
**Department**: {department}
**Classification**: Internal

---

## 1. Policy Statement
{policy_statement}

## 2. Purpose
{purpose}

## 3. Scope
{scope}

## 4. Definitions
{definitions}

## 5. Policy Details
{policy_details}

## 6. Compliance
{compliance}

## 7. Exceptions
{exceptions}

## 8. Enforcement
{enforcement}

## 9. Related Policies
{related_policies}

## 10. Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Author | | {date} | |
| Reviewer | | | |
| Approver | | | |

---
*This policy is effective immediately upon approval.*
"""
}


@tool
@tool_error_handler
def get_template(doc_type: str) -> str:
    """Get the template structure for a document type.

    Args:
        doc_type: Type of document (sop/wli/policy)

    Returns:
        Template structure with placeholders
    """
    template = TEMPLATES.get(doc_type.lower())
    if not template:
        return f"Error: Unknown document type '{doc_type}'. Available: sop, wli, policy"

    return f"Template for {doc_type.upper()}:\n\n{template}"


@tool
@tool_error_handler
def generate_section(
    section_name: str,
    content_requirements: str,
    doc_type: str = "sop"
) -> str:
    """Generate content for a specific document section.

    Args:
        section_name: Name of the section to generate
        content_requirements: Requirements/context for the section
        doc_type: Document type for context

    Returns:
        Generated section content
    """
    return (
        f"## {section_name}\n\n"
        f"[Content to be generated based on: {content_requirements}]\n\n"
        f"Note: This section should be tailored to your specific needs. "
        f"The AI will help draft appropriate content."
    )


@tool
@tool_error_handler
def validate_document(content: str, doc_type: str) -> str:
    """Validate document against required elements.

    Args:
        content: Document content to validate
        doc_type: Type of document

    Returns:
        Validation results with any missing elements
    """
    required = {
        "sop": ["Purpose", "Scope", "Responsibilities", "Procedure", "Version"],
        "wli": ["Overview", "Prerequisites", "Instructions", "Outcomes"],
        "policy": ["Policy Statement", "Purpose", "Scope", "Compliance", "Enforcement"],
    }

    doc_required = required.get(doc_type.lower(), [])
    content_lower = content.lower()

    missing = []
    present = []

    for element in doc_required:
        if element.lower() in content_lower:
            present.append(f"[OK] {element}")
        else:
            missing.append(f"[MISSING] {element}")

    result = "Document Validation Results\n" + "=" * 40 + "\n\n"
    result += "Present elements:\n" + "\n".join(present) + "\n\n"

    if missing:
        result += "Missing elements:\n" + "\n".join(missing) + "\n\n"
        result += "Recommendation: Add the missing sections before finalizing."
    else:
        result += "All required elements are present!"

    return result


@tool
@tool_error_handler
def format_document(content: str, output_format: str = "markdown") -> str:
    """Format the document for final output.

    Args:
        content: Document content
        output_format: Output format (markdown/html/text)

    Returns:
        Formatted document
    """
    if output_format == "html":
        # Basic markdown to HTML conversion
        html = content.replace("# ", "<h1>").replace("\n## ", "</h1>\n<h2>")
        html = html.replace("\n### ", "</h2>\n<h3>")
        html = f"<html><body>{html}</body></html>"
        return html
    elif output_format == "text":
        # Strip markdown formatting
        text = content.replace("#", "").replace("**", "").replace("*", "")
        return text
    else:
        return content  # Return markdown as-is


class DocumentAgent(BaseAgent):
    """IT Document Generator Agent for enterprise documentation.

    Creates professional IT documentation:
    - Standard Operating Procedures (SOP)
    - Work Level Instructions (WLI)
    - Policy Documents

    Example:
        >>> agent = DocumentAgent()
        >>> result = agent.create_document(
        ...     doc_type="sop",
        ...     title="Password Reset Procedure",
        ...     department="IT Security"
        ... )
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize the Document Agent."""
        super().__init__(config)

        self.register_tools([
            get_template,
            generate_section,
            validate_document,
            format_document,
        ])

    def _get_system_prompt(self) -> str:
        """Get the document agent's system prompt."""
        return """You are an expert IT Documentation Specialist creating professional
enterprise documentation including SOPs, WLIs, and Policy documents.

## Your Capabilities:
1. **Templates**: Get document templates with get_template
2. **Sections**: Generate specific sections with generate_section
3. **Validation**: Validate completeness with validate_document
4. **Formatting**: Format output with format_document

## Document Types:

### SOP (Standard Operating Procedure):
- Detailed step-by-step procedures
- Clear responsibilities
- Compliance requirements
- Version control

### WLI (Work Level Instruction):
- Task-specific instructions
- Prerequisites
- Step-by-step guidance
- Troubleshooting tips

### Policy:
- Clear policy statement
- Scope and applicability
- Compliance requirements
- Enforcement procedures

## Guidelines:
- Use clear, unambiguous language
- Include all required sections
- Be specific and actionable
- Consider compliance requirements
- Maintain consistent formatting
- Include version control information

## Process:
1. Understand the document requirements
2. Get the appropriate template
3. Generate content for each section
4. Validate completeness
5. Format for final output

Always validate documents before finalizing."""

    def _build_graph(self) -> StateGraph:
        """Build the document agent's workflow graph."""

        def call_model(state: DocumentState) -> dict:
            system_prompt = SystemMessage(content=self._get_system_prompt())
            messages = [system_prompt] + list(state.messages)
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def should_continue(state: DocumentState) -> str:
            messages = list(state.messages)
            if not messages:
                return "end"
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "end"

        graph = StateGraph(DocumentState)
        graph.add_node("agent", call_model)
        graph.add_node("tools", ToolNode(self._tools))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
        graph.add_edge("tools", "agent")

        return graph

    @traceable(name="document_create")
    def create_document(
        self,
        doc_type: Literal["sop", "wli", "policy"],
        title: str,
        department: str = "",
        purpose: str = "",
        additional_context: str = "",
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a new IT document.

        Args:
            doc_type: Type of document to create
            title: Document title
            department: Department/team
            purpose: Document purpose
            additional_context: Any additional context
            session_id: Optional session ID

        Returns:
            Generated document
        """
        message = f"""Please create a {doc_type.upper()} document with the following details:

Title: {title}
Department: {department}
Purpose: {purpose}

Additional Context: {additional_context or 'None provided'}

Please:
1. Get the appropriate template
2. Generate comprehensive content for each section
3. Validate the document
4. Present the final formatted document"""

        return self.invoke(
            message=message,
            session_id=session_id,
            doc_type=doc_type,
            title=title,
            department=department,
            purpose=purpose,
        )
