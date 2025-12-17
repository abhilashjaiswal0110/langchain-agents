"""Code Assistant Agent for Application Modernization.

This agent helps with:
- Legacy code analysis
- Modernization recommendations
- Code transformation suggestions
- Best practice enforcement

Following Enterprise Development Standards:
- Software Architect: Pattern detection and suggestions
- Security Architect: Security vulnerability detection
- Data Architect: Code structure analysis
- Software Engineer: Clean code principles
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


class CodeAssistantState(BaseModel):
    """State schema for the Code Assistant Agent."""

    messages: Annotated[list, add_messages] = Field(
        default_factory=list,
        description="Conversation history"
    )
    session_id: str | None = Field(default=None)
    user_id: str | None = Field(default=None)
    source_code: str = Field(default="", description="Code being analyzed")
    source_language: str = Field(default="python", description="Source language")
    target_framework: str | None = Field(default=None, description="Target framework")
    analysis: dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis results"
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Improvement suggestions"
    )
    modernized_code: str | None = Field(
        default=None,
        description="Modernized code output"
    )


# Code pattern definitions
LEGACY_PATTERNS = {
    "python": {
        "old_style_class": r"class\s+\w+\s*:\s*$",
        "print_statement": r"print\s+[^(]",
        "old_string_format": r"%[sd]",
        "except_all": r"except\s*:",
        "mutable_default": r"def\s+\w+\([^)]*=\s*\[\]",
    },
    "javascript": {
        "var_usage": r"\bvar\s+",
        "callback_hell": r"function.*\{.*function.*\{",
        "jquery_selector": r"\$\(['\"]",
        "synchronous_xhr": r"\.open\([^)]*false\)",
    },
    "java": {
        "raw_types": r"List\s+\w+\s*=\s*new\s+ArrayList\(\)",
        "old_date_api": r"new\s+Date\(\)",
        "string_concatenation": r'\+\s*"',
    },
}

MODERNIZATION_SUGGESTIONS = {
    "python": {
        "type_hints": "Add type hints for better code documentation and IDE support",
        "dataclasses": "Use dataclasses for data containers instead of plain classes",
        "pathlib": "Use pathlib instead of os.path for file operations",
        "f_strings": "Use f-strings instead of .format() or % formatting",
        "context_managers": "Use context managers (with statements) for resource handling",
        "async_await": "Consider async/await for I/O-bound operations",
    },
    "javascript": {
        "const_let": "Use const/let instead of var",
        "arrow_functions": "Use arrow functions for cleaner syntax",
        "async_await": "Use async/await instead of callbacks/promises",
        "template_literals": "Use template literals instead of string concatenation",
        "destructuring": "Use destructuring for cleaner variable assignment",
        "modules": "Use ES6 modules instead of CommonJS require",
    },
    "java": {
        "generics": "Use generics instead of raw types",
        "streams": "Use Stream API for collection operations",
        "optional": "Use Optional instead of null checks",
        "records": "Use records for immutable data classes (Java 16+)",
        "var_keyword": "Use var for local variable type inference (Java 10+)",
        "new_date_api": "Use java.time API instead of java.util.Date",
    },
}


@tool
@tool_error_handler
def analyze_code(code: str, language: str = "python") -> str:
    """Analyze code for patterns and potential issues.

    Args:
        code: Source code to analyze
        language: Programming language

    Returns:
        Analysis results with identified patterns
    """
    import re

    results = {
        "language": language,
        "lines": len(code.split("\n")),
        "characters": len(code),
        "issues": [],
        "patterns": [],
    }

    patterns = LEGACY_PATTERNS.get(language, {})

    for pattern_name, pattern_regex in patterns.items():
        matches = re.findall(pattern_regex, code, re.MULTILINE)
        if matches:
            results["patterns"].append({
                "name": pattern_name,
                "count": len(matches),
                "severity": "warning",
            })

    # Basic complexity analysis
    if language == "python":
        indent_levels = [len(line) - len(line.lstrip()) for line in code.split("\n") if line.strip()]
        max_indent = max(indent_levels) if indent_levels else 0
        if max_indent > 16:
            results["issues"].append("High nesting depth detected - consider refactoring")

        if code.count("def ") > 10:
            results["issues"].append("Many functions in single file - consider splitting")

    output = f"Code Analysis Results\n{'=' * 40}\n\n"
    output += f"Language: {results['language']}\n"
    output += f"Lines: {results['lines']}\n"
    output += f"Characters: {results['characters']}\n\n"

    if results["patterns"]:
        output += "Legacy Patterns Found:\n"
        for p in results["patterns"]:
            output += f"  - {p['name']}: {p['count']} occurrence(s)\n"

    if results["issues"]:
        output += "\nPotential Issues:\n"
        for issue in results["issues"]:
            output += f"  - {issue}\n"

    if not results["patterns"] and not results["issues"]:
        output += "No significant issues detected.\n"

    return output


@tool
@tool_error_handler
def detect_security_issues(code: str, language: str = "python") -> str:
    """Detect potential security vulnerabilities in code.

    Args:
        code: Source code to check
        language: Programming language

    Returns:
        Security analysis results
    """
    import re

    vulnerabilities = []

    # Common patterns across languages
    common_patterns = {
        "hardcoded_secret": (r'(password|secret|api_key|token)\s*=\s*["\'][^"\']+["\']', "HIGH"),
        "sql_injection": (r'(execute|query)\s*\([^)]*\+|f["\'].*SELECT', "HIGH"),
        "eval_usage": (r'\beval\s*\(', "HIGH"),
        "exec_usage": (r'\bexec\s*\(', "HIGH"),
    }

    language_patterns = {
        "python": {
            "pickle_load": (r'pickle\.load', "MEDIUM"),
            "shell_true": (r'shell\s*=\s*True', "MEDIUM"),
            "yaml_load": (r'yaml\.load\s*\([^)]*\)', "MEDIUM"),
        },
        "javascript": {
            "innerhtml": (r'\.innerHTML\s*=', "MEDIUM"),
            "document_write": (r'document\.write\s*\(', "MEDIUM"),
        },
    }

    all_patterns = {**common_patterns, **language_patterns.get(language, {})}

    for name, (pattern, severity) in all_patterns.items():
        if re.search(pattern, code, re.IGNORECASE):
            vulnerabilities.append({
                "name": name.replace("_", " ").title(),
                "severity": severity,
                "pattern": pattern,
            })

    output = f"Security Analysis\n{'=' * 40}\n\n"

    if vulnerabilities:
        output += f"Found {len(vulnerabilities)} potential issue(s):\n\n"
        for v in vulnerabilities:
            output += f"[{v['severity']}] {v['name']}\n"
    else:
        output += "No obvious security issues detected.\n"
        output += "Note: This is a basic scan. Consider using dedicated security tools.\n"

    return output


@tool
@tool_error_handler
def suggest_improvements(code: str, language: str = "python") -> str:
    """Suggest modernization improvements for the code.

    Args:
        code: Source code to improve
        language: Programming language

    Returns:
        Improvement suggestions
    """
    suggestions = MODERNIZATION_SUGGESTIONS.get(language, {})

    output = f"Modernization Suggestions for {language.title()}\n{'=' * 50}\n\n"

    # Check which suggestions apply
    applicable = []

    if language == "python":
        if "def " in code and ": " not in code.split("def ")[1].split(")")[0]:
            applicable.append(("type_hints", suggestions.get("type_hints", "")))
        if "class " in code and "__init__" in code:
            applicable.append(("dataclasses", suggestions.get("dataclasses", "")))
        if "os.path" in code:
            applicable.append(("pathlib", suggestions.get("pathlib", "")))
        if ".format(" in code or "% " in code:
            applicable.append(("f_strings", suggestions.get("f_strings", "")))

    elif language == "javascript":
        if "var " in code:
            applicable.append(("const_let", suggestions.get("const_let", "")))
        if "function(" in code:
            applicable.append(("arrow_functions", suggestions.get("arrow_functions", "")))
        if ".then(" in code:
            applicable.append(("async_await", suggestions.get("async_await", "")))

    if applicable:
        output += "Recommended Improvements:\n\n"
        for name, suggestion in applicable:
            output += f"**{name.replace('_', ' ').title()}**\n"
            output += f"  {suggestion}\n\n"
    else:
        output += "Code appears to follow modern practices.\n"
        output += "General recommendations:\n"
        for name, suggestion in list(suggestions.items())[:3]:
            output += f"- {suggestion}\n"

    return output


@tool
@tool_error_handler
def transform_code_pattern(
    code: str,
    transformation: str,
    language: str = "python"
) -> str:
    """Apply a specific code transformation.

    Args:
        code: Source code to transform
        transformation: Type of transformation
        language: Programming language

    Returns:
        Transformed code example
    """
    import re

    transformations = {
        "python": {
            "format_to_fstring": lambda c: re.sub(
                r'"([^"]*)"\.format\(([^)]+)\)',
                lambda m: f'f"{m.group(1).replace("{}", "{" + m.group(2) + "}")}"',
                c
            ),
            "add_type_hints_example": lambda c: (
                "# Example transformation:\n"
                "# Before: def greet(name):\n"
                "# After:  def greet(name: str) -> str:\n\n"
                f"# Your code:\n{c}"
            ),
        },
    }

    lang_transforms = transformations.get(language, {})
    transform_func = lang_transforms.get(transformation)

    if transform_func:
        result = transform_func(code)
        return f"Transformed Code ({transformation}):\n\n```{language}\n{result}\n```"
    else:
        return (
            f"Transformation '{transformation}' not available for {language}.\n"
            f"Available: {', '.join(lang_transforms.keys())}"
        )


class CodeAssistantAgent(BaseAgent):
    """Code Assistant Agent for application modernization.

    Features:
    - Legacy code analysis
    - Security vulnerability detection
    - Modernization suggestions
    - Code transformation

    Example:
        >>> agent = CodeAssistantAgent()
        >>> result = agent.analyze("def foo(x): return x + 1")
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize the Code Assistant Agent."""
        super().__init__(config)

        self.register_tools([
            analyze_code,
            detect_security_issues,
            suggest_improvements,
            transform_code_pattern,
        ])

    def _get_system_prompt(self) -> str:
        """Get the code assistant's system prompt."""
        return """You are a Code Modernization Expert helping developers
update legacy code to modern standards.

## Your Capabilities:
1. **Analysis**: Analyze code for patterns (analyze_code)
2. **Security**: Detect vulnerabilities (detect_security_issues)
3. **Suggestions**: Provide improvements (suggest_improvements)
4. **Transform**: Apply transformations (transform_code_pattern)

## Process:
1. First analyze the code structure and patterns
2. Check for security vulnerabilities
3. Suggest applicable modernizations
4. Provide code examples when helpful

## Guidelines:
- Explain WHY changes are recommended
- Prioritize security issues
- Consider backward compatibility
- Provide before/after examples
- Focus on maintainability

## Languages Supported:
- Python (primary)
- JavaScript
- Java

For unsupported languages, provide general best practices."""

    def _build_graph(self) -> StateGraph:
        """Build the code assistant's workflow graph."""

        def call_model(state: dict) -> dict:
            system_prompt = SystemMessage(content=self._get_system_prompt())
            messages = [system_prompt] + state["messages"]
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def should_continue(state: dict) -> str:
            messages = state["messages"]
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "end"

        graph = StateGraph(CodeAssistantState)
        graph.add_node("agent", call_model)
        graph.add_node("tools", ToolNode(self._tools))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
        graph.add_edge("tools", "agent")

        return graph

    @traceable(name="code_analyze")
    def analyze(
        self,
        code: str,
        language: str = "python",
        include_security: bool = True,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Analyze code for modernization opportunities.

        Args:
            code: Source code to analyze
            language: Programming language
            include_security: Include security analysis
            session_id: Optional session ID

        Returns:
            Analysis results
        """
        message = f"""Please analyze this {language} code:

```{language}
{code}
```

Provide:
1. Code structure analysis
2. Legacy patterns found
{'3. Security vulnerabilities' if include_security else ''}
4. Modernization suggestions
5. Example improvements"""

        return self.invoke(
            message=message,
            session_id=session_id,
            source_code=code,
            source_language=language,
        )

    @traceable(name="code_modernize")
    def modernize(
        self,
        code: str,
        language: str = "python",
        target_framework: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Get modernization recommendations.

        Args:
            code: Source code to modernize
            language: Programming language
            target_framework: Target framework if migrating
            session_id: Optional session ID

        Returns:
            Modernization plan
        """
        message = f"""Please help modernize this {language} code:

```{language}
{code}
```

{'Target framework: ' + target_framework if target_framework else ''}

Provide step-by-step modernization recommendations."""

        return self.invoke(
            message=message,
            session_id=session_id,
            source_code=code,
            source_language=language,
            target_framework=target_framework,
        )
