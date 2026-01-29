"""Debugging & Optimization Tools.

Tools for analyzing errors, tracing execution, identifying root causes,
and optimizing performance.
"""

import json
import re
import uuid

from langchain_core.tools import tool
from langsmith import traceable


# Session storage
_debug_store: dict[str, dict] = {}


@tool
@traceable(name="analyze_error", tags=["debugging", "errors"])
def analyze_error(
    error_message: str,
    stack_trace: str | None = None,
    code_context: str | None = None,
    language: str = "python",
    session_id: str = "default",
) -> str:
    """Analyze an error and provide diagnosis.

    Parses error messages and stack traces to:
    - Identify error type
    - Locate source of error
    - Suggest possible causes
    - Recommend fixes

    Args:
        error_message: The error message.
        stack_trace: Full stack trace if available.
        code_context: Relevant code context.
        language: Programming language.
        session_id: Session identifier.

    Returns:
        JSON string with error analysis.
    """
    debug_id = f"DBG-{str(uuid.uuid4())[:8].upper()}"

    analysis = {
        "id": debug_id,
        "error_type": "Unknown",
        "error_message": error_message,
        "location": None,
        "possible_causes": [],
        "recommended_fixes": [],
        "related_docs": [],
    }

    # Common Python errors
    python_errors = {
        "TypeError": {
            "causes": [
                "Incorrect argument types",
                "Calling non-callable object",
                "Unsupported operand types",
            ],
            "fixes": [
                "Check argument types match function signature",
                "Verify object is callable before calling",
                "Add type checking or conversion",
            ],
        },
        "AttributeError": {
            "causes": [
                "Accessing non-existent attribute",
                "NoneType object",
                "Incorrect class inheritance",
            ],
            "fixes": [
                "Check if attribute exists with hasattr()",
                "Add null checks before accessing",
                "Verify object is not None",
            ],
        },
        "KeyError": {
            "causes": [
                "Dictionary key does not exist",
                "Typo in key name",
                "Key not populated",
            ],
            "fixes": [
                "Use dict.get() with default value",
                "Check key existence with 'in' operator",
                "Verify data population logic",
            ],
        },
        "ValueError": {
            "causes": [
                "Invalid value for operation",
                "Incorrect data format",
                "Out of range value",
            ],
            "fixes": [
                "Validate input before processing",
                "Add try-except for conversion",
                "Check value ranges",
            ],
        },
        "ImportError": {
            "causes": [
                "Module not installed",
                "Circular import",
                "Incorrect module path",
            ],
            "fixes": [
                "Install missing package",
                "Reorganize imports to avoid circular dependencies",
                "Verify module path and name",
            ],
        },
        "IndexError": {
            "causes": [
                "List index out of range",
                "Empty list access",
                "Off-by-one error",
            ],
            "fixes": [
                "Check list length before access",
                "Use try-except or conditional",
                "Review loop bounds",
            ],
        },
        "ConnectionError": {
            "causes": [
                "Network unreachable",
                "Service unavailable",
                "Timeout exceeded",
            ],
            "fixes": [
                "Implement retry with backoff",
                "Check network connectivity",
                "Verify service endpoint",
            ],
        },
    }

    # Identify error type
    for error_type, info in python_errors.items():
        if error_type in error_message:
            analysis["error_type"] = error_type
            analysis["possible_causes"] = info["causes"]
            analysis["recommended_fixes"] = info["fixes"]
            break

    # Parse stack trace for location
    if stack_trace:
        # Find file and line info
        file_match = re.search(r'File "([^"]+)", line (\d+)', stack_trace)
        if file_match:
            analysis["location"] = {
                "file": file_match.group(1),
                "line": int(file_match.group(2)),
            }

        # Find the most relevant frames
        frames = re.findall(r'File "([^"]+)", line (\d+), in (\w+)', stack_trace)
        if frames:
            analysis["stack_frames"] = [
                {"file": f[0], "line": int(f[1]), "function": f[2]}
                for f in frames[-5:]  # Last 5 frames
            ]

    # Add documentation links
    analysis["related_docs"] = [
        f"https://docs.python.org/3/library/exceptions.html#{analysis['error_type']}",
        "https://stackoverflow.com/search?q=" + error_message.replace(" ", "+")[:50],
    ]

    _debug_store[debug_id] = analysis

    return json.dumps(analysis, indent=2)


@tool
@traceable(name="trace_execution", tags=["debugging", "tracing"])
def trace_execution(
    code: str,
    inputs: dict | None = None,
    breakpoints: list[int] | None = None,
    language: str = "python",
) -> str:
    """Trace code execution flow.

    Simulates stepping through code to understand:
    - Variable states at each step
    - Control flow path
    - Function call sequence

    Args:
        code: Code to trace.
        inputs: Input values for variables.
        breakpoints: Line numbers to pause at.
        language: Programming language.

    Returns:
        JSON string with execution trace.
    """
    inputs = inputs or {}
    breakpoints = breakpoints or []

    lines = code.split("\n")
    trace_steps = []
    variables = dict(inputs)

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        step = {
            "line": i,
            "code": stripped[:80],
            "action": "execute",
            "variables": dict(variables),
        }

        # Simple variable assignment detection
        if "=" in stripped and not any(op in stripped for op in ["==", "!=", "<=", ">="]):
            parts = stripped.split("=", 1)
            var_name = parts[0].strip()
            if var_name.isidentifier():
                step["action"] = f"assign {var_name}"
                variables[var_name] = "<assigned>"

        # Control flow detection
        if stripped.startswith("if "):
            step["action"] = "condition"
            step["branch"] = "if"
        elif stripped.startswith("elif "):
            step["action"] = "condition"
            step["branch"] = "elif"
        elif stripped.startswith("else:"):
            step["action"] = "else branch"
        elif stripped.startswith("for "):
            step["action"] = "loop start"
            step["loop_type"] = "for"
        elif stripped.startswith("while "):
            step["action"] = "loop start"
            step["loop_type"] = "while"
        elif stripped.startswith("return"):
            step["action"] = "return"
        elif stripped.startswith("def "):
            step["action"] = "function definition"
            func_name = stripped.split("(")[0].replace("def ", "")
            step["function"] = func_name

        # Check for breakpoint
        if i in breakpoints:
            step["breakpoint"] = True

        trace_steps.append(step)

    result = {
        "total_lines": len(lines),
        "executed_steps": len(trace_steps),
        "final_variables": variables,
        "trace": trace_steps,
        "breakpoints_hit": [s["line"] for s in trace_steps if s.get("breakpoint")],
    }

    return json.dumps(result, indent=2)


@tool
@traceable(name="identify_root_cause", tags=["debugging", "rca"])
def identify_root_cause(
    symptoms: list[str],
    error_logs: str | None = None,
    system_state: dict | None = None,
    session_id: str = "default",
) -> str:
    """Identify root cause of an issue using RCA methodology.

    Applies 5 Whys and fault tree analysis to determine root cause.

    Args:
        symptoms: List of observed symptoms.
        error_logs: Relevant error logs.
        system_state: Current system state information.
        session_id: Session identifier.

    Returns:
        JSON string with root cause analysis.
    """
    system_state = system_state or {}

    # Categorize symptoms
    categories = {
        "performance": [],
        "functionality": [],
        "connectivity": [],
        "data": [],
        "security": [],
    }

    for symptom in symptoms:
        symptom_lower = symptom.lower()
        if any(word in symptom_lower for word in ["slow", "timeout", "latency", "cpu", "memory"]):
            categories["performance"].append(symptom)
        elif any(word in symptom_lower for word in ["connection", "network", "unreachable", "refused"]):
            categories["connectivity"].append(symptom)
        elif any(word in symptom_lower for word in ["data", "null", "missing", "corrupt"]):
            categories["data"].append(symptom)
        elif any(word in symptom_lower for word in ["auth", "permission", "denied", "unauthorized"]):
            categories["security"].append(symptom)
        else:
            categories["functionality"].append(symptom)

    # Generate 5 Whys analysis
    five_whys = []
    primary_symptom = symptoms[0] if symptoms else "Issue observed"

    # Simulated 5 Whys
    why_chain = [
        {"level": 1, "question": f"Why did {primary_symptom}?", "answer": "Direct cause"},
        {"level": 2, "question": "Why did the direct cause occur?", "answer": "Intermediate cause"},
        {"level": 3, "question": "Why did the intermediate cause occur?", "answer": "Contributing factor"},
        {"level": 4, "question": "Why did the contributing factor exist?", "answer": "Process gap"},
        {"level": 5, "question": "Why did the process gap exist?", "answer": "Root cause identified"},
    ]

    # Generate potential root causes based on categories
    potential_causes = []

    if categories["performance"]:
        potential_causes.extend([
            {"cause": "Resource exhaustion", "probability": "high", "evidence": categories["performance"]},
            {"cause": "Inefficient algorithm", "probability": "medium", "evidence": []},
            {"cause": "External service latency", "probability": "medium", "evidence": []},
        ])

    if categories["connectivity"]:
        potential_causes.extend([
            {"cause": "Network configuration issue", "probability": "high", "evidence": categories["connectivity"]},
            {"cause": "Service unavailable", "probability": "medium", "evidence": []},
            {"cause": "Firewall blocking traffic", "probability": "low", "evidence": []},
        ])

    if categories["data"]:
        potential_causes.extend([
            {"cause": "Data validation missing", "probability": "high", "evidence": categories["data"]},
            {"cause": "Database corruption", "probability": "low", "evidence": []},
            {"cause": "Race condition", "probability": "medium", "evidence": []},
        ])

    # Determine most likely root cause
    most_likely = potential_causes[0] if potential_causes else {"cause": "Unknown", "probability": "unknown"}

    result = {
        "symptoms_analyzed": len(symptoms),
        "symptom_categories": {k: len(v) for k, v in categories.items()},
        "five_whys": why_chain,
        "potential_root_causes": potential_causes,
        "most_likely_root_cause": most_likely,
        "recommended_investigation": [
            "Review logs around the time of first symptom",
            "Check resource utilization metrics",
            "Verify recent changes or deployments",
            "Test in isolation to confirm root cause",
        ],
    }

    return json.dumps(result, indent=2)


@tool
@traceable(name="propose_fix", tags=["debugging", "remediation"])
def propose_fix(
    issue_description: str,
    root_cause: str | None = None,
    affected_code: str | None = None,
    language: str = "python",
) -> str:
    """Propose a fix for an identified issue.

    Args:
        issue_description: Description of the issue.
        root_cause: Identified root cause.
        affected_code: Code that needs to be fixed.
        language: Programming language.

    Returns:
        JSON string with proposed fix.
    """
    fixes = []
    issue_lower = issue_description.lower()

    # Match issue patterns to fixes
    if "null" in issue_lower or "none" in issue_lower:
        fixes.append({
            "type": "null_check",
            "description": "Add null/None checks before accessing",
            "before": "result = obj.method()",
            "after": "result = obj.method() if obj is not None else None",
            "explanation": "Prevents NoneType errors by checking before access",
        })

    if "timeout" in issue_lower or "slow" in issue_lower:
        fixes.append({
            "type": "timeout_handling",
            "description": "Add timeout and retry logic",
            "before": "response = requests.get(url)",
            "after": '''from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_with_retry(url):
    return requests.get(url, timeout=30)

response = fetch_with_retry(url)''',
            "explanation": "Implements retry with exponential backoff for transient failures",
        })

    if "memory" in issue_lower:
        fixes.append({
            "type": "memory_optimization",
            "description": "Optimize memory usage",
            "before": "data = [process(item) for item in large_list]",
            "after": '''def process_generator(items):
    for item in items:
        yield process(item)

# Use generator instead of list
for result in process_generator(large_list):
    handle(result)''',
            "explanation": "Uses generator to process items one at a time, reducing memory",
        })

    if "exception" in issue_lower or "error" in issue_lower:
        fixes.append({
            "type": "error_handling",
            "description": "Add proper error handling",
            "before": '''result = risky_operation()''',
            "after": '''try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    result = fallback_value
    # Optionally re-raise or handle gracefully''',
            "explanation": "Catches specific exceptions and handles them appropriately",
        })

    if not fixes:
        fixes.append({
            "type": "general",
            "description": "General debugging approach",
            "steps": [
                "Add logging around the problematic area",
                "Validate inputs and state before operation",
                "Check for edge cases",
                "Add unit tests to reproduce the issue",
            ],
        })

    result = {
        "issue": issue_description,
        "root_cause": root_cause,
        "proposed_fixes": fixes,
        "testing_recommendation": "Create unit test to reproduce issue before applying fix",
        "rollback_plan": "Revert to previous version if fix causes new issues",
    }

    return json.dumps(result, indent=2)


@tool
@traceable(name="analyze_performance", tags=["debugging", "performance"])
def analyze_performance(
    code: str,
    metrics: dict | None = None,
    language: str = "python",
) -> str:
    """Analyze code for performance issues.

    Identifies:
    - Algorithm complexity
    - Memory usage patterns
    - I/O bottlenecks
    - Optimization opportunities

    Args:
        code: Code to analyze.
        metrics: Runtime metrics if available.
        language: Programming language.

    Returns:
        JSON string with performance analysis.
    """
    metrics = metrics or {}
    issues = []
    optimizations = []

    lines = code.split("\n")

    # Detect potential performance issues
    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Nested loops (O(n^2) or worse)
        if "for " in stripped:
            # Check for nested loops
            indent = len(line) - len(line.lstrip())
            for j in range(max(0, i-5), i):
                prev_line = lines[j-1] if j > 0 else ""
                prev_indent = len(prev_line) - len(prev_line.lstrip())
                if "for " in prev_line and prev_indent < indent:
                    issues.append({
                        "line": i,
                        "type": "nested_loop",
                        "severity": "medium",
                        "message": "Nested loop detected - O(n^2) complexity",
                        "optimization": "Consider using dictionary lookup or set operations",
                    })
                    break

        # String concatenation in loop
        if "+=" in stripped and ("str" in stripped or '"' in stripped or "'" in stripped):
            issues.append({
                "line": i,
                "type": "string_concat",
                "severity": "low",
                "message": "String concatenation may be inefficient in loops",
                "optimization": "Use list.append() and ''.join() instead",
            })

        # List comprehension vs map/filter
        if "map(" in stripped or "filter(" in stripped:
            optimizations.append({
                "line": i,
                "suggestion": "Consider list comprehension for readability",
            })

        # Database query in loop (N+1 pattern)
        if any(kw in stripped for kw in [".query(", ".execute(", ".fetch", ".find("]):
            # Check if inside a loop
            for j in range(max(0, i-10), i):
                if "for " in lines[j-1]:
                    issues.append({
                        "line": i,
                        "type": "n_plus_1",
                        "severity": "high",
                        "message": "Database query inside loop - N+1 problem",
                        "optimization": "Fetch all data in single query before loop",
                    })
                    break

    # General optimizations
    if "import time" in code and "sleep" in code:
        optimizations.append({
            "type": "async",
            "suggestion": "Consider async/await for I/O-bound operations",
        })

    result = {
        "lines_analyzed": len(lines),
        "issues_found": len(issues),
        "performance_issues": issues,
        "optimization_suggestions": optimizations,
        "complexity_estimate": {
            "time": "O(n)" if not issues else "O(n^2) or higher",
            "space": "O(n)",
        },
        "recommendations": [
            "Profile code to identify actual bottlenecks",
            "Use caching for repeated computations",
            "Consider lazy evaluation for large datasets",
            "Batch database operations",
        ],
    }

    return json.dumps(result, indent=2)


@tool
@traceable(name="detect_memory_issues", tags=["debugging", "memory"])
def detect_memory_issues(
    code: str,
    language: str = "python",
) -> str:
    """Detect potential memory issues in code.

    Identifies:
    - Memory leaks
    - Large object creation
    - Circular references
    - Resource not closed

    Args:
        code: Code to analyze.
        language: Programming language.

    Returns:
        JSON string with memory issue analysis.
    """
    issues = []
    lines = code.split("\n")

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Large list/dict creation
        if re.search(r'\[\s*\w+\s+for\s+\w+\s+in\s+range\s*\(\s*\d{5,}', stripped):
            issues.append({
                "line": i,
                "type": "large_allocation",
                "severity": "high",
                "message": "Large list allocation detected",
                "recommendation": "Use generator expression or itertools",
            })

        # File not closed
        if "open(" in stripped and "with " not in stripped:
            issues.append({
                "line": i,
                "type": "unclosed_resource",
                "severity": "medium",
                "message": "File opened without context manager",
                "recommendation": "Use 'with open(...)' to ensure file is closed",
            })

        # Connection not closed
        if any(kw in stripped for kw in ["connect(", "Connection(", "session("]):
            if "with " not in stripped and ".close()" not in code:
                issues.append({
                    "line": i,
                    "type": "unclosed_connection",
                    "severity": "high",
                    "message": "Connection opened without explicit close",
                    "recommendation": "Use context manager or try-finally with close()",
                })

        # Global variable accumulation
        if stripped.startswith("global ") or (stripped.endswith(".append(") and "global" in code):
            issues.append({
                "line": i,
                "type": "global_accumulation",
                "severity": "medium",
                "message": "Data accumulating in global variable",
                "recommendation": "Implement cleanup or use bounded data structures",
            })

        # Caching without limit
        if "@cache" in stripped or "@lru_cache" in stripped:
            if "maxsize" not in stripped:
                issues.append({
                    "line": i,
                    "type": "unbounded_cache",
                    "severity": "low",
                    "message": "Cache without size limit",
                    "recommendation": "Set maxsize parameter to limit memory usage",
                })

    result = {
        "lines_analyzed": len(lines),
        "issues_found": len(issues),
        "memory_issues": issues,
        "by_severity": {
            "high": len([i for i in issues if i["severity"] == "high"]),
            "medium": len([i for i in issues if i["severity"] == "medium"]),
            "low": len([i for i in issues if i["severity"] == "low"]),
        },
        "general_recommendations": [
            "Use generators for large data processing",
            "Implement proper resource cleanup",
            "Profile memory usage with memory_profiler",
            "Consider using __slots__ for classes with many instances",
        ],
    }

    return json.dumps(result, indent=2)
