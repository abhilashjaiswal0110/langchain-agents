"""Code Review & Quality Tools.

Tools for performing automated code reviews, checking code style,
analyzing complexity, and suggesting improvements.
"""

import json
import re
import uuid
from datetime import datetime

from langchain_core.tools import tool
from langsmith import traceable


# Session storage for review results
_review_store: dict[str, dict] = {}


@tool
@traceable(name="review_code", tags=["review", "quality"])
def review_code(
    code: str,
    language: str = "python",
    focus_areas: list[str] | None = None,
    session_id: str = "default",
) -> str:
    """Perform automated code review.

    Reviews code for:
    - Code quality and maintainability
    - Potential bugs and issues
    - Performance concerns
    - Security vulnerabilities
    - Best practice violations

    Args:
        code: Code to review.
        language: Programming language.
        focus_areas: Specific areas to focus on (security, performance, style).
        session_id: Session identifier.

    Returns:
        JSON string with review results.
    """
    focus_areas = focus_areas or ["quality", "security", "performance", "style"]
    review_id = f"REV-{str(uuid.uuid4())[:8].upper()}"

    issues = []
    suggestions = []
    lines = code.split("\n")

    # Analyze code
    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Check for common issues
        if "security" in focus_areas or "quality" in focus_areas:
            # Hardcoded credentials
            if re.search(r'(password|secret|api_key|token)\s*=\s*["\'][^"\']+["\']', line, re.I):
                issues.append({
                    "line": i,
                    "severity": "critical",
                    "category": "security",
                    "message": "Potential hardcoded credential detected",
                    "suggestion": "Use environment variables for sensitive data",
                })

            # SQL injection risk
            if re.search(r'(execute|query)\s*\([^)]*%s|f["\'].*{.*}.*["\'].*(?:select|insert|update|delete)', line, re.I):
                issues.append({
                    "line": i,
                    "severity": "high",
                    "category": "security",
                    "message": "Potential SQL injection vulnerability",
                    "suggestion": "Use parameterized queries",
                })

        if "quality" in focus_areas:
            # Bare except
            if stripped == "except:" or stripped.startswith("except Exception:"):
                issues.append({
                    "line": i,
                    "severity": "medium",
                    "category": "quality",
                    "message": "Bare except clause catches all exceptions",
                    "suggestion": "Catch specific exceptions",
                })

            # TODO/FIXME comments
            if re.search(r'#\s*(TODO|FIXME|XXX|HACK)', line, re.I):
                issues.append({
                    "line": i,
                    "severity": "low",
                    "category": "quality",
                    "message": "Unresolved TODO/FIXME comment",
                    "suggestion": "Address or create issue for tracking",
                })

            # Magic numbers
            if re.search(r'[^0-9a-zA-Z_]([2-9]\d{2,}|[1-9]\d{3,})[^0-9a-zA-Z_]', line):
                issues.append({
                    "line": i,
                    "severity": "low",
                    "category": "quality",
                    "message": "Magic number detected",
                    "suggestion": "Extract to named constant",
                })

        if "performance" in focus_areas:
            # Nested loops
            if "for " in line and any("for " in lines[j] for j in range(max(0, i-3), i)):
                issues.append({
                    "line": i,
                    "severity": "medium",
                    "category": "performance",
                    "message": "Nested loop detected",
                    "suggestion": "Consider algorithm optimization or caching",
                })

        if "style" in focus_areas:
            # Long lines
            if len(line) > 120:
                issues.append({
                    "line": i,
                    "severity": "info",
                    "category": "style",
                    "message": f"Line too long ({len(line)} chars)",
                    "suggestion": "Break into multiple lines",
                })

            # Missing docstring (simplified check)
            if stripped.startswith("def ") and i < len(lines):
                next_line = lines[i].strip() if i < len(lines) else ""
                if not next_line.startswith('"""') and not next_line.startswith("'''"):
                    issues.append({
                        "line": i,
                        "severity": "low",
                        "category": "style",
                        "message": "Function missing docstring",
                        "suggestion": "Add docstring describing function purpose",
                    })

    # Generate suggestions
    if len([i for i in issues if i["severity"] == "critical"]) > 0:
        suggestions.append("Address critical security issues immediately")
    if len([i for i in issues if i["category"] == "quality"]) > 3:
        suggestions.append("Consider refactoring to improve code quality")
    if not issues:
        suggestions.append("Code looks good! No major issues found.")

    # Calculate score
    severity_weights = {"critical": 25, "high": 15, "medium": 5, "low": 2, "info": 1}
    total_penalty = sum(severity_weights.get(i["severity"], 0) for i in issues)
    score = max(0, 100 - total_penalty)

    review = {
        "id": review_id,
        "language": language,
        "lines_reviewed": len(lines),
        "issues_found": len(issues),
        "issues": issues,
        "suggestions": suggestions,
        "score": score,
        "grade": "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 60 else "F",
        "summary": {
            "critical": len([i for i in issues if i["severity"] == "critical"]),
            "high": len([i for i in issues if i["severity"] == "high"]),
            "medium": len([i for i in issues if i["severity"] == "medium"]),
            "low": len([i for i in issues if i["severity"] == "low"]),
            "info": len([i for i in issues if i["severity"] == "info"]),
        },
        "created_at": datetime.now().isoformat(),
    }

    _review_store[review_id] = review

    return json.dumps(review, indent=2)


@tool
@traceable(name="check_code_style", tags=["review", "style"])
def check_code_style(
    code: str,
    language: str = "python",
    style_guide: str = "pep8",
) -> str:
    """Check code against style guidelines.

    Supported style guides:
    - pep8: Python PEP 8
    - google: Google style guide
    - airbnb: Airbnb JavaScript guide

    Args:
        code: Code to check.
        language: Programming language.
        style_guide: Style guide to apply.

    Returns:
        JSON string with style violations.
    """
    violations = []
    lines = code.split("\n")

    for i, line in enumerate(lines, 1):
        # Line length
        max_length = 79 if style_guide == "pep8" else 100
        if len(line) > max_length:
            violations.append({
                "line": i,
                "rule": "line-length",
                "message": f"Line exceeds {max_length} characters ({len(line)})",
            })

        # Trailing whitespace
        if line != line.rstrip():
            violations.append({
                "line": i,
                "rule": "trailing-whitespace",
                "message": "Trailing whitespace detected",
            })

        # Indentation (Python)
        if language == "python":
            leading_spaces = len(line) - len(line.lstrip())
            if leading_spaces > 0 and leading_spaces % 4 != 0:
                violations.append({
                    "line": i,
                    "rule": "indentation",
                    "message": "Indentation should be a multiple of 4 spaces",
                })

        # Multiple blank lines
        if i > 2 and not lines[i-1].strip() and not lines[i-2].strip() and not line.strip():
            violations.append({
                "line": i,
                "rule": "blank-lines",
                "message": "Too many consecutive blank lines",
            })

    result = {
        "style_guide": style_guide,
        "language": language,
        "lines_checked": len(lines),
        "violations_count": len(violations),
        "violations": violations,
        "compliant": len(violations) == 0,
    }

    return json.dumps(result, indent=2)


@tool
@traceable(name="analyze_complexity", tags=["review", "metrics"])
def analyze_complexity(
    code: str,
    language: str = "python",
) -> str:
    """Analyze code complexity metrics.

    Calculates:
    - Cyclomatic complexity
    - Cognitive complexity
    - Lines of code (LOC)
    - Comment ratio
    - Function length

    Args:
        code: Code to analyze.
        language: Programming language.

    Returns:
        JSON string with complexity metrics.
    """
    lines = code.split("\n")
    total_lines = len(lines)
    code_lines = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
    comment_lines = len([l for l in lines if l.strip().startswith("#")])
    blank_lines = len([l for l in lines if not l.strip()])

    # Calculate cyclomatic complexity (simplified)
    complexity_keywords = ["if", "elif", "for", "while", "except", "and", "or"]
    cyclomatic_complexity = 1  # Base complexity
    for line in lines:
        for keyword in complexity_keywords:
            if re.search(rf'\b{keyword}\b', line):
                cyclomatic_complexity += 1

    # Find functions and their lengths
    functions = []
    current_func = None
    func_start = 0

    for i, line in enumerate(lines):
        if line.strip().startswith("def ") or line.strip().startswith("async def "):
            if current_func:
                functions.append({
                    "name": current_func,
                    "start_line": func_start,
                    "end_line": i,
                    "length": i - func_start,
                })
            current_func = line.strip().split("(")[0].replace("def ", "").replace("async ", "")
            func_start = i

    if current_func:
        functions.append({
            "name": current_func,
            "start_line": func_start,
            "end_line": total_lines,
            "length": total_lines - func_start,
        })

    # Complexity rating
    if cyclomatic_complexity <= 5:
        complexity_rating = "low"
    elif cyclomatic_complexity <= 10:
        complexity_rating = "moderate"
    elif cyclomatic_complexity <= 20:
        complexity_rating = "high"
    else:
        complexity_rating = "very_high"

    result = {
        "metrics": {
            "total_lines": total_lines,
            "code_lines": code_lines,
            "comment_lines": comment_lines,
            "blank_lines": blank_lines,
            "comment_ratio": round(comment_lines / max(code_lines, 1) * 100, 1),
            "cyclomatic_complexity": cyclomatic_complexity,
            "complexity_rating": complexity_rating,
            "function_count": len(functions),
            "avg_function_length": round(sum(f["length"] for f in functions) / max(len(functions), 1), 1),
        },
        "functions": functions,
        "recommendations": [],
    }

    # Add recommendations
    if cyclomatic_complexity > 10:
        result["recommendations"].append("Consider breaking down complex logic into smaller functions")
    if result["metrics"]["comment_ratio"] < 10:
        result["recommendations"].append("Consider adding more documentation comments")
    for func in functions:
        if func["length"] > 50:
            result["recommendations"].append(f"Function '{func['name']}' is too long ({func['length']} lines)")

    return json.dumps(result, indent=2)


@tool
@traceable(name="detect_code_smells", tags=["review", "quality"])
def detect_code_smells(
    code: str,
    language: str = "python",
) -> str:
    """Detect code smells and anti-patterns.

    Detects:
    - Long methods
    - Large classes
    - Feature envy
    - Data clumps
    - Dead code
    - Duplicated code

    Args:
        code: Code to analyze.
        language: Programming language.

    Returns:
        JSON string with detected code smells.
    """
    smells = []
    lines = code.split("\n")

    # Long method detection
    in_function = False
    func_start = 0
    func_name = ""

    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            if in_function and (i - func_start) > 30:
                smells.append({
                    "type": "long_method",
                    "severity": "medium",
                    "location": f"line {func_start}",
                    "description": f"Method '{func_name}' is too long ({i - func_start} lines)",
                    "suggestion": "Extract parts into smaller, focused methods",
                })
            in_function = True
            func_start = i
            func_name = line.strip().split("(")[0].replace("def ", "")

    # Duplicated code (simplified)
    line_counts = {}
    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) > 20:  # Only count significant lines
            line_counts[stripped] = line_counts.get(stripped, 0) + 1

    for line_text, count in line_counts.items():
        if count >= 3:
            smells.append({
                "type": "duplicated_code",
                "severity": "medium",
                "description": f"Code appears {count} times: '{line_text[:50]}...'",
                "suggestion": "Extract to a reusable function or constant",
            })

    # Feature envy (simplified - check for excessive external method calls)
    external_calls = re.findall(r'\b(\w+)\.\w+\(', code)
    call_counts = {}
    for obj in external_calls:
        if obj not in ["self", "cls"]:
            call_counts[obj] = call_counts.get(obj, 0) + 1

    for obj, count in call_counts.items():
        if count >= 5:
            smells.append({
                "type": "feature_envy",
                "severity": "low",
                "description": f"Excessive use of '{obj}' object ({count} calls)",
                "suggestion": "Consider moving this logic to the related class",
            })

    # Dead code detection (simplified)
    for i, line in enumerate(lines, 1):
        if "# DEAD CODE" in line.upper() or "# UNUSED" in line.upper():
            smells.append({
                "type": "dead_code",
                "severity": "low",
                "location": f"line {i}",
                "description": "Commented-out or unused code detected",
                "suggestion": "Remove dead code or explain why it's kept",
            })

    result = {
        "total_smells": len(smells),
        "by_type": {},
        "smells": smells,
        "overall_health": "healthy" if len(smells) < 3 else "needs_attention" if len(smells) < 7 else "unhealthy",
    }

    for smell in smells:
        t = smell["type"]
        result["by_type"][t] = result["by_type"].get(t, 0) + 1

    return json.dumps(result, indent=2)


@tool
@traceable(name="suggest_improvements", tags=["review", "refactoring"])
def suggest_improvements(
    code: str,
    review_id: str | None = None,
    language: str = "python",
) -> str:
    """Suggest improvements for code.

    Based on review results or direct analysis,
    provides actionable improvement suggestions.

    Args:
        code: Code to improve.
        review_id: Optional review ID to base suggestions on.
        language: Programming language.

    Returns:
        JSON string with improvement suggestions.
    """
    suggestions = []

    # Get review if available
    review = _review_store.get(review_id) if review_id else None

    # Analyze code for improvement opportunities
    lines = code.split("\n")

    # Check for type hints (Python)
    if language == "python":
        for i, line in enumerate(lines, 1):
            if line.strip().startswith("def ") and "->" not in line:
                suggestions.append({
                    "type": "type_hints",
                    "priority": "medium",
                    "location": f"line {i}",
                    "suggestion": "Add return type annotation",
                    "example": "def func(arg: str) -> str:",
                })
            if re.match(r'\s*def \w+\([^:)]+\)', line):
                suggestions.append({
                    "type": "type_hints",
                    "priority": "medium",
                    "location": f"line {i}",
                    "suggestion": "Add parameter type annotations",
                })

    # Check for error handling
    if "try:" not in code and len(lines) > 20:
        suggestions.append({
            "type": "error_handling",
            "priority": "high",
            "suggestion": "Add error handling for potential failure points",
        })

    # Check for logging
    if "logging" not in code and "logger" not in code and len(lines) > 30:
        suggestions.append({
            "type": "logging",
            "priority": "medium",
            "suggestion": "Add logging for better observability",
        })

    # Check for constants
    string_literals = re.findall(r'["\'][^"\']{10,}["\']', code)
    if len(string_literals) > 3:
        suggestions.append({
            "type": "constants",
            "priority": "low",
            "suggestion": "Consider extracting repeated strings to constants",
        })

    # Based on review results
    if review:
        for issue in review.get("issues", []):
            if issue["severity"] in ["critical", "high"]:
                suggestions.append({
                    "type": "from_review",
                    "priority": "high",
                    "location": f"line {issue.get('line', 'unknown')}",
                    "suggestion": issue.get("suggestion", "Fix the identified issue"),
                })

    result = {
        "total_suggestions": len(suggestions),
        "by_priority": {
            "high": len([s for s in suggestions if s["priority"] == "high"]),
            "medium": len([s for s in suggestions if s["priority"] == "medium"]),
            "low": len([s for s in suggestions if s["priority"] == "low"]),
        },
        "suggestions": suggestions,
        "next_steps": [
            "Address high priority items first",
            "Run tests after each change",
            "Request peer review for significant changes",
        ],
    }

    return json.dumps(result, indent=2)


@tool
@traceable(name="check_best_practices", tags=["review", "best_practices"])
def check_best_practices(
    code: str,
    language: str = "python",
    framework: str | None = None,
) -> str:
    """Check code against language/framework best practices.

    Args:
        code: Code to check.
        language: Programming language.
        framework: Optional framework (fastapi, django, express, react).

    Returns:
        JSON string with best practice violations.
    """
    violations = []
    compliances = []

    if language == "python":
        # Check for docstrings
        if '"""' in code or "'''" in code:
            compliances.append("Uses docstrings for documentation")
        else:
            violations.append({
                "practice": "documentation",
                "message": "Missing docstrings",
                "recommendation": "Add docstrings to modules, classes, and functions",
            })

        # Check for type hints
        if "->" in code and ": " in code:
            compliances.append("Uses type annotations")
        else:
            violations.append({
                "practice": "type_safety",
                "message": "Missing type annotations",
                "recommendation": "Add type hints for better IDE support and documentation",
            })

        # Check for context managers
        if "with " in code:
            compliances.append("Uses context managers for resource handling")

        # Check for f-strings vs % formatting
        if "%" in code and '"%' in code:
            violations.append({
                "practice": "string_formatting",
                "message": "Using old-style % formatting",
                "recommendation": "Use f-strings for better readability",
            })

        if framework == "fastapi":
            # FastAPI specific checks
            if "async def" in code:
                compliances.append("Uses async functions (FastAPI best practice)")
            if "Depends" in code:
                compliances.append("Uses dependency injection")
            if "HTTPException" in code or "status_code" in code:
                compliances.append("Uses proper HTTP status codes")

    result = {
        "language": language,
        "framework": framework,
        "violations_count": len(violations),
        "compliances_count": len(compliances),
        "violations": violations,
        "compliances": compliances,
        "score": round(len(compliances) / max(len(violations) + len(compliances), 1) * 100),
    }

    return json.dumps(result, indent=2)
