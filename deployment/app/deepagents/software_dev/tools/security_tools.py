"""Security & Compliance Tools.

Tools for security scanning, vulnerability detection, and compliance checking.
"""

import json
import re
import uuid
from datetime import datetime

from langchain_core.tools import tool
from langsmith import traceable

from app.deepagents.config.software_dev_config import (
    SecuritySeverity,
    OWASP_TOP_10,
)


# Session storage
_security_store: dict[str, dict] = {}


@tool
@traceable(name="scan_security_issues", tags=["security", "scanning"])
def scan_security_issues(
    code: str,
    language: str = "python",
    scan_level: str = "standard",
    session_id: str = "default",
) -> str:
    """Scan code for security vulnerabilities.

    Detects:
    - Hardcoded credentials
    - SQL injection vulnerabilities
    - XSS vulnerabilities
    - Insecure deserialization
    - Command injection

    Args:
        code: Code to scan.
        language: Programming language.
        scan_level: Scan depth (quick, standard, deep).
        session_id: Session identifier.

    Returns:
        JSON string with security scan results.
    """
    scan_id = f"SEC-{str(uuid.uuid4())[:8].upper()}"
    issues = []

    lines = code.split("\n")

    for i, line in enumerate(lines, 1):
        # Hardcoded credentials
        if re.search(r'(password|secret|api_key|token|credential)\s*=\s*["\'][^"\']+["\']', line, re.I):
            issues.append({
                "id": f"{scan_id}-{len(issues)+1}",
                "severity": SecuritySeverity.CRITICAL.value,
                "category": "A02",
                "title": "Hardcoded Credential",
                "description": "Hardcoded secret detected in source code",
                "line": i,
                "cwe_id": "CWE-798",
                "recommendation": "Use environment variables or a secrets manager",
            })

        # SQL Injection
        if re.search(r'(execute|query|cursor\.execute)\s*\([^)]*(%s|{|}|f["\'])', line, re.I):
            issues.append({
                "id": f"{scan_id}-{len(issues)+1}",
                "severity": SecuritySeverity.HIGH.value,
                "category": "A03",
                "title": "Potential SQL Injection",
                "description": "User input may be directly concatenated into SQL query",
                "line": i,
                "cwe_id": "CWE-89",
                "recommendation": "Use parameterized queries or ORM",
            })

        # Command Injection
        if re.search(r'(os\.system|subprocess\.call|subprocess\.run|exec|eval)\s*\([^)]*(\+|%|f["\']|\.format)', line, re.I):
            issues.append({
                "id": f"{scan_id}-{len(issues)+1}",
                "severity": SecuritySeverity.CRITICAL.value,
                "category": "A03",
                "title": "Potential Command Injection",
                "description": "User input may be passed to shell command",
                "line": i,
                "cwe_id": "CWE-78",
                "recommendation": "Avoid shell commands with user input; use subprocess with shell=False",
            })

        # Insecure deserialization
        if re.search(r'(pickle\.loads?|yaml\.load\s*\([^)]*\)(?!.*Loader))', line, re.I):
            issues.append({
                "id": f"{scan_id}-{len(issues)+1}",
                "severity": SecuritySeverity.HIGH.value,
                "category": "A08",
                "title": "Insecure Deserialization",
                "description": "Unsafe deserialization of untrusted data",
                "line": i,
                "cwe_id": "CWE-502",
                "recommendation": "Use safe serialization formats like JSON; use yaml.safe_load()",
            })

        # Weak cryptography
        if re.search(r'(md5|sha1)\s*\(', line, re.I):
            issues.append({
                "id": f"{scan_id}-{len(issues)+1}",
                "severity": SecuritySeverity.MEDIUM.value,
                "category": "A02",
                "title": "Weak Cryptographic Hash",
                "description": "MD5 or SHA1 are not recommended for security purposes",
                "line": i,
                "cwe_id": "CWE-327",
                "recommendation": "Use SHA-256 or stronger for security; use bcrypt for passwords",
            })

        # XSS (simplified check for template rendering)
        if re.search(r'(render_template_string|innerHTML|\.html\()', line, re.I):
            issues.append({
                "id": f"{scan_id}-{len(issues)+1}",
                "severity": SecuritySeverity.HIGH.value,
                "category": "A03",
                "title": "Potential XSS Vulnerability",
                "description": "Unsanitized user input may be rendered as HTML",
                "line": i,
                "cwe_id": "CWE-79",
                "recommendation": "Escape user input before rendering; use auto-escaping templates",
            })

        # Insecure random
        if re.search(r'random\.(random|randint|choice)\s*\(', line) and "secret" in code.lower():
            issues.append({
                "id": f"{scan_id}-{len(issues)+1}",
                "severity": SecuritySeverity.MEDIUM.value,
                "category": "A02",
                "title": "Insecure Random Number Generator",
                "description": "Using non-cryptographic random for security purpose",
                "line": i,
                "cwe_id": "CWE-330",
                "recommendation": "Use secrets module for cryptographic purposes",
            })

    # Calculate risk score
    severity_weights = {
        SecuritySeverity.CRITICAL.value: 10,
        SecuritySeverity.HIGH.value: 7,
        SecuritySeverity.MEDIUM.value: 4,
        SecuritySeverity.LOW.value: 1,
    }
    risk_score = sum(severity_weights.get(i["severity"], 0) for i in issues)

    result = {
        "id": scan_id,
        "language": language,
        "scan_level": scan_level,
        "lines_scanned": len(lines),
        "issues_found": len(issues),
        "risk_score": risk_score,
        "risk_level": "critical" if risk_score > 20 else "high" if risk_score > 10 else "medium" if risk_score > 5 else "low",
        "summary": {
            "critical": len([i for i in issues if i["severity"] == SecuritySeverity.CRITICAL.value]),
            "high": len([i for i in issues if i["severity"] == SecuritySeverity.HIGH.value]),
            "medium": len([i for i in issues if i["severity"] == SecuritySeverity.MEDIUM.value]),
            "low": len([i for i in issues if i["severity"] == SecuritySeverity.LOW.value]),
        },
        "issues": issues,
        "scanned_at": datetime.now().isoformat(),
    }

    _security_store[scan_id] = result

    return json.dumps(result, indent=2)


@tool
@traceable(name="check_owasp_compliance", tags=["security", "compliance"])
def check_owasp_compliance(
    code: str,
    categories: list[str] | None = None,
    language: str = "python",
) -> str:
    """Check code against OWASP Top 10.

    Validates code against OWASP Top 10 2021 categories:
    A01-A10 security controls.

    Args:
        code: Code to check.
        categories: Specific OWASP categories to check (e.g., ["A01", "A03"]).
        language: Programming language.

    Returns:
        JSON string with OWASP compliance results.
    """
    categories = categories or list(OWASP_TOP_10.keys())

    compliance_results = []

    checks = {
        "A01": {  # Broken Access Control
            "patterns": [r'@require_login|@login_required|authorize|permission', r'RBAC|role\s*check'],
            "description": "Access control mechanisms",
        },
        "A02": {  # Cryptographic Failures
            "patterns": [r'bcrypt|argon2|sha256|sha512|secrets\.', r'HTTPS|TLS|encrypt'],
            "antipatterns": [r'md5|sha1|base64.*password'],
            "description": "Cryptographic implementations",
        },
        "A03": {  # Injection
            "patterns": [r'parameterized|prepared_statement|bindparam|sanitize|escape'],
            "antipatterns": [r'execute.*%s|f["\'].*{.*}.*(?:select|delete|update|insert)'],
            "description": "Injection prevention",
        },
        "A04": {  # Insecure Design
            "patterns": [r'validate|schema|type\s*check|assert', r'unittest|pytest|test_'],
            "description": "Secure design patterns",
        },
        "A05": {  # Security Misconfiguration
            "patterns": [r'DEBUG\s*=\s*False|production|secure_headers'],
            "antipatterns": [r'DEBUG\s*=\s*True|allow_all|disabled?.*security'],
            "description": "Security configuration",
        },
        "A06": {  # Vulnerable Components
            "patterns": [r'requirements.*txt|package.*json|Pipfile'],
            "description": "Dependency management",
        },
        "A07": {  # Authentication Failures
            "patterns": [r'password.*hash|bcrypt|session.*secure|csrf|rate.*limit'],
            "description": "Authentication security",
        },
        "A08": {  # Software Integrity
            "patterns": [r'verify.*signature|checksum|integrity', r'json\.loads|yaml\.safe_load'],
            "antipatterns": [r'pickle\.load|eval\(|exec\('],
            "description": "Data integrity controls",
        },
        "A09": {  # Logging Failures
            "patterns": [r'logging\.|logger\.|audit|monitor'],
            "description": "Security logging",
        },
        "A10": {  # SSRF
            "patterns": [r'whitelist|allowlist|validate.*url'],
            "antipatterns": [r'requests\.get\s*\(\s*(?:request\.|user)'],
            "description": "SSRF prevention",
        },
    }

    for cat in categories:
        if cat not in checks:
            continue

        check = checks[cat]
        status = "unknown"
        findings = []

        # Check for positive patterns
        has_controls = False
        for pattern in check.get("patterns", []):
            if re.search(pattern, code, re.I):
                has_controls = True
                findings.append(f"Found security control: {pattern[:30]}...")

        # Check for anti-patterns
        has_violations = False
        for pattern in check.get("antipatterns", []):
            if re.search(pattern, code, re.I):
                has_violations = True
                findings.append(f"Found potential issue: {pattern[:30]}...")

        if has_controls and not has_violations:
            status = "compliant"
        elif has_violations:
            status = "non_compliant"
        elif has_controls:
            status = "partial"
        else:
            status = "not_assessed"

        compliance_results.append({
            "category": cat,
            "name": OWASP_TOP_10.get(cat, "Unknown"),
            "description": check["description"],
            "status": status,
            "findings": findings,
        })

    compliant_count = len([r for r in compliance_results if r["status"] == "compliant"])
    total = len(compliance_results)

    result = {
        "framework": "OWASP Top 10 (2021)",
        "categories_checked": len(compliance_results),
        "compliant": compliant_count,
        "non_compliant": len([r for r in compliance_results if r["status"] == "non_compliant"]),
        "compliance_score": round(compliant_count / max(total, 1) * 100, 1),
        "results": compliance_results,
        "recommendations": [
            "Address non-compliant categories first",
            "Implement missing security controls",
            "Consider security code review",
        ],
    }

    return json.dumps(result, indent=2)


@tool
@traceable(name="detect_secrets", tags=["security", "secrets"])
def detect_secrets(
    code: str,
    file_path: str | None = None,
) -> str:
    """Detect secrets and credentials in code.

    Scans for:
    - API keys
    - Passwords
    - Tokens
    - Private keys
    - Connection strings

    Args:
        code: Code to scan.
        file_path: Optional file path for context.

    Returns:
        JSON string with detected secrets.
    """
    secrets = []

    patterns = {
        "api_key": (r'["\']?(?:api[_-]?key|apikey)["\']?\s*[:=]\s*["\']([^"\']{10,})["\']', SecuritySeverity.CRITICAL),
        "aws_key": (r'AKIA[0-9A-Z]{16}', SecuritySeverity.CRITICAL),
        "password": (r'["\']?password["\']?\s*[:=]\s*["\']([^"\']+)["\']', SecuritySeverity.CRITICAL),
        "token": (r'["\']?(?:token|bearer)["\']?\s*[:=]\s*["\']([^"\']{20,})["\']', SecuritySeverity.HIGH),
        "private_key": (r'-----BEGIN (?:RSA |EC )?PRIVATE KEY-----', SecuritySeverity.CRITICAL),
        "connection_string": (r'(?:mongodb|postgresql|mysql|redis)://[^\s"\']+', SecuritySeverity.HIGH),
        "jwt": (r'eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*', SecuritySeverity.HIGH),
        "github_token": (r'ghp_[a-zA-Z0-9]{36}', SecuritySeverity.CRITICAL),
        "slack_token": (r'xox[baprs]-[a-zA-Z0-9-]+', SecuritySeverity.HIGH),
    }

    lines = code.split("\n")

    for i, line in enumerate(lines, 1):
        for secret_type, (pattern, severity) in patterns.items():
            matches = re.findall(pattern, line, re.I)
            if matches or re.search(pattern, line, re.I):
                # Mask the secret
                masked = re.sub(pattern, lambda m: m.group(0)[:10] + "..." + m.group(0)[-4:] if len(m.group(0)) > 14 else "****", line)

                secrets.append({
                    "type": secret_type,
                    "severity": severity.value,
                    "line": i,
                    "masked_content": masked[:100],
                    "recommendation": f"Remove {secret_type} and use environment variables",
                })

    result = {
        "file_path": file_path,
        "secrets_found": len(secrets),
        "has_critical": any(s["severity"] == SecuritySeverity.CRITICAL.value for s in secrets),
        "secrets": secrets,
        "recommendations": [
            "Use environment variables for all secrets",
            "Add pre-commit hooks to prevent secret commits",
            "Use a secrets manager (AWS Secrets Manager, HashiCorp Vault)",
            "Rotate any exposed credentials immediately",
        ] if secrets else ["No secrets detected - continue following best practices"],
    }

    return json.dumps(result, indent=2)


@tool
@traceable(name="analyze_dependencies_security", tags=["security", "dependencies"])
def analyze_dependencies_security(
    dependencies: list[str],
    check_updates: bool = True,
) -> str:
    """Analyze dependencies for security vulnerabilities.

    Args:
        dependencies: List of dependencies to analyze.
        check_updates: Check for available updates.

    Returns:
        JSON string with dependency security analysis.
    """
    # Simulated vulnerability database
    known_vulnerabilities = {
        "requests": [{"version": "<2.25.0", "severity": "high", "cve": "CVE-2023-XXXX"}],
        "django": [{"version": "<3.2.0", "severity": "critical", "cve": "CVE-2023-YYYY"}],
        "flask": [{"version": "<2.0.0", "severity": "medium", "cve": "CVE-2023-ZZZZ"}],
        "pyyaml": [{"version": "<5.4", "severity": "high", "cve": "CVE-2020-14343"}],
        "pillow": [{"version": "<9.0.0", "severity": "high", "cve": "CVE-2022-XXXX"}],
    }

    results = []

    for dep in dependencies:
        # Parse dependency (name==version format)
        parts = re.split(r'[=<>]+', dep)
        name = parts[0].lower().strip()
        version = parts[1] if len(parts) > 1 else "latest"

        dep_result = {
            "name": name,
            "version": version,
            "vulnerabilities": [],
            "status": "secure",
            "latest_version": None,
        }

        # Check for known vulnerabilities
        if name in known_vulnerabilities:
            dep_result["vulnerabilities"] = known_vulnerabilities[name]
            dep_result["status"] = "vulnerable"

        if check_updates:
            # Simulate checking for updates
            dep_result["latest_version"] = f"{name} (check pypi.org for latest)"

        results.append(dep_result)

    vulnerable_count = len([r for r in results if r["status"] == "vulnerable"])

    result = {
        "total_dependencies": len(dependencies),
        "vulnerable": vulnerable_count,
        "secure": len(dependencies) - vulnerable_count,
        "results": results,
        "recommendations": [
            "Update vulnerable dependencies immediately",
            "Use dependabot or similar for automated updates",
            "Pin dependency versions in production",
            "Run security audits regularly (pip-audit, npm audit)",
        ] if vulnerable_count > 0 else ["All dependencies appear secure"],
    }

    return json.dumps(result, indent=2)


@tool
@traceable(name="generate_security_report", tags=["security", "reporting"])
def generate_security_report(
    scan_id: str | None = None,
    session_id: str = "default",
) -> str:
    """Generate comprehensive security report.

    Combines all security scan results into a report.

    Args:
        scan_id: Specific scan to report on.
        session_id: Session identifier for all scans.

    Returns:
        JSON string with security report.
    """
    # Get scan results
    scan = _security_store.get(scan_id) if scan_id else None

    report = {
        "id": f"SECRPT-{str(uuid.uuid4())[:8].upper()}",
        "generated_at": datetime.now().isoformat(),
        "executive_summary": {
            "overall_risk": "medium",
            "critical_issues": 0,
            "high_issues": 0,
            "recommendations": [],
        },
        "sections": [],
    }

    if scan:
        report["executive_summary"]["critical_issues"] = scan["summary"].get("critical", 0)
        report["executive_summary"]["high_issues"] = scan["summary"].get("high", 0)
        report["executive_summary"]["overall_risk"] = scan.get("risk_level", "medium")

        report["sections"].append({
            "title": "Code Security Scan",
            "scan_id": scan["id"],
            "findings": scan["issues"],
        })

    report["executive_summary"]["recommendations"] = [
        "Address all critical and high severity issues immediately",
        "Implement security code review process",
        "Enable security scanning in CI/CD pipeline",
        "Conduct regular security training for developers",
    ]

    return json.dumps(report, indent=2)


@tool
@traceable(name="suggest_security_fixes", tags=["security", "remediation"])
def suggest_security_fixes(
    issue_id: str | None = None,
    issue_type: str | None = None,
    code_context: str | None = None,
) -> str:
    """Suggest fixes for security issues.

    Args:
        issue_id: Specific issue ID to fix.
        issue_type: Type of security issue.
        code_context: Code context for the fix.

    Returns:
        JSON string with fix suggestions.
    """
    fixes = {
        "sql_injection": {
            "description": "SQL Injection vulnerability",
            "fix": "Use parameterized queries",
            "example_before": 'cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")',
            "example_after": 'cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))',
        },
        "hardcoded_credential": {
            "description": "Hardcoded credential in source code",
            "fix": "Use environment variables",
            "example_before": 'API_KEY = "sk-1234567890"',
            "example_after": 'API_KEY = os.environ.get("API_KEY")',
        },
        "xss": {
            "description": "Cross-Site Scripting vulnerability",
            "fix": "Escape user input before rendering",
            "example_before": 'return f"<div>{user_input}</div>"',
            "example_after": 'from markupsafe import escape\nreturn f"<div>{escape(user_input)}</div>"',
        },
        "command_injection": {
            "description": "Command injection vulnerability",
            "fix": "Use subprocess with shell=False",
            "example_before": 'os.system(f"process {user_input}")',
            "example_after": 'subprocess.run(["process", user_input], shell=False)',
        },
    }

    if issue_type and issue_type.lower().replace(" ", "_") in fixes:
        fix = fixes[issue_type.lower().replace(" ", "_")]
    else:
        fix = {
            "description": "General security issue",
            "fix": "Review and apply security best practices",
            "recommendations": [
                "Validate and sanitize all inputs",
                "Use parameterized queries",
                "Implement proper authentication/authorization",
                "Enable security headers",
            ],
        }

    result = {
        "issue_id": issue_id,
        "issue_type": issue_type,
        **fix,
        "additional_resources": [
            "OWASP Cheat Sheet Series",
            "CWE/SANS Top 25 Most Dangerous Software Errors",
            "Language-specific security guidelines",
        ],
    }

    return json.dumps(result, indent=2)
