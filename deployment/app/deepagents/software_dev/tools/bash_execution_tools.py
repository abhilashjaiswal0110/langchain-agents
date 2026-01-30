"""
Bash Execution Tools for Software Development Agent

Provides secure command execution capabilities with:
- Multi-platform support (Bash, PowerShell, Python)
- Security validation and dangerous command detection
- Fallback mechanisms for cross-platform compatibility
- Command history tracking
- Integration with terminal and UI interfaces

Inspired by Claude Code's hook-based security system.
"""

import os
import platform
import re
import subprocess
import sys
from typing import Any, Literal
from langchain_core.tools import tool
from langsmith import traceable


# Security patterns for dangerous commands
DANGEROUS_PATTERNS = [
    (r"rm\s+-rf\s+/", "Attempting to delete root directory"),
    (r":\(\)\{\s*:\|:&\s*\};:", "Fork bomb detected"),
    (r"dd\s+if=.*\s+of=/dev/", "Attempting to write to device"),
    (r"mkfs\.", "Attempting to format filesystem"),
    (r":(){ :|:& };:", "Bash fork bomb detected"),
    (r"mv\s+.*\s+/dev/null", "Moving files to /dev/null"),
    (r">\s*/dev/sda", "Writing to disk device"),
    (r"chmod\s+-R\s+777\s+/", "Setting dangerous permissions on root"),
]

# Warning patterns for risky commands
WARNING_PATTERNS = [
    (r"rm\s+-rf", "Recursive force delete"),
    (r"sudo\s+", "Elevated privileges"),
    (r"curl.*\|\s*bash", "Piping web content to bash"),
    (r"wget.*\|\s*sh", "Piping web content to shell"),
    (r"eval\s*\(", "Using eval() which can be dangerous"),
    (r"exec\s*\(", "Using exec() which can be dangerous"),
]


def detect_dangerous_command(command: str) -> tuple[bool, str | None]:
    """
    Detect dangerous commands that should be blocked.

    Args:
        command: The command to validate.

    Returns:
        Tuple of (is_dangerous, reason). If is_dangerous is True, reason explains why.
    """
    command_lower = command.lower().strip()

    # Check dangerous patterns
    for pattern, reason in DANGEROUS_PATTERNS:
        if re.search(pattern, command_lower, re.IGNORECASE):
            return True, reason

    return False, None


def detect_risky_command(command: str) -> tuple[bool, str | None]:
    """
    Detect risky commands that should show warnings.

    Args:
        command: The command to validate.

    Returns:
        Tuple of (is_risky, warning). If is_risky is True, warning explains the risk.
    """
    command_lower = command.lower().strip()

    # Check warning patterns
    for pattern, warning in WARNING_PATTERNS:
        if re.search(pattern, command_lower, re.IGNORECASE):
            return True, warning

    return False, None


def get_shell_type() -> Literal["bash", "powershell", "cmd"]:
    """
    Determine the appropriate shell type for the current platform.

    Returns:
        Shell type: "bash" for Unix-like systems, "powershell" for Windows.
    """
    system = platform.system()
    if system in ["Linux", "Darwin"]:  # macOS is Darwin
        return "bash"
    elif system == "Windows":
        # Check if PowerShell is available
        try:
            subprocess.run(
                ["powershell", "-Command", "echo test"],
                capture_output=True,
                timeout=2,
                check=True,
            )
            return "powershell"
        except (subprocess.SubprocessError, FileNotFoundError):
            return "cmd"
    return "bash"


@tool
@traceable(name="execute_bash_command", tags=["bash", "execution", "sdlc"])
def execute_bash_command(
    command: str,
    *,
    timeout: int = 30,
    working_directory: str | None = None,
) -> dict[str, Any]:
    """
    Execute a bash command with security validation and cross-platform support.

    This tool executes shell commands for development tasks like:
    - Running tests (pytest, npm test, cargo test)
    - Building projects (npm run build, cargo build, go build)
    - Installing dependencies (pip install, npm install)
    - Git operations (git status, git commit, git push)
    - File operations (ls, cat, grep - prefer specialized tools for these)
    - Running scripts (python script.py, node script.js)

    Security features:
    - Blocks dangerous commands (rm -rf /, fork bombs, etc.)
    - Warns about risky operations (sudo, recursive deletes)
    - Platform-aware execution (Bash on Linux/Mac, PowerShell on Windows)
    - Timeout protection (default 30s)
    - Command validation before execution

    Args:
        command: The shell command to execute.
        timeout: Maximum execution time in seconds.
        working_directory: Directory to execute the command in (optional).

    Returns:
        Dictionary containing:
        - success: Whether command executed successfully
        - stdout: Standard output from the command
        - stderr: Standard error from the command
        - exit_code: Process exit code
        - command: The executed command
        - shell_type: Which shell was used (bash, powershell, cmd)
        - warning: Any security warnings (optional)

    Raises:
        ValueError: If the command is dangerous or invalid.

    Examples:
        >>> execute_bash_command(command="pytest tests/")
        >>> execute_bash_command(command="npm run build", timeout=60)
        >>> execute_bash_command(command="git status", working_directory="./myproject")
    """
    # Security validation
    is_dangerous, danger_reason = detect_dangerous_command(command)
    if is_dangerous:
        raise ValueError(
            f"🚫 Dangerous command blocked: {danger_reason}\n"
            f"Command: {command}\n"
            f"This command has been prevented for security reasons."
        )

    # Check for risky patterns
    is_risky, risk_warning = detect_risky_command(command)
    warning = None
    if is_risky:
        warning = f"⚠️ Warning: {risk_warning}"

    # Determine shell type
    shell_type = get_shell_type()

    # Prepare command based on shell type
    if shell_type == "bash":
        cmd_list = ["bash", "-c", command]
    elif shell_type == "powershell":
        cmd_list = ["powershell", "-Command", command]
    else:  # cmd
        cmd_list = ["cmd", "/c", command]

    # Set working directory
    cwd = working_directory or os.getcwd()

    try:
        # Execute command
        result = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            check=False,  # Don't raise on non-zero exit
        )

        # Prepare response
        response = {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "command": command,
            "shell_type": shell_type,
        }

        if warning:
            response["warning"] = warning

        return response

    except subprocess.TimeoutExpired as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "exit_code": -1,
            "command": command,
            "shell_type": shell_type,
            "error": "timeout",
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "exit_code": -1,
            "command": command,
            "shell_type": shell_type,
            "error": type(e).__name__,
        }


@tool
@traceable(name="execute_python_code", tags=["python", "execution", "sdlc"])
def execute_python_code(
    code: str,
    *,
    timeout: int = 30,
    working_directory: str | None = None,
) -> dict[str, Any]:
    """
    Execute Python code directly with security validation.

    This tool runs Python code snippets for development tasks like:
    - Running Python scripts
    - Testing Python functions
    - Data processing and analysis
    - Quick calculations and validations

    Security features:
    - Blocks dangerous operations (eval, exec on untrusted input)
    - Timeout protection (default 30s)
    - Isolated execution context

    Args:
        code: The Python code to execute.
        timeout: Maximum execution time in seconds.
        working_directory: Directory to execute the code in (optional).

    Returns:
        Dictionary containing:
        - success: Whether code executed successfully
        - stdout: Standard output from the code
        - stderr: Standard error from the code
        - exit_code: Process exit code
        - code: The executed code (first 200 chars)

    Examples:
        >>> execute_python_code(code="print('Hello, World!')")
        >>> execute_python_code(code="import sys; print(sys.version)")
    """
    # Basic security check
    if "eval(" in code or "exec(" in code:
        return {
            "success": False,
            "stdout": "",
            "stderr": "⚠️ Warning: eval() and exec() are disabled for security",
            "exit_code": -1,
            "code": code[:200],
        }

    # Set working directory
    cwd = working_directory or os.getcwd()

    try:
        # Execute Python code
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            check=False,
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "code": code[:200],  # Only show first 200 chars
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Python code timed out after {timeout} seconds",
            "exit_code": -1,
            "code": code[:200],
            "error": "timeout",
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "exit_code": -1,
            "code": code[:200],
            "error": type(e).__name__,
        }


@tool
@traceable(name="execute_tests_real", tags=["testing", "execution", "sdlc"])
def execute_tests_real(
    test_framework: Literal["pytest", "unittest", "jest", "mocha", "cargo", "go"] = "pytest",
    *,
    test_path: str = "tests/",
    additional_args: str = "",
    timeout: int = 120,
) -> dict[str, Any]:
    """
    Run tests using the specified testing framework.

    This is a specialized tool for running test suites with common frameworks.

    Supported frameworks:
    - pytest: Python testing (pytest tests/)
    - unittest: Python unittest (python -m unittest discover)
    - jest: JavaScript/TypeScript (npm test or jest)
    - mocha: JavaScript (npx mocha)
    - cargo: Rust (cargo test)
    - go: Go (go test)

    Args:
        test_framework: The testing framework to use.
        test_path: Path to tests directory or test files.
        additional_args: Additional arguments to pass to the test command.
        timeout: Maximum execution time in seconds (default 120s for tests).

    Returns:
        Dictionary with test execution results.

    Examples:
        >>> execute_tests_real(test_framework="pytest", test_path="tests/unit/")
        >>> execute_tests_real(test_framework="jest", additional_args="--coverage")
        >>> execute_tests_real(test_framework="cargo", additional_args="--release")
    """
    # Build command based on framework
    commands = {
        "pytest": f"pytest {test_path} {additional_args}",
        "unittest": f"python -m unittest discover {test_path} {additional_args}",
        "jest": f"npm test {additional_args}" if not test_path or test_path == "tests/" else f"jest {test_path} {additional_args}",
        "mocha": f"npx mocha {test_path} {additional_args}",
        "cargo": f"cargo test {additional_args}",
        "go": f"go test {test_path} {additional_args}",
    }

    command = commands.get(test_framework, f"{test_framework} {test_path} {additional_args}")

    # Use the execute_bash_command tool
    result = execute_bash_command.invoke({"command": command.strip(), "timeout": timeout})

    # Add test-specific information
    result["test_framework"] = test_framework
    result["test_path"] = test_path

    return result


@tool
@traceable(name="install_dependencies", tags=["dependencies", "execution", "sdlc"])
def install_dependencies(
    package_manager: Literal["pip", "npm", "yarn", "cargo", "go"] = "pip",
    *,
    packages: str = "",
    requirements_file: str = "",
    timeout: int = 300,
) -> dict[str, Any]:
    """
    Install project dependencies using the specified package manager.

    Supported package managers:
    - pip: Python (pip install -r requirements.txt)
    - npm: JavaScript/TypeScript (npm install)
    - yarn: JavaScript/TypeScript (yarn install)
    - cargo: Rust (cargo build)
    - go: Go (go mod download)

    Args:
        package_manager: The package manager to use.
        packages: Specific packages to install (space-separated).
        requirements_file: Path to requirements/dependencies file.
        timeout: Maximum execution time in seconds (default 300s).

    Returns:
        Dictionary with installation results.

    Examples:
        >>> install_dependencies(package_manager="pip", requirements_file="requirements.txt")
        >>> install_dependencies(package_manager="npm")
        >>> install_dependencies(package_manager="pip", packages="pytest black ruff")
    """
    # Build command based on package manager
    if packages:
        commands = {
            "pip": f"pip install {packages}",
            "npm": f"npm install {packages}",
            "yarn": f"yarn add {packages}",
            "cargo": f"cargo add {packages}",
            "go": f"go get {packages}",
        }
    elif requirements_file:
        commands = {
            "pip": f"pip install -r {requirements_file}",
            "npm": "npm install",
            "yarn": "yarn install",
            "cargo": "cargo build",
            "go": "go mod download",
        }
    else:
        commands = {
            "pip": "pip install -r requirements.txt",
            "npm": "npm install",
            "yarn": "yarn install",
            "cargo": "cargo build",
            "go": "go mod download",
        }

    command = commands.get(package_manager, f"{package_manager} install")

    # Use the execute_bash_command tool
    result = execute_bash_command.invoke({"command": command.strip(), "timeout": timeout})

    # Add dependency-specific information
    result["package_manager"] = package_manager
    if packages:
        result["packages"] = packages
    if requirements_file:
        result["requirements_file"] = requirements_file

    return result


# Export all tools
__all__ = [
    "execute_bash_command",
    "execute_python_code",
    "execute_tests_real",
    "install_dependencies",
]
