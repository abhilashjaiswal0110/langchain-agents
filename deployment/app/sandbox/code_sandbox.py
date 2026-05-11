"""Resource-limited code execution sandbox for Python snippets.

Runs untrusted code in a subprocess with:
- Configurable timeout (default 30 s)
- Temporary working directory (auto-cleaned)
- Restricted environment variables (no API keys, no HOME secrets)
- Memory limits via resource.setrlimit on Linux/macOS
- Process count limits on Linux/macOS

On Windows the resource-limit feature is silently skipped; timeout and
environment isolation still apply.

Usage:
    from app.sandbox.code_sandbox import CodeSandbox

    sandbox = CodeSandbox(timeout=10, memory_mb=128)
    result = sandbox.execute("print('hello')")
    print(result.stdout)
"""

import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_IS_POSIX = sys.platform != "win32"


@dataclass
class SandboxResult:
    """Result of a sandbox execution.

    Attributes:
        stdout: Captured standard output.
        stderr: Captured standard error.
        returncode: Process exit code.
        timed_out: Whether the process exceeded the timeout.
    """

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


def _make_restricted_env() -> dict[str, str]:
    """Build a minimal environment for the sandboxed process.

    Strips sensitive keys (API keys, tokens, passwords) and passes only
    the PATH and language settings needed to run Python.

    Returns:
        Restricted environment dict.
    """
    safe_keys = {"PATH", "SYSTEMROOT", "TEMP", "TMP", "LANG", "LC_ALL"}
    env = {k: v for k, v in os.environ.items() if k in safe_keys}
    # Override HOME to the system temp directory so sandboxed code cannot read
    # the real user's home directory or sensitive dotfiles.
    env["HOME"] = os.path.join(os.environ.get("TEMP", os.environ.get("TMP", "/tmp")))
    return env


class CodeSandbox:
    """Executes Python code snippets in an isolated subprocess.

    Args:
        timeout: Maximum execution time in seconds.
        memory_mb: Memory limit in megabytes (Linux/macOS only).
        max_processes: Maximum number of child processes (Linux/macOS only).
    """

    def __init__(
        self,
        timeout: int = int(os.getenv("SANDBOX_TIMEOUT", "30")),
        memory_mb: int = int(os.getenv("SANDBOX_MEMORY_MB", "256")),
        max_processes: int = 50,
    ) -> None:
        self._timeout = timeout
        self._memory_bytes = memory_mb * 1024 * 1024
        self._max_processes = max_processes

    def _preexec(self) -> None:
        """Apply resource limits before exec (POSIX only)."""
        try:
            import resource  # type: ignore[import]

            resource.setrlimit(
                resource.RLIMIT_AS,
                (self._memory_bytes, self._memory_bytes),
            )
            resource.setrlimit(
                resource.RLIMIT_NPROC,
                (self._max_processes, self._max_processes),
            )
        except Exception as exc:
            logger.debug("Could not set resource limits: %s", exc)

    def execute(self, code: str, language: str = "python") -> SandboxResult:
        """Execute *code* in a sandboxed subprocess.

        Args:
            code: Python source code to execute.
            language: Reserved for future language support (currently only
                ``"python"`` is supported).

        Returns:
            SandboxResult with stdout, stderr, returncode, and timed_out.
        """
        with tempfile.TemporaryDirectory(prefix="sandbox_") as tmpdir:
            script_path = os.path.join(tmpdir, "script.py")
            with open(script_path, "w", encoding="utf-8") as fh:
                fh.write(code)

            # Use subprocess.run with a fixed argument list (no shell=True)
            # to avoid shell injection vulnerabilities.
            kwargs: dict = {
                "capture_output": True,
                "text": True,
                "timeout": self._timeout,
                "env": _make_restricted_env(),
                "cwd": tmpdir,
            }
            if _IS_POSIX:
                kwargs["preexec_fn"] = self._preexec

            try:
                proc = subprocess.run(
                    [sys.executable, script_path],
                    **kwargs,
                )
                return SandboxResult(
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    returncode=proc.returncode,
                )
            except subprocess.TimeoutExpired as exc:
                logger.warning("Sandbox execution timed out after %ds", self._timeout)
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors="replace")
                return SandboxResult(
                    stdout=stdout,
                    stderr=stderr,
                    returncode=-1,
                    timed_out=True,
                )
            except Exception as exc:
                logger.error("Sandbox execution error: %s", exc)
                return SandboxResult(
                    stdout="",
                    stderr=str(exc),
                    returncode=-1,
                )
