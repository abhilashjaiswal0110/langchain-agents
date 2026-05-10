"""Tests for the code execution sandbox."""
import sys

import pytest

from app.sandbox.code_sandbox import CodeSandbox, SandboxResult


class TestCodeSandbox:
    @pytest.fixture()
    def sandbox(self) -> CodeSandbox:
        return CodeSandbox(timeout=10)

    def test_execute_simple_python(self, sandbox: CodeSandbox) -> None:
        result = sandbox.execute("print('hello')")
        assert result.stdout.strip() == "hello"
        assert result.returncode == 0

    def test_execute_captures_stderr(self, sandbox: CodeSandbox) -> None:
        result = sandbox.execute("import sys; sys.stderr.write('err')")
        assert "err" in result.stderr

    def test_execute_nonzero_exit_on_error(self, sandbox: CodeSandbox) -> None:
        result = sandbox.execute("raise ValueError('oops')")
        assert result.returncode != 0

    def test_execute_timeout(self) -> None:
        sandbox = CodeSandbox(timeout=1)
        result = sandbox.execute("import time; time.sleep(10)")
        assert result.returncode != 0
        assert result.timed_out is True

    def test_execute_output_captured(self, sandbox: CodeSandbox) -> None:
        result = sandbox.execute("print(1 + 1)")
        assert "2" in result.stdout

    def test_execute_multiline_code(self, sandbox: CodeSandbox) -> None:
        code = "x = 10\ny = 20\nprint(x + y)"
        result = sandbox.execute(code)
        assert "30" in result.stdout

    def test_sandbox_result_has_required_fields(self, sandbox: CodeSandbox) -> None:
        result = sandbox.execute("print('x')")
        assert hasattr(result, "stdout")
        assert hasattr(result, "stderr")
        assert hasattr(result, "returncode")
        assert hasattr(result, "timed_out")

    def test_execute_empty_code(self, sandbox: CodeSandbox) -> None:
        result = sandbox.execute("")
        assert result.returncode == 0

    def test_execute_syntax_error(self, sandbox: CodeSandbox) -> None:
        result = sandbox.execute("def broken(")
        assert result.returncode != 0


class TestSandboxResult:
    def test_sandbox_result_defaults(self) -> None:
        r = SandboxResult(stdout="out", stderr="err", returncode=0)
        assert r.timed_out is False

    def test_sandbox_result_timed_out(self) -> None:
        r = SandboxResult(stdout="", stderr="", returncode=-1, timed_out=True)
        assert r.timed_out is True
