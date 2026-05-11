"""Regression test runner for enterprise IT agents.

Provides:
- Automated regression testing against datasets
- CI/CD integration support
- Pass/fail threshold enforcement
- Results reporting and export
"""

import asyncio
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from app.agents.evals.datasets import (
    EvalDataset,
    TestCase,
    get_dataset,
)
from app.agents.evals.evaluators import (
    BaseEvaluator,
    ResponseQualityEvaluator,
    TaskCompletionEvaluator,
    evaluate_agent_response,
)


@dataclass
class RegressionConfig:
    """Configuration for regression testing.

    Attributes:
        pass_threshold: Minimum score to pass (0.0 to 1.0)
        fail_on_error: Whether to fail entire run on any error
        max_concurrent: Maximum concurrent test executions
        timeout_seconds: Timeout for each test case
        output_format: Output format (json, markdown, junit)
        output_path: Path for results output
        verbose: Whether to print verbose output
    """

    pass_threshold: float = 0.7
    fail_on_error: bool = False
    max_concurrent: int = 5
    timeout_seconds: int = 60
    output_format: Literal["json", "markdown", "junit"] = "json"
    output_path: str | None = None
    verbose: bool = True

    @classmethod
    def from_env(cls) -> "RegressionConfig":
        """Create config from environment variables."""
        return cls(
            pass_threshold=float(os.getenv("EVAL_THRESHOLD_PASS", "0.7")),
            fail_on_error=os.getenv("EVAL_FAIL_ON_ERROR", "false").lower() == "true",
            max_concurrent=int(os.getenv("EVAL_MAX_CONCURRENT", "5")),
            timeout_seconds=int(os.getenv("EVAL_TIMEOUT_SECONDS", "60")),
            output_format=os.getenv("EVAL_OUTPUT_FORMAT", "json"),
            output_path=os.getenv("EVAL_OUTPUT_PATH"),
            verbose=os.getenv("EVAL_VERBOSE", "true").lower() == "true",
        )


@dataclass
class TestResult:
    """Result from a single test case execution.

    Attributes:
        test_case_id: ID of the test case
        passed: Whether the test passed
        score: Overall score
        execution_time_ms: Execution time in milliseconds
        input: Test input
        output: Agent output
        expected: Expected output
        evaluations: Individual evaluation results
        error: Error message if any
    """

    test_case_id: str
    passed: bool
    score: float
    execution_time_ms: int = 0
    input: str = ""
    output: str = ""
    expected: str | None = None
    evaluations: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: str | None = None


@dataclass
class RegressionReport:
    """Report from a regression test run.

    Attributes:
        run_id: Unique run identifier
        timestamp: Run timestamp
        dataset_name: Dataset tested
        agent_type: Type of agent tested
        total_tests: Total number of tests
        passed_tests: Number of passed tests
        failed_tests: Number of failed tests
        error_tests: Number of tests with errors
        pass_rate: Pass rate percentage
        average_score: Average score across tests
        duration_seconds: Total run duration
        results: Individual test results
        config: Configuration used
        metadata: Additional metadata
    """

    run_id: str = ""
    timestamp: str = ""
    dataset_name: str = ""
    agent_type: str = ""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error_tests: int = 0
    pass_rate: float = 0.0
    average_score: float = 0.0
    duration_seconds: float = 0.0
    results: list[TestResult] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def overall_passed(self) -> bool:
        """Check if overall run passed threshold."""
        return self.pass_rate >= self.config.get("pass_threshold", 0.7) * 100


class RegressionRunner:
    """Runs regression tests against agent datasets.

    Supports:
    - Parallel test execution
    - Multiple evaluation metrics
    - CI/CD integration
    - Multiple output formats
    """

    def __init__(
        self,
        config: RegressionConfig | None = None,
        evaluators: list[BaseEvaluator] | None = None,
    ) -> None:
        """Initialize the regression runner.

        Args:
            config: Regression test configuration.
            evaluators: Evaluators to use for testing.
        """
        self.config = config or RegressionConfig.from_env()
        self.evaluators = evaluators or [
            ResponseQualityEvaluator(),
            TaskCompletionEvaluator(),
        ]

    async def run_test_case(
        self,
        test_case: TestCase,
        agent_func: Callable,
    ) -> TestResult:
        """Run a single test case.

        Args:
            test_case: Test case to run.
            agent_func: Agent function to test.

        Returns:
            TestResult with evaluation results.
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Execute agent with timeout
            output = await asyncio.wait_for(
                self._execute_agent(agent_func, test_case.input),
                timeout=self.config.timeout_seconds,
            )
            output_text = str(output)
            error = None

        except asyncio.TimeoutError:
            output_text = ""
            error = f"Timeout after {self.config.timeout_seconds} seconds"

        except Exception as e:
            output_text = ""
            error = str(e)

        end_time = datetime.now(timezone.utc)
        execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        # Run evaluations if no error
        evaluations = {}
        total_score = 0.0

        if not error:
            results = evaluate_agent_response(
                input_text=test_case.input,
                output_text=output_text,
                evaluators=self.evaluators,
                expected=test_case.expected_output,
            )

            for name, result in results.items():
                evaluations[name] = {
                    "score": result.score,
                    "passed": result.passed,
                    "feedback": result.feedback,
                }
                total_score += result.score

            if evaluations:
                avg_score = total_score / len(evaluations)
            else:
                avg_score = 0.0
        else:
            avg_score = 0.0

        passed = avg_score >= self.config.pass_threshold and error is None

        return TestResult(
            test_case_id=test_case.id,
            passed=passed,
            score=avg_score,
            execution_time_ms=execution_time_ms,
            input=test_case.input,
            output=output_text,
            expected=test_case.expected_output,
            evaluations=evaluations,
            error=error,
        )

    async def _execute_agent(
        self,
        agent_func: Callable,
        input_text: str,
    ) -> str:
        """Execute agent function (handles both sync and async).

        Args:
            agent_func: Agent function to execute.
            input_text: Input to pass to agent.

        Returns:
            Agent output as string.
        """
        if asyncio.iscoroutinefunction(agent_func):
            result = await agent_func(input_text)
        else:
            result = agent_func(input_text)
        return str(result)

    async def run_dataset(
        self,
        dataset: EvalDataset,
        agent_func: Callable,
    ) -> RegressionReport:
        """Run regression tests for an entire dataset.

        Args:
            dataset: Dataset to test against.
            agent_func: Agent function to test.

        Returns:
            RegressionReport with all results.
        """
        start_time = datetime.now(timezone.utc)
        run_id = f"regression-{start_time.strftime('%Y%m%d-%H%M%S')}"

        if self.config.verbose:
            print(f"\nRunning regression tests: {dataset.name}")
            print(f"Test cases: {len(dataset.test_cases)}")
            print("-" * 40)

        # Run tests with concurrency limit
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def run_with_semaphore(test_case: TestCase) -> TestResult:
            async with semaphore:
                result = await self.run_test_case(test_case, agent_func)
                if self.config.verbose:
                    status = "PASS" if result.passed else "FAIL"
                    print(f"  [{status}] {test_case.id}: {result.score:.2f}")
                return result

        tasks = [run_with_semaphore(tc) for tc in dataset.test_cases]
        results = await asyncio.gather(*tasks)

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        # Calculate statistics
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = sum(1 for r in results if not r.passed and r.error is None)
        error_tests = sum(1 for r in results if r.error is not None)
        total_tests = len(results)

        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        avg_score = sum(r.score for r in results) / total_tests if total_tests > 0 else 0

        report = RegressionReport(
            run_id=run_id,
            timestamp=start_time.isoformat(),
            dataset_name=dataset.name,
            agent_type=dataset.agent_type,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            error_tests=error_tests,
            pass_rate=pass_rate,
            average_score=avg_score,
            duration_seconds=duration,
            results=results,
            config={
                "pass_threshold": self.config.pass_threshold,
                "timeout_seconds": self.config.timeout_seconds,
            },
        )

        if self.config.verbose:
            print("-" * 40)
            print(f"Results: {passed_tests}/{total_tests} passed ({pass_rate:.1f}%)")
            print(f"Average score: {avg_score:.2f}")
            print(f"Duration: {duration:.2f}s")

        # Save report if output path specified
        if self.config.output_path:
            self._save_report(report)

        return report

    async def run_all_datasets(
        self,
        agent_funcs: dict[str, Callable],
    ) -> dict[str, RegressionReport]:
        """Run regression tests for all datasets.

        Args:
            agent_funcs: Dictionary mapping agent type to function.

        Returns:
            Dictionary of dataset name to report.
        """
        reports = {}

        for agent_type, agent_func in agent_funcs.items():
            dataset = get_dataset(agent_type)
            if dataset:
                report = await self.run_dataset(dataset, agent_func)
                reports[dataset.name] = report

        return reports

    def _save_report(self, report: RegressionReport) -> None:
        """Save report to file.

        Args:
            report: Report to save.
        """
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.config.output_format == "json":
            self._save_json(report, output_path)
        elif self.config.output_format == "markdown":
            self._save_markdown(report, output_path)
        elif self.config.output_format == "junit":
            self._save_junit(report, output_path)

    def _save_json(self, report: RegressionReport, path: Path) -> None:
        """Save report as JSON."""
        data = {
            "run_id": report.run_id,
            "timestamp": report.timestamp,
            "dataset_name": report.dataset_name,
            "agent_type": report.agent_type,
            "summary": {
                "total_tests": report.total_tests,
                "passed_tests": report.passed_tests,
                "failed_tests": report.failed_tests,
                "error_tests": report.error_tests,
                "pass_rate": report.pass_rate,
                "average_score": report.average_score,
                "duration_seconds": report.duration_seconds,
                "overall_passed": report.overall_passed,
            },
            "config": report.config,
            "results": [
                {
                    "test_case_id": r.test_case_id,
                    "passed": r.passed,
                    "score": r.score,
                    "execution_time_ms": r.execution_time_ms,
                    "evaluations": r.evaluations,
                    "error": r.error,
                }
                for r in report.results
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        if self.config.verbose:
            print(f"Report saved to: {path}")

    def _save_markdown(self, report: RegressionReport, path: Path) -> None:
        """Save report as Markdown."""
        lines = [
            "# Regression Test Report",
            "",
            f"**Run ID:** {report.run_id}",
            f"**Timestamp:** {report.timestamp}",
            f"**Dataset:** {report.dataset_name}",
            f"**Agent Type:** {report.agent_type}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Tests | {report.total_tests} |",
            f"| Passed | {report.passed_tests} |",
            f"| Failed | {report.failed_tests} |",
            f"| Errors | {report.error_tests} |",
            f"| Pass Rate | {report.pass_rate:.1f}% |",
            f"| Average Score | {report.average_score:.2f} |",
            f"| Duration | {report.duration_seconds:.2f}s |",
            f"| **Overall** | {'PASSED' if report.overall_passed else 'FAILED'} |",
            "",
            "## Test Results",
            "",
            "| Test Case | Status | Score | Time (ms) |",
            "|-----------|--------|-------|-----------|",
        ]

        for r in report.results:
            status = "PASS" if r.passed else ("ERROR" if r.error else "FAIL")
            lines.append(f"| {r.test_case_id} | {status} | {r.score:.2f} | {r.execution_time_ms} |")

        lines.append("")

        # Add failed test details
        failed = [r for r in report.results if not r.passed]
        if failed:
            lines.append("## Failed Tests")
            lines.append("")
            for r in failed:
                lines.append(f"### {r.test_case_id}")
                lines.append("")
                if r.error:
                    lines.append(f"**Error:** {r.error}")
                else:
                    lines.append(f"**Score:** {r.score:.2f}")
                    for name, eval_result in r.evaluations.items():
                        lines.append(f"- {name}: {eval_result.get('feedback', '')}")
                lines.append("")

        with open(path, "w") as f:
            f.write("\n".join(lines))

        if self.config.verbose:
            print(f"Report saved to: {path}")

    def _save_junit(self, report: RegressionReport, path: Path) -> None:
        """Save report as JUnit XML for CI/CD integration."""
        from xml.etree.ElementTree import Element, SubElement, tostring

        testsuite = Element("testsuite")
        testsuite.set("name", report.dataset_name)
        testsuite.set("tests", str(report.total_tests))
        testsuite.set("failures", str(report.failed_tests))
        testsuite.set("errors", str(report.error_tests))
        testsuite.set("time", str(report.duration_seconds))
        testsuite.set("timestamp", report.timestamp)

        for r in report.results:
            testcase = SubElement(testsuite, "testcase")
            testcase.set("name", r.test_case_id)
            testcase.set("time", str(r.execution_time_ms / 1000))
            testcase.set("classname", report.agent_type)

            if r.error:
                error = SubElement(testcase, "error")
                error.set("message", r.error)
            elif not r.passed:
                failure = SubElement(testcase, "failure")
                feedback = "; ".join(e.get("feedback", "") for e in r.evaluations.values())
                failure.set("message", f"Score: {r.score:.2f} - {feedback}")

        xml_str = tostring(testsuite, encoding="unicode")
        with open(path, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write(xml_str)

        if self.config.verbose:
            print(f"JUnit report saved to: {path}")


def run_regression_sync(
    agent_func: Callable,
    agent_type: str,
    config: RegressionConfig | None = None,
) -> RegressionReport:
    """Synchronous wrapper for running regression tests.

    Args:
        agent_func: Agent function to test.
        agent_type: Type of agent (must match dataset key).
        config: Optional configuration.

    Returns:
        RegressionReport with results.
    """
    dataset = get_dataset(agent_type)
    if not dataset:
        raise ValueError(f"No dataset found for agent type: {agent_type}")

    runner = RegressionRunner(config=config)
    return asyncio.run(runner.run_dataset(dataset, agent_func))


async def run_regression_async(
    agent_func: Callable,
    agent_type: str,
    config: RegressionConfig | None = None,
) -> RegressionReport:
    """Asynchronous regression test runner.

    Args:
        agent_func: Agent function to test.
        agent_type: Type of agent (must match dataset key).
        config: Optional configuration.

    Returns:
        RegressionReport with results.
    """
    dataset = get_dataset(agent_type)
    if not dataset:
        raise ValueError(f"No dataset found for agent type: {agent_type}")

    runner = RegressionRunner(config=config)
    return await runner.run_dataset(dataset, agent_func)


def check_regression_passed(report: RegressionReport) -> bool:
    """Check if regression tests passed (for CI/CD exit code).

    Args:
        report: Regression report to check.

    Returns:
        True if tests passed threshold.
    """
    return report.overall_passed
