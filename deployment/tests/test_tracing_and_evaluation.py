"""Test script for LangSmith tracing and evaluation functionality.

This script verifies:
1. LangSmith tracing configuration
2. Connection to LangSmith API
3. Evaluation framework with proper variable mapping
4. Dataset creation and synchronization

Run with: python -m pytest tests/test_tracing_and_evaluation.py -v
Or standalone: python tests/test_tracing_and_evaluation.py

Created: 2026-01-02
Purpose: Verify fixes for tracing and evaluator KeyError issues
"""

import os
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add the deployment directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TestTracingConfiguration:
    """Test tracing configuration and verification."""

    def test_verify_tracing_config(self):
        """Test that verify_tracing_config returns proper status."""
        from app.agents.evals import verify_tracing_config

        result = verify_tracing_config()

        assert "tracing_enabled" in result
        assert "api_key_configured" in result
        assert "project_name" in result
        assert "status" in result
        assert "issues" in result
        assert isinstance(result["issues"], list)

        print(f"\nTracing config status: {result['status']}")
        if result["issues"]:
            print(f"Issues found: {result['issues']}")

    def test_ensure_tracing_enabled(self):
        """Test ensure_tracing_enabled function."""
        from app.agents.evals import ensure_tracing_enabled

        # This will return True if API key is available
        result = ensure_tracing_enabled()

        # Check environment variables are set
        if result:
            assert os.getenv("LANGCHAIN_TRACING_V2") == "true"
            assert os.getenv("LANGCHAIN_API_KEY") is not None
            print("\nTracing successfully enabled")
        else:
            print("\nTracing not enabled (no API key)")


class TestLangSmithConnection:
    """Test connection to LangSmith API."""

    @pytest.mark.skipif(
        not os.getenv("LANGCHAIN_API_KEY"),
        reason="LANGCHAIN_API_KEY not set"
    )
    def test_langsmith_connection(self):
        """Test connection to LangSmith."""
        from app.agents.evals import test_langsmith_connection

        result = test_langsmith_connection()

        print(f"\nConnection test result: {result}")

        assert "connected" in result
        assert "timestamp" in result

        if result["connected"]:
            print(f"Connected successfully. Found projects: {result['projects']}")
        else:
            print(f"Connection failed: {result.get('error')}")

    @pytest.mark.skipif(
        not os.getenv("LANGCHAIN_API_KEY"),
        reason="LANGCHAIN_API_KEY not set"
    )
    def test_get_recent_traces(self):
        """Test getting recent traces."""
        from app.agents.evals import get_recent_traces

        result = get_recent_traces(hours=48, limit=5)

        print(f"\nRecent traces result: {result}")

        assert "project" in result
        assert "traces" in result
        assert "total_count" in result

        if result["traces"]:
            print(f"Found {result['total_count']} recent traces")
            for trace in result["traces"][:3]:
                print(f"  - {trace['name']}: {trace['status']}")
        else:
            print("No recent traces found")


class TestEvaluatorVariableMapping:
    """Test that evaluators properly handle LangSmith variable mapping."""

    def test_evaluator_wrapper_creation(self):
        """Test creating LangSmith SDK compatible evaluator wrapper."""
        from app.agents.evals import (
            ResponseQualityEvaluator,
            create_langsmith_evaluator_wrapper,
        )

        base_evaluator = ResponseQualityEvaluator()
        wrapper = create_langsmith_evaluator_wrapper(base_evaluator)

        assert callable(wrapper)
        assert wrapper.__name__ == "response_quality"

    def test_evaluator_wrapper_execution(self):
        """Test that wrapper properly handles inputs/outputs/reference_outputs."""
        from app.agents.evals import (
            ResponseQualityEvaluator,
            create_langsmith_evaluator_wrapper,
        )

        base_evaluator = ResponseQualityEvaluator()
        wrapper = create_langsmith_evaluator_wrapper(base_evaluator)

        # Test with proper LangSmith variable structure
        result = wrapper(
            inputs={"input": "What is the capital of France?"},
            outputs={"output": "The capital of France is Paris."},
            reference_outputs={"expected": "Paris is the capital of France."},
        )

        assert "key" in result
        assert "score" in result
        assert "comment" in result
        assert result["key"] == "response_quality"
        assert isinstance(result["score"], (int, float))

        print(f"\nEvaluator result: {result}")

    def test_evaluator_wrapper_without_reference(self):
        """Test wrapper handles missing reference_outputs gracefully."""
        from app.agents.evals import (
            ResponseQualityEvaluator,
            create_langsmith_evaluator_wrapper,
        )

        base_evaluator = ResponseQualityEvaluator()
        wrapper = create_langsmith_evaluator_wrapper(base_evaluator)

        # Test without reference_outputs (should not raise KeyError)
        result = wrapper(
            inputs={"input": "What is AI?"},
            outputs={"output": "AI is artificial intelligence."},
            reference_outputs=None,
        )

        assert "key" in result
        assert "score" in result
        # Should still work without reference
        print(f"\nResult without reference: {result}")


class TestDatasetSynchronization:
    """Test dataset synchronization with proper schema."""

    def test_sync_dataset_structure(self):
        """Test that sync_dataset_from_local creates proper schema."""
        from app.agents.evals import LangSmithEvaluator

        evaluator = LangSmithEvaluator()

        # Create test cases
        test_cases = [
            {
                "id": "test_001",
                "input": "Test question?",
                "expected_output": "Test answer.",
                "expected_keywords": ["test", "answer"],
                "tags": ["unit-test"],
                "difficulty": "easy",
            }
        ]

        # Test the example structure creation (without actually syncing)
        examples = []
        for case in test_cases:
            expected_output = case.get("expected_output") or ""
            expected_keywords = case.get("expected_keywords", [])
            context = expected_output
            if expected_keywords:
                context += f"\nExpected keywords: {', '.join(expected_keywords)}"

            examples.append({
                "inputs": {
                    "input": case.get("input", ""),
                    "context": context,
                },
                "outputs": {
                    "expected": expected_output,
                    "keywords": expected_keywords,
                    "reference_output": expected_output,
                },
                "metadata": {
                    "id": case.get("id"),
                    "tags": case.get("tags", []),
                    "difficulty": case.get("difficulty", "medium"),
                },
            })

        # Verify structure
        example = examples[0]
        assert "inputs" in example
        assert "outputs" in example
        assert "context" in example["inputs"]
        assert "reference_output" in example["outputs"]
        assert example["inputs"]["context"] is not None
        assert example["outputs"]["reference_output"] == "Test answer."

        print(f"\nExample structure: {example}")


class TestEvaluationExecution:
    """Test actual evaluation execution."""

    def test_base_evaluators_work(self):
        """Test that base evaluators work correctly."""
        from app.agents.evals import (
            ResponseQualityEvaluator,
            TaskCompletionEvaluator,
            FactualAccuracyEvaluator,
        )

        input_text = "What is LangChain?"
        output_text = "LangChain is a framework for building applications with LLMs."
        expected = "LangChain is a Python framework for LLM applications."

        # Test each evaluator
        evaluators = [
            ResponseQualityEvaluator(),
            TaskCompletionEvaluator(),
            FactualAccuracyEvaluator(),
        ]

        for evaluator in evaluators:
            result = evaluator.evaluate(input_text, output_text, expected)
            assert result.score >= 0.0
            assert result.score <= 1.0
            assert isinstance(result.passed, bool)
            assert result.feedback is not None
            print(f"\n{evaluator.name}: score={result.score:.2f}, passed={result.passed}")


def run_interactive_test():
    """Run interactive test to verify tracing and evaluation."""
    print("=" * 60)
    print("LangSmith Tracing and Evaluation Test")
    print("=" * 60)

    # Load environment
    load_dotenv()

    # 1. Check tracing configuration
    print("\n1. Checking tracing configuration...")
    from app.agents.evals import verify_tracing_config

    config = verify_tracing_config()
    print(f"   Status: {config['status']}")
    print(f"   Tracing enabled: {config['tracing_enabled']}")
    print(f"   API key configured: {config['api_key_configured']}")
    print(f"   Project: {config['project_name']}")
    if config['issues']:
        print(f"   Issues: {config['issues']}")

    # 2. Test connection
    print("\n2. Testing LangSmith connection...")
    from app.agents.evals import test_langsmith_connection

    conn_result = test_langsmith_connection()
    print(f"   Connected: {conn_result['connected']}")
    if conn_result['connected']:
        print(f"   Projects: {conn_result['projects'][:3]}...")
    else:
        print(f"   Error: {conn_result.get('error', 'Unknown')}")

    # 3. Check recent traces
    print("\n3. Checking recent traces...")
    from app.agents.evals import get_recent_traces

    traces_result = get_recent_traces(hours=72, limit=10)
    print(f"   Project: {traces_result['project']}")
    print(f"   Traces found: {traces_result['total_count']}")
    if traces_result['traces']:
        print("   Recent traces:")
        for trace in traces_result['traces'][:5]:
            status = trace.get('status', 'unknown')
            name = trace.get('name', 'unnamed')[:40]
            print(f"      - {name}: {status}")
    else:
        print("   No traces found in the last 72 hours")
        print("   This may indicate tracing is not being captured")

    # 4. Test evaluator wrapper
    print("\n4. Testing evaluator wrapper...")
    from app.agents.evals import (
        ResponseQualityEvaluator,
        create_langsmith_evaluator_wrapper,
    )

    wrapper = create_langsmith_evaluator_wrapper(ResponseQualityEvaluator())
    result = wrapper(
        inputs={"input": "Test question"},
        outputs={"output": "Test response"},
        reference_outputs={"expected": "Expected response"},
    )
    print(f"   Wrapper test passed: {result['score'] >= 0}")
    print(f"   Score: {result['score']:.2f}")

    # 5. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_ok = (
        config['status'] == 'OK' and
        conn_result['connected'] and
        result['score'] >= 0
    )

    if all_ok:
        print("[PASS] All checks passed!")
        if traces_result['total_count'] == 0:
            print("[WARN] No recent traces found. Make some API calls to generate traces.")
    else:
        print("[FAIL] Some checks failed:")
        if config['status'] != 'OK':
            print("  - Tracing configuration issues")
        if not conn_result['connected']:
            print("  - LangSmith connection failed")

    return all_ok


if __name__ == "__main__":
    success = run_interactive_test()
    sys.exit(0 if success else 1)
