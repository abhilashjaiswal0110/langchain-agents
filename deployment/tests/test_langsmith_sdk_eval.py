"""Test LangSmith SDK evaluate function with sync target.

This verifies that:
1. Datasets are created correctly with proper schema
2. LangSmith SDK evaluate() works with sync functions
3. Custom evaluators integrate properly

Run with: python tests/test_langsmith_sdk_eval.py
"""

import os
import sys
from pathlib import Path

# Add deployment to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


def run_langsmith_sdk_evaluation_test():
    """Test the LangSmith SDK evaluate function."""
    print("=" * 60)
    print("LangSmith SDK Evaluation Test")
    print("=" * 60)

    from langsmith import Client
    from langsmith.evaluation import evaluate

    client = Client()

    # 1. Define a simple sync target function
    def simple_qa(inputs: dict) -> dict:
        """A simple QA function that returns a response."""
        question = inputs.get("input", "")
        return {"output": f"Response to: {question}"}

    # 2. Define custom evaluators matching LangSmith schema
    def length_evaluator(outputs: dict, reference_outputs: dict = None) -> dict:
        """Evaluator that checks output length."""
        output = outputs.get("output", "")
        return {
            "key": "output_length",
            "score": min(1.0, len(output) / 50),
            "comment": f"Output has {len(output)} characters"
        }

    def relevance_evaluator(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
        """Evaluator that checks if output relates to input."""
        input_text = inputs.get("input", "").lower()
        output_text = outputs.get("output", "").lower()
        # Simple check - does output mention any input words?
        input_words = set(input_text.split())
        output_words = set(output_text.split())
        overlap = len(input_words & output_words)
        score = min(1.0, overlap / max(1, len(input_words)))
        return {
            "key": "relevance",
            "score": score,
            "comment": f"Word overlap: {overlap}"
        }

    # 3. Create test dataset
    dataset_name = "test_sdk_evaluation_jan03_2026"
    print(f"\n1. Creating dataset: {dataset_name}")

    # Clean up if exists
    try:
        for ds in client.list_datasets(dataset_name=dataset_name):
            client.delete_dataset(dataset_id=ds.id)
            print(f"   Deleted existing dataset")
    except Exception as e:
        print(f"   No existing dataset to delete")

    # Create new dataset
    dataset = client.create_dataset(dataset_name)
    print(f"   Created dataset: {dataset.id}")

    # 4. Add examples with proper schema (inputs + reference outputs)
    print("\n2. Adding test examples...")
    examples = [
        {
            "inputs": {"input": "What is LangChain?"},
            "outputs": {"output": "LangChain is a framework for building LLM applications."}
        },
        {
            "inputs": {"input": "What is LangGraph?"},
            "outputs": {"output": "LangGraph is a library for building agentic workflows."}
        },
        {
            "inputs": {"input": "How do I trace LLM calls?"},
            "outputs": {"output": "Use LangSmith tracing with LANGCHAIN_TRACING_V2=true."}
        }
    ]

    for i, ex in enumerate(examples):
        client.create_example(
            inputs=ex["inputs"],
            outputs=ex["outputs"],
            dataset_id=dataset.id
        )
        print(f"   Added example {i+1}: {ex['inputs']['input'][:40]}...")

    # 5. Run evaluation
    print("\n3. Running LangSmith SDK evaluate()...")
    try:
        results = evaluate(
            simple_qa,
            data=dataset_name,
            evaluators=[length_evaluator, relevance_evaluator],
            experiment_prefix="sdk_eval_test"
        )
        print(f"   [PASS] Evaluation completed successfully!")
        print(f"   Results type: {type(results).__name__}")

        # Try to access results
        try:
            result_list = list(results)
            print(f"   Number of results: {len(result_list)}")
        except:
            print("   Results iterable")

    except Exception as e:
        print(f"   [FAIL] Evaluation error: {e}")
        import traceback
        traceback.print_exc()

    # 6. Cleanup
    print("\n4. Cleaning up...")
    try:
        client.delete_dataset(dataset_id=dataset.id)
        print(f"   Deleted test dataset")
    except Exception as e:
        print(f"   Cleanup error: {e}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_langsmith_sdk_evaluation_test()
