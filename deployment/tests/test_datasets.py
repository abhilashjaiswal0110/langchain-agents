"""Tests for the evaluation datasets."""

import pytest
from app.agents.evals.datasets import (
    ALL_DATASETS,
    CODE_ASSISTANT_DATASET,
    CONTENT_AGENT_DATASET,
    DATA_ANALYST_DATASET,
    DOCUMENT_AGENT_DATASET,
    HITL_SUPPORT_DATASET,
    MULTILINGUAL_RAG_DATASET,
    RESEARCH_AGENT_DATASET,
    EvalDataset,
    TestCase,
    get_dataset,
    get_test_cases_by_difficulty,
    get_test_cases_by_tag,
)


class TestTestCase:
    """Tests for TestCase dataclass."""

    def test_create_minimal(self):
        """Test creating test case with minimal fields."""
        tc = TestCase(id="test_001", input="Test input")
        assert tc.id == "test_001"
        assert tc.input == "Test input"
        assert tc.expected_output is None
        assert tc.expected_keywords == []
        assert tc.tags == []
        assert tc.difficulty == "medium"

    def test_create_full(self):
        """Test creating test case with all fields."""
        tc = TestCase(
            id="test_002",
            input="What is AI?",
            expected_output="AI is artificial intelligence",
            expected_keywords=["artificial", "intelligence"],
            tags=["basic", "definition"],
            difficulty="easy",
            metadata={"source": "test"},
        )
        assert tc.expected_keywords == ["artificial", "intelligence"]
        assert tc.tags == ["basic", "definition"]
        assert tc.difficulty == "easy"
        assert tc.metadata["source"] == "test"


class TestEvalDataset:
    """Tests for EvalDataset dataclass."""

    def test_create_dataset(self):
        """Test creating an evaluation dataset."""
        dataset = EvalDataset(
            name="test_dataset",
            description="Test dataset for unit tests",
            agent_type="test_agent",
        )
        assert dataset.name == "test_dataset"
        assert dataset.agent_type == "test_agent"
        assert dataset.test_cases == []
        assert dataset.version == "1.0"

    def test_dataset_with_test_cases(self):
        """Test creating dataset with test cases."""
        tc = TestCase(id="tc_001", input="Test")
        dataset = EvalDataset(
            name="test",
            description="Test",
            agent_type="test",
            test_cases=[tc],
        )
        assert len(dataset.test_cases) == 1
        assert dataset.test_cases[0].id == "tc_001"


class TestPredefinedDatasets:
    """Tests for predefined evaluation datasets."""

    def test_research_dataset_exists(self):
        """Test research agent dataset."""
        assert RESEARCH_AGENT_DATASET.agent_type == "research"
        assert len(RESEARCH_AGENT_DATASET.test_cases) >= 3

    def test_content_dataset_exists(self):
        """Test content agent dataset."""
        assert CONTENT_AGENT_DATASET.agent_type == "content"
        assert len(CONTENT_AGENT_DATASET.test_cases) >= 3

    def test_data_analyst_dataset_exists(self):
        """Test data analyst dataset."""
        assert DATA_ANALYST_DATASET.agent_type == "data_analyst"
        assert len(DATA_ANALYST_DATASET.test_cases) >= 3

    def test_document_dataset_exists(self):
        """Test document agent dataset."""
        assert DOCUMENT_AGENT_DATASET.agent_type == "document"
        assert len(DOCUMENT_AGENT_DATASET.test_cases) >= 3

    def test_multilingual_rag_dataset_exists(self):
        """Test multilingual RAG dataset."""
        assert MULTILINGUAL_RAG_DATASET.agent_type == "multilingual_rag"
        assert len(MULTILINGUAL_RAG_DATASET.test_cases) >= 3

    def test_hitl_support_dataset_exists(self):
        """Test HITL support dataset."""
        assert HITL_SUPPORT_DATASET.agent_type == "hitl_support"
        assert len(HITL_SUPPORT_DATASET.test_cases) >= 3

    def test_code_assistant_dataset_exists(self):
        """Test code assistant dataset."""
        assert CODE_ASSISTANT_DATASET.agent_type == "code_assistant"
        assert len(CODE_ASSISTANT_DATASET.test_cases) >= 3

    def test_all_datasets_registry(self):
        """Test ALL_DATASETS contains all agent types."""
        expected_types = [
            "research",
            "content",
            "data_analyst",
            "document",
            "multilingual_rag",
            "hitl_support",
            "code_assistant",
        ]
        for agent_type in expected_types:
            assert agent_type in ALL_DATASETS
            assert ALL_DATASETS[agent_type].agent_type == agent_type


class TestGetDataset:
    """Tests for get_dataset function."""

    def test_get_existing_dataset(self):
        """Test getting an existing dataset."""
        dataset = get_dataset("research")
        assert dataset is not None
        assert dataset.agent_type == "research"

    def test_get_nonexistent_dataset(self):
        """Test getting a non-existent dataset."""
        dataset = get_dataset("nonexistent_agent")
        assert dataset is None

    def test_get_all_datasets(self):
        """Test getting all predefined datasets."""
        for agent_type in ALL_DATASETS.keys():
            dataset = get_dataset(agent_type)
            assert dataset is not None
            assert dataset.agent_type == agent_type


class TestGetTestCasesByTag:
    """Tests for get_test_cases_by_tag function."""

    def test_get_by_existing_tag(self):
        """Test getting test cases by existing tag."""
        cases = get_test_cases_by_tag("security")
        assert len(cases) > 0
        for case in cases:
            assert "security" in case.tags

    def test_get_by_nonexistent_tag(self):
        """Test getting test cases by non-existent tag."""
        cases = get_test_cases_by_tag("nonexistent_tag_xyz")
        assert len(cases) == 0

    def test_tag_filtering_accuracy(self):
        """Test that tag filtering is accurate."""
        cases = get_test_cases_by_tag("technical")
        for case in cases:
            assert "technical" in case.tags


class TestGetTestCasesByDifficulty:
    """Tests for get_test_cases_by_difficulty function."""

    def test_get_easy_cases(self):
        """Test getting easy test cases."""
        cases = get_test_cases_by_difficulty("easy")
        assert len(cases) > 0
        for case in cases:
            assert case.difficulty == "easy"

    def test_get_medium_cases(self):
        """Test getting medium test cases."""
        cases = get_test_cases_by_difficulty("medium")
        assert len(cases) > 0
        for case in cases:
            assert case.difficulty == "medium"

    def test_get_hard_cases(self):
        """Test getting hard test cases."""
        cases = get_test_cases_by_difficulty("hard")
        assert len(cases) > 0
        for case in cases:
            assert case.difficulty == "hard"

    def test_invalid_difficulty(self):
        """Test getting test cases with invalid difficulty."""
        cases = get_test_cases_by_difficulty("impossible")
        assert len(cases) == 0


class TestDatasetIntegrity:
    """Tests for dataset integrity and quality."""

    def test_all_test_cases_have_ids(self):
        """Test that all test cases have unique IDs."""
        all_ids = set()
        for dataset in ALL_DATASETS.values():
            for tc in dataset.test_cases:
                assert tc.id, f"Test case in {dataset.name} missing ID"
                assert tc.id not in all_ids, f"Duplicate ID: {tc.id}"
                all_ids.add(tc.id)

    def test_all_test_cases_have_input(self):
        """Test that all test cases have input."""
        for dataset in ALL_DATASETS.values():
            for tc in dataset.test_cases:
                assert tc.input, f"Test case {tc.id} missing input"

    def test_all_test_cases_have_keywords(self):
        """Test that all test cases have expected keywords."""
        for dataset in ALL_DATASETS.values():
            for tc in dataset.test_cases:
                assert len(tc.expected_keywords) > 0, \
                    f"Test case {tc.id} missing expected keywords"

    def test_all_test_cases_have_valid_difficulty(self):
        """Test that all test cases have valid difficulty."""
        valid_difficulties = {"easy", "medium", "hard"}
        for dataset in ALL_DATASETS.values():
            for tc in dataset.test_cases:
                assert tc.difficulty in valid_difficulties, \
                    f"Test case {tc.id} has invalid difficulty: {tc.difficulty}"
