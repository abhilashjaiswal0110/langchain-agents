"""Smoke tests that validate the Locust load test files can be imported."""
import importlib
import sys


def _import_or_skip(module_path: str):
    """Import a module or skip the test if locust is not installed."""
    import pytest
    try:
        return importlib.import_module(module_path)
    except ImportError as exc:
        if "locust" in str(exc).lower():
            pytest.skip("locust not installed")
        raise


class TestLocustfileStructure:
    def test_locustfile_importable(self) -> None:
        _import_or_skip("tests.load.locustfile")

    def test_research_scenario_importable(self) -> None:
        _import_or_skip("tests.load.scenarios.research_scenario")

    def test_deep_agent_scenario_importable(self) -> None:
        _import_or_skip("tests.load.scenarios.deep_agent_scenario")

    def test_conversation_scenario_importable(self) -> None:
        _import_or_skip("tests.load.scenarios.conversation_scenario")

    def test_locustfile_has_agent_load_user(self) -> None:
        mod = _import_or_skip("tests.load.locustfile")
        assert hasattr(mod, "AgentLoadUser")

    def test_research_scenario_has_user(self) -> None:
        mod = _import_or_skip("tests.load.scenarios.research_scenario")
        assert hasattr(mod, "ResearchAgentUser")

    def test_deep_agent_scenario_has_user(self) -> None:
        mod = _import_or_skip("tests.load.scenarios.deep_agent_scenario")
        assert hasattr(mod, "DeepAgentUser")

    def test_conversation_scenario_has_user(self) -> None:
        mod = _import_or_skip("tests.load.scenarios.conversation_scenario")
        assert hasattr(mod, "ConversationUser")
