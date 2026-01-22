"""Unit tests for LLM Factory cache key functionality."""

import pytest
from unittest.mock import patch, MagicMock

from app.agents.base.llm_factory import get_llm, clear_llm_cache


class TestLLMFactoryCaching:
    """Test cache key generation includes all configuration parameters."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_llm_cache()

    def teardown_method(self):
        """Clear cache after each test."""
        clear_llm_cache()

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("langchain_anthropic.ChatAnthropic")
    def test_different_kwargs_create_different_instances(self, mock_anthropic):
        """Test that different kwargs result in different cached instances."""
        # Create two mock LLM instances
        mock_llm1 = MagicMock()
        mock_llm2 = MagicMock()
        mock_anthropic.side_effect = [mock_llm1, mock_llm2]

        # Call get_llm with different max_tokens
        llm1 = get_llm(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            temperature=0.0,
            max_tokens=100,
        )

        llm2 = get_llm(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            temperature=0.0,
            max_tokens=500,
        )

        # Should create two different instances
        assert llm1 is not llm2
        assert mock_anthropic.call_count == 2

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("langchain_anthropic.ChatAnthropic")
    def test_same_kwargs_return_cached_instance(self, mock_anthropic):
        """Test that identical kwargs return the same cached instance."""
        mock_llm = MagicMock()
        mock_anthropic.return_value = mock_llm

        # Call get_llm twice with identical parameters
        llm1 = get_llm(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            temperature=0.0,
            max_tokens=100,
        )

        llm2 = get_llm(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            temperature=0.0,
            max_tokens=100,
        )

        # Should return the same cached instance
        assert llm1 is llm2
        assert mock_anthropic.call_count == 1

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("langchain_anthropic.ChatAnthropic")
    def test_no_kwargs_vs_with_kwargs_different_instances(self, mock_anthropic):
        """Test that calls with and without kwargs create different instances."""
        mock_llm1 = MagicMock()
        mock_llm2 = MagicMock()
        mock_anthropic.side_effect = [mock_llm1, mock_llm2]

        # Call without kwargs
        llm1 = get_llm(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            temperature=0.0,
        )

        # Call with kwargs
        llm2 = get_llm(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            temperature=0.0,
            max_tokens=100,
        )

        # Should create two different instances
        assert llm1 is not llm2
        assert mock_anthropic.call_count == 2

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("langchain_anthropic.ChatAnthropic")
    def test_multiple_kwargs_in_cache_key(self, mock_anthropic):
        """Test that multiple different kwargs are all included in cache key."""
        mock_llm1 = MagicMock()
        mock_llm2 = MagicMock()
        mock_llm3 = MagicMock()
        mock_anthropic.side_effect = [mock_llm1, mock_llm2, mock_llm3]

        # Call with different combinations of kwargs
        llm1 = get_llm(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            temperature=0.0,
            max_tokens=100,
            top_p=0.9,
        )

        llm2 = get_llm(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            temperature=0.0,
            max_tokens=100,
            top_p=0.5,  # Different top_p
        )

        llm3 = get_llm(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            temperature=0.0,
            max_tokens=200,  # Different max_tokens
            top_p=0.9,
        )

        # Should create three different instances
        assert llm1 is not llm2
        assert llm1 is not llm3
        assert llm2 is not llm3
        assert mock_anthropic.call_count == 3

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("langchain_anthropic.ChatAnthropic")
    def test_kwargs_order_does_not_matter(self, mock_anthropic):
        """Test that kwargs order doesn't affect cache key (sorted)."""
        mock_llm = MagicMock()
        mock_anthropic.return_value = mock_llm

        # Call with kwargs in different orders
        llm1 = get_llm(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            temperature=0.0,
            max_tokens=100,
            top_p=0.9,
        )

        llm2 = get_llm(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            temperature=0.0,
            top_p=0.9,  # Swapped order
            max_tokens=100,
        )

        # Should return the same cached instance
        assert llm1 is llm2
        assert mock_anthropic.call_count == 1

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "OPENAI_ENABLED": "true"})
    @patch("langchain_openai.ChatOpenAI")
    def test_cache_works_across_providers(self, mock_openai):
        """Test that cache correctly differentiates between providers."""
        clear_llm_cache()

        mock_llm1 = MagicMock()
        mock_llm2 = MagicMock()
        mock_openai.side_effect = [mock_llm1, mock_llm2]

        # Call with same model and kwargs but different provider (if we had anthropic too)
        # For now, test that same provider with different models creates different instances
        llm1 = get_llm(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=100,
        )

        llm2 = get_llm(
            provider="openai",
            model="gpt-4o",  # Different model
            temperature=0.0,
            max_tokens=100,
        )

        # Should create two different instances
        assert llm1 is not llm2
        assert mock_openai.call_count == 2

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("langchain_anthropic.ChatAnthropic")
    def test_cache_clear_works(self, mock_anthropic):
        """Test that clearing cache creates new instances."""
        mock_llm1 = MagicMock()
        mock_llm2 = MagicMock()
        mock_anthropic.side_effect = [mock_llm1, mock_llm2]

        # Create instance
        llm1 = get_llm(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            temperature=0.0,
            max_tokens=100,
        )

        # Clear cache
        clear_llm_cache()

        # Create again with same params
        llm2 = get_llm(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            temperature=0.0,
            max_tokens=100,
        )

        # Should create a new instance
        assert llm1 is not llm2
        assert mock_anthropic.call_count == 2
