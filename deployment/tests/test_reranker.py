"""Tests for cross-encoder reranking functionality."""
from unittest.mock import MagicMock, patch

import pytest

from app.agents.rag.reranker import CrossEncoderReranker


def _make_docs(*texts: str) -> list[dict]:
    return [{"content": t, "score": 0.5} for t in texts]


class TestCrossEncoderReranker:
    def test_rerank_returns_top_k(self) -> None:
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5]
        reranker._model = mock_model

        docs = _make_docs("doc A", "doc B", "doc C")
        result = reranker.rerank("query", docs, top_k=2)

        assert len(result) == 2

    def test_rerank_most_relevant_first(self) -> None:
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.95, 0.3]
        reranker._model = mock_model

        docs = _make_docs("low relevance", "high relevance", "medium relevance")
        result = reranker.rerank("query", docs, top_k=3)

        assert result[0]["content"] == "high relevance"
        assert result[1]["content"] == "medium relevance"
        assert result[2]["content"] == "low relevance"

    def test_rerank_empty_documents_returns_empty(self) -> None:
        reranker = CrossEncoderReranker()
        result = reranker.rerank("query", [], top_k=5)
        assert result == []

    def test_rerank_fewer_docs_than_top_k(self) -> None:
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8, 0.2]
        reranker._model = mock_model

        docs = _make_docs("doc A", "doc B")
        result = reranker.rerank("query", docs, top_k=10)

        assert len(result) == 2

    def test_rerank_falls_back_when_model_unavailable(self) -> None:
        reranker = CrossEncoderReranker()
        reranker._model = None

        with patch.object(reranker, "_load_model", return_value=None):
            docs = _make_docs("a", "b", "c")
            result = reranker.rerank("query", docs, top_k=2)

        assert len(result) == 2
        assert result[0]["content"] == "a"

    def test_rerank_falls_back_on_predict_error(self) -> None:
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("model error")
        reranker._model = mock_model

        docs = _make_docs("a", "b", "c")
        result = reranker.rerank("query", docs, top_k=2)

        assert len(result) == 2

    def test_lazy_load_skips_import_error(self) -> None:
        reranker = CrossEncoderReranker()
        reranker._model = None

        with patch.dict("sys.modules", {"sentence_transformers": None}):
            model = reranker._load_model()

        assert model is None

    def test_rerank_exact_top_k(self) -> None:
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]
        reranker._model = mock_model

        docs = _make_docs("a", "b", "c", "d", "e")
        result = reranker.rerank("query", docs, top_k=3)

        assert len(result) == 3
        assert result[0]["content"] == "a"


class TestRerankerEnvironmentDefaults:
    def test_default_model_name(self) -> None:
        reranker = CrossEncoderReranker()
        assert "ms-marco" in reranker._model_name or "MiniLM" in reranker._model_name

    def test_custom_model_name(self) -> None:
        reranker = CrossEncoderReranker(model="cross-encoder/my-custom-model")
        assert reranker._model_name == "cross-encoder/my-custom-model"
