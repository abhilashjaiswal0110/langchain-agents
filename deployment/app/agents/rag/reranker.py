"""Cross-encoder reranker for improving RAG retrieval quality.

Wraps `sentence_transformers.CrossEncoder` with lazy loading,
graceful fallback when the library is absent, and env-based configuration.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "5"))


class CrossEncoderReranker:
    """Reranks retrieval results using a cross-encoder model.

    The model is lazy-loaded on first call to `rerank()` so server startup
    is not blocked. If `sentence_transformers` is unavailable the reranker
    returns the original list truncated to `top_k` (no-op fallback).

    Args:
        model: HuggingFace model ID for the cross-encoder.
    """

    def __init__(self, model: str = RERANKER_MODEL) -> None:
        self._model_name = model
        self._model: Any = None

    def _load_model(self) -> Any:
        """Lazy-load the CrossEncoder model.

        Returns:
            Loaded CrossEncoder instance, or None if unavailable.
        """
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import]

            self._model = CrossEncoder(self._model_name)
            logger.info("CrossEncoder reranker loaded: %s", self._model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; reranking disabled. "
                "Install with: pip install sentence-transformers"
            )
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = RERANKER_TOP_K,
    ) -> list[dict[str, Any]]:
        """Rerank documents by relevance to query.

        Each document must have a `"content"` key with the text to score.
        On any failure the original list is returned truncated to `top_k`.

        Args:
            query: The search query string.
            documents: Retrieval results, each a dict containing `"content"`.
            top_k: Maximum number of results to return.

        Returns:
            Documents sorted by cross-encoder relevance, most relevant first.
        """
        if not documents:
            return documents

        model = self._load_model()
        if model is None:
            return documents[:top_k]

        try:
            pairs = [(query, doc["content"]) for doc in documents]
            scores = model.predict(pairs)
            ranked = sorted(zip(scores, documents, strict=False), key=lambda x: x[0], reverse=True)
            return [doc for _, doc in ranked[:top_k]]
        except Exception as exc:
            logger.warning("Reranking failed, returning original order: %s", exc)
            return documents[:top_k]


_reranker_instance: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker:
    """Return the module-level reranker singleton.

    Returns:
        Shared CrossEncoderReranker instance.
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance
