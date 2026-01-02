"""LLM Factory for creating LangChain LLM instances.

Provides centralized LLM creation with:
- Provider selection (OpenAI, Anthropic)
- Environment-based configuration
- Singleton pattern for efficiency
- Fallback handling

Following Enterprise Development Standards:
- Software Architect: Factory pattern, centralized configuration
- Software Engineer: Type-safe, environment-aware
"""

import os
from typing import Any

from langchain_core.language_models import BaseChatModel


# Cache for LLM instances
_llm_cache: dict[str, BaseChatModel] = {}


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> BaseChatModel:
    """Get or create an LLM instance.

    Args:
        provider: LLM provider ("openai" or "anthropic").
                  Defaults to env ENTERPRISE_AGENT_PROVIDER or auto-detect.
        model: Model name. Defaults to env ENTERPRISE_AGENT_MODEL.
        temperature: Model temperature.
        **kwargs: Additional model parameters.

    Returns:
        Configured LangChain chat model.

    Raises:
        ValueError: If no API key is configured.

    Example:
        >>> llm = get_llm()  # Auto-detect provider
        >>> llm = get_llm(provider="openai", model="gpt-4o-mini")
        >>> llm = get_llm(provider="anthropic", model="claude-3-haiku-20240307")
    """
    # Determine provider
    if provider is None:
        provider = os.getenv("ENTERPRISE_AGENT_PROVIDER", "").lower()

    # Auto-detect if not specified
    if not provider:
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            msg = "No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY."
            raise ValueError(msg)

    # Determine model
    if model is None:
        if provider == "openai":
            model = os.getenv("ENTERPRISE_AGENT_MODEL", "gpt-4o-mini")
        else:
            model = os.getenv("ENTERPRISE_AGENT_MODEL", "claude-3-haiku-20240307")

    # Cache key
    cache_key = f"{provider}:{model}:{temperature}"

    # Return cached instance if available
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    # Create new instance
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            **kwargs,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model=model,
            temperature=temperature,
            **kwargs,
        )
    else:
        msg = f"Unsupported provider: {provider}"
        raise ValueError(msg)

    # Cache and return
    _llm_cache[cache_key] = llm
    return llm


def get_embedding_model(
    provider: str | None = None,
    model: str | None = None,
) -> Any:
    """Get embedding model for vector operations.

    Args:
        provider: Embedding provider ("openai" or "huggingface").
        model: Model name.

    Returns:
        Configured embedding model.
    """
    if provider is None:
        provider = os.getenv("EMBEDDING_PROVIDER", "openai")

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        )
    elif provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=model or os.getenv(
                "HUGGINGFACE_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
        )
    else:
        msg = f"Unsupported embedding provider: {provider}"
        raise ValueError(msg)


def clear_llm_cache() -> None:
    """Clear the LLM instance cache.

    Useful for testing or when changing configurations.
    """
    global _llm_cache
    _llm_cache.clear()
