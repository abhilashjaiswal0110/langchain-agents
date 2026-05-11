"""LLM Factory for creating LangChain LLM instances.

Provides centralized LLM creation with:
- Provider selection (Azure OpenAI, OpenAI, Anthropic)
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

    Provider priority (when auto-detecting):
    1. Azure OpenAI (primary - for production)
    2. OpenAI (disabled by default - secondary)
    3. Anthropic (fallback)

    Args:
        provider: LLM provider ("azure_openai", "openai", or "anthropic").
                  Defaults to env ENTERPRISE_AGENT_PROVIDER or auto-detect.
        model: Model name. Defaults to env ENTERPRISE_AGENT_MODEL.
               For Azure OpenAI, this is the deployment name.
        temperature: Model temperature.
        **kwargs: Additional model parameters.

    Returns:
        Configured LangChain chat model.

    Raises:
        ValueError: If no API key is configured.

    Example:
        >>> llm = get_llm()  # Auto-detect provider (Azure OpenAI first)
        >>> llm = get_llm(provider="azure_openai")  # Explicit Azure OpenAI
        >>> llm = get_llm(provider="openai", model="gpt-4o-mini")
        >>> llm = get_llm(provider="anthropic", model="claude-3-haiku-20240307")
    """
    # Determine provider
    if provider is None:
        provider = os.getenv("ENTERPRISE_AGENT_PROVIDER", "").lower()

    # Auto-detect if not specified (Azure OpenAI is primary)
    if not provider:
        if _is_azure_openai_configured():
            provider = "azure_openai"
        elif os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_ENABLED", "false").lower() == "true":
            # OpenAI only if explicitly enabled (disabled by default)
            provider = "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            msg = (
                "No LLM API key found. Configure one of the following:\n"
                "- Azure OpenAI: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME\n"
                "- OpenAI: OPENAI_API_KEY and OPENAI_ENABLED=true\n"
                "- Anthropic: ANTHROPIC_API_KEY"
            )
            raise ValueError(msg)

    # Determine model/deployment name
    if model is None:
        if provider == "azure_openai":
            model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", os.getenv("ENTERPRISE_AGENT_MODEL", "gpt-4o-mini"))
        elif provider == "openai":
            model = os.getenv("ENTERPRISE_AGENT_MODEL", "gpt-4o-mini")
        else:
            model = os.getenv("ENTERPRISE_AGENT_MODEL", "claude-3-haiku-20240307")

    # Cache key
    cache_key = f"{provider}:{model}:{temperature}"

    # Return cached instance if available
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    # Create new instance
    if provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI

        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        # LangChain uses OPENAI_API_VERSION for Azure, but we support both for convenience
        api_version = os.getenv("OPENAI_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

        if not azure_endpoint or not api_key:
            msg = "Azure OpenAI requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"
            raise ValueError(msg)

        if not model:
            msg = (
                "Azure OpenAI requires AZURE_OPENAI_DEPLOYMENT_NAME. "
                "This is the deployment name from your Azure OpenAI resource, not the model name."
            )
            raise ValueError(msg)

        # Azure OpenAI reasoning models (o1, o3, o4 series) don't support temperature
        # Check deployment name for reasoning model indicators
        reasoning_models = ("o1", "o3", "o4", "o1-mini", "o3-mini", "o4-mini")
        is_reasoning_model = any(model.lower().startswith(prefix) for prefix in reasoning_models)

        if is_reasoning_model:
            print(
                f"[LLM Factory] Using Azure OpenAI reasoning model: deployment={model}, endpoint={azure_endpoint[:50]}..., api_version={api_version} (temperature not supported)"
            )

            try:
                llm = AzureChatOpenAI(
                    azure_deployment=model,
                    azure_endpoint=azure_endpoint,
                    api_key=api_key,
                    api_version=api_version,
                    **kwargs,
                )
            except Exception as e:
                msg = (
                    f"Failed to create Azure OpenAI reasoning model client: {e}\n\n"
                    f"Please verify:\n"
                    f"1. AZURE_OPENAI_ENDPOINT is correct (e.g., https://your-resource.openai.azure.com/)\n"
                    f"2. AZURE_OPENAI_DEPLOYMENT_NAME matches your reasoning model deployment in Azure Portal\n"
                    f"3. OPENAI_API_VERSION is supported (try: 2024-08-01-preview or 2024-02-15-preview)\n"
                    f"4. Your reasoning model deployment '{model}' exists and is running in your Azure OpenAI resource"
                )
                raise ValueError(msg) from e
        else:
            print(
                f"[LLM Factory] Using Azure OpenAI: deployment={model}, endpoint={azure_endpoint[:50]}..., api_version={api_version}"
            )

            try:
                llm = AzureChatOpenAI(
                    azure_deployment=model,
                    azure_endpoint=azure_endpoint,
                    api_key=api_key,
                    api_version=api_version,
                    temperature=temperature,
                    **kwargs,
                )
            except Exception as e:
                msg = (
                    f"Failed to create Azure OpenAI client: {e}\n\n"
                    f"Please verify:\n"
                    f"1. AZURE_OPENAI_ENDPOINT is correct (e.g., https://your-resource.openai.azure.com/)\n"
                    f"2. AZURE_OPENAI_DEPLOYMENT_NAME matches your deployment in Azure Portal\n"
                    f"3. OPENAI_API_VERSION is supported (try: 2024-08-01-preview or 2024-02-15-preview)\n"
                    f"4. Your deployment '{model}' exists in your Azure OpenAI resource"
                )
                raise ValueError(msg) from e
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        # OpenAI reasoning models (o1, o3, o4 series) don't support temperature
        # They only accept the default value of 1
        reasoning_models = ("o1", "o3", "o4", "o1-mini", "o3-mini", "o4-mini")
        is_reasoning_model = any(model.startswith(prefix) for prefix in reasoning_models)

        if is_reasoning_model:
            print(f"[LLM Factory] Using OpenAI reasoning model {model} (temperature not supported)")
            llm = ChatOpenAI(model=model, **kwargs)
        else:
            print(f"[LLM Factory] Using OpenAI: model={model}")
            llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                **kwargs,
            )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        print(f"[LLM Factory] Using Anthropic: model={model}")
        llm = ChatAnthropic(
            model=model,
            temperature=temperature,
            **kwargs,
        )
    else:
        msg = f"Unsupported provider: {provider}. Supported: azure_openai, openai, anthropic"
        raise ValueError(msg)

    # Cache and return
    _llm_cache[cache_key] = llm
    return llm


def _is_azure_openai_configured() -> bool:
    """Check if Azure OpenAI is fully configured.

    Returns:
        True if all required Azure OpenAI env vars are set.
    """
    return all(
        [
            os.getenv("AZURE_OPENAI_API_KEY"),
            os.getenv("AZURE_OPENAI_ENDPOINT"),
            os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        ]
    )


def get_embedding_model(
    provider: str | None = None,
    model: str | None = None,
) -> Any:
    """Get embedding model for vector operations.

    Provider priority (when auto-detecting):
    1. Azure OpenAI (primary - for production)
    2. OpenAI (disabled by default)
    3. HuggingFace (local fallback)

    Args:
        provider: Embedding provider ("azure_openai", "openai", or "huggingface").
        model: Model name or deployment name for Azure.

    Returns:
        Configured embedding model.
    """
    if provider is None:
        provider = os.getenv("EMBEDDING_PROVIDER", "").lower()

    # Auto-detect if not specified (Azure OpenAI is primary)
    if not provider:
        if _is_azure_openai_embedding_configured():
            provider = "azure_openai"
        elif os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_ENABLED", "false").lower() == "true":
            provider = "openai"
        else:
            # Default to huggingface if no cloud provider configured
            provider = "huggingface"

    if provider == "azure_openai":
        from langchain_openai import AzureOpenAIEmbeddings

        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        # LangChain uses OPENAI_API_VERSION for Azure, but we support both for convenience
        api_version = os.getenv("OPENAI_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        deployment = model or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

        if not azure_endpoint or not api_key:
            msg = "Azure OpenAI Embeddings requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"
            raise ValueError(msg)

        print(f"[Embedding Factory] Using Azure OpenAI Embeddings: deployment={deployment}, api_version={api_version}")

        try:
            return AzureOpenAIEmbeddings(
                azure_deployment=deployment,
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
            )
        except Exception as e:
            msg = (
                f"Failed to create Azure OpenAI Embeddings client: {e}\n\n"
                f"Please verify:\n"
                f"1. AZURE_OPENAI_ENDPOINT is correct\n"
                f"2. AZURE_OPENAI_EMBEDDING_DEPLOYMENT exists in your Azure OpenAI resource\n"
                f"3. OPENAI_API_VERSION is supported\n"
                f"4. Your embedding deployment '{deployment}' is deployed and running"
            )
            raise ValueError(msg) from e
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        embedding_model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        print(f"[Embedding Factory] Using OpenAI Embeddings: model={embedding_model}")
        return OpenAIEmbeddings(model=embedding_model)
    elif provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        hf_model = model or os.getenv(
            "HUGGINGFACE_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        print(f"[Embedding Factory] Using HuggingFace Embeddings: model={hf_model}")
        return HuggingFaceEmbeddings(model_name=hf_model)
    else:
        msg = f"Unsupported embedding provider: {provider}. Supported: azure_openai, openai, huggingface"
        raise ValueError(msg)


def _is_azure_openai_embedding_configured() -> bool:
    """Check if Azure OpenAI Embeddings is configured.

    Returns:
        True if Azure OpenAI endpoint and key are set, and embedding deployment exists.
    """
    return all(
        [
            os.getenv("AZURE_OPENAI_API_KEY"),
            os.getenv("AZURE_OPENAI_ENDPOINT"),
            os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        ]
    )


def clear_llm_cache() -> None:
    """Clear the LLM instance cache.

    Useful for testing or when changing configurations.
    """
    global _llm_cache
    _llm_cache.clear()
