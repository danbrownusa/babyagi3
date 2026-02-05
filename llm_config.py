"""
LLM Configuration for multi-provider support using LiteLLM.

This module provides:
- Configurable models for different use cases (coding, research, agent, memory)
- Multi-provider support (OpenAI, Anthropic) via LiteLLM
- Automatic provider detection based on available API keys
- Instrumented clients for metrics tracking

Use cases:
- coding_model: Powerful model for code generation and analysis
- research_model: Model for research and information synthesis
- agent_model: General agent operations
- memory_model: Memory operations (extraction, summaries, learning)
- fast_model: Quick, cheap operations (classification, routing)
"""

import os
from dataclasses import dataclass, field
from typing import Any, Literal

# Check which API keys are available
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    model_id: str
    max_tokens: int = 4096
    temperature: float = 0.0
    provider: str = "auto"  # auto, openai, anthropic

    def get_litellm_model(self) -> str:
        """Get the model ID formatted for LiteLLM."""
        # LiteLLM uses prefixes for some providers
        # anthropic/ prefix for Anthropic models
        # openai/ or no prefix for OpenAI models
        if self.model_id.startswith("claude"):
            return self.model_id  # LiteLLM auto-detects claude models
        elif self.model_id.startswith("gpt") or self.model_id.startswith("o1"):
            return self.model_id  # LiteLLM auto-detects OpenAI models
        return self.model_id


@dataclass
class LLMConfig:
    """
    Configuration for all LLM models used in the system.

    Different use cases can use different models:
    - coding: Powerful model for code generation (e.g., Claude Opus, GPT-4)
    - research: Model for research tasks (e.g., Claude Sonnet, GPT-4)
    - agent: Main agent operations (e.g., Claude Sonnet, GPT-4)
    - memory: Memory operations like extraction, summaries (e.g., Claude Sonnet)
    - fast: Quick operations like classification (e.g., Claude Haiku, GPT-3.5)
    """

    # Model configurations for different use cases
    coding_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_id="claude-sonnet-4-20250514",
        max_tokens=8096,
        temperature=0.0,
    ))

    research_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_id="claude-sonnet-4-20250514",
        max_tokens=4096,
        temperature=0.0,
    ))

    agent_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_id="claude-sonnet-4-20250514",
        max_tokens=8096,
        temperature=0.0,
    ))

    memory_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_id="claude-sonnet-4-20250514",
        max_tokens=2048,
        temperature=0.0,
    ))

    fast_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_id="claude-3-5-haiku-20241022",
        max_tokens=1024,
        temperature=0.0,
    ))

    # Embedding configuration
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = "auto"  # auto, openai, voyage, local

    @classmethod
    def from_config(cls, config: dict) -> "LLMConfig":
        """Create LLMConfig from a configuration dictionary."""
        llm_config = config.get("llm", {})

        instance = cls()

        # Parse model configurations
        if "coding_model" in llm_config:
            instance.coding_model = cls._parse_model_config(llm_config["coding_model"])
        if "research_model" in llm_config:
            instance.research_model = cls._parse_model_config(llm_config["research_model"])
        if "agent_model" in llm_config:
            instance.agent_model = cls._parse_model_config(llm_config["agent_model"])
        if "memory_model" in llm_config:
            instance.memory_model = cls._parse_model_config(llm_config["memory_model"])
        if "fast_model" in llm_config:
            instance.fast_model = cls._parse_model_config(llm_config["fast_model"])

        # Embedding configuration
        if "embedding_model" in llm_config:
            instance.embedding_model = llm_config["embedding_model"]
        if "embedding_provider" in llm_config:
            instance.embedding_provider = llm_config["embedding_provider"]

        return instance

    @staticmethod
    def _parse_model_config(config: dict | str) -> ModelConfig:
        """Parse a model configuration from dict or string."""
        if isinstance(config, str):
            return ModelConfig(model_id=config)
        return ModelConfig(
            model_id=config.get("model", config.get("model_id", "claude-sonnet-4-20250514")),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.0),
            provider=config.get("provider", "auto"),
        )


def get_available_provider() -> str:
    """
    Detect which LLM provider is available based on API keys.

    Returns:
        'anthropic' if ANTHROPIC_API_KEY is set
        'openai' if OPENAI_API_KEY is set
        'none' if neither is set
    """
    if ANTHROPIC_API_KEY:
        return "anthropic"
    elif OPENAI_API_KEY:
        return "openai"
    return "none"


def get_default_models_for_provider(provider: str) -> dict[str, str]:
    """
    Get default model IDs for each use case based on provider.

    Args:
        provider: 'anthropic' or 'openai'

    Returns:
        Dictionary mapping use case to model ID
    """
    if provider == "anthropic":
        return {
            "coding": "claude-sonnet-4-20250514",
            "research": "claude-sonnet-4-20250514",
            "agent": "claude-sonnet-4-20250514",
            "memory": "claude-sonnet-4-20250514",
            "fast": "claude-3-5-haiku-20241022",
        }
    elif provider == "openai":
        return {
            "coding": "gpt-4o",
            "research": "gpt-4o",
            "agent": "gpt-4o",
            "memory": "gpt-4o-mini",
            "fast": "gpt-4o-mini",
        }
    else:
        # Default to Anthropic models (will fail without API key)
        return {
            "coding": "claude-sonnet-4-20250514",
            "research": "claude-sonnet-4-20250514",
            "agent": "claude-sonnet-4-20250514",
            "memory": "claude-sonnet-4-20250514",
            "fast": "claude-3-5-haiku-20241022",
        }


def create_default_config() -> LLMConfig:
    """
    Create a default LLMConfig based on available API keys.

    Automatically selects appropriate models based on which provider is available.
    """
    provider = get_available_provider()
    models = get_default_models_for_provider(provider)

    return LLMConfig(
        coding_model=ModelConfig(model_id=models["coding"], max_tokens=8096),
        research_model=ModelConfig(model_id=models["research"], max_tokens=4096),
        agent_model=ModelConfig(model_id=models["agent"], max_tokens=8096),
        memory_model=ModelConfig(model_id=models["memory"], max_tokens=2048),
        fast_model=ModelConfig(model_id=models["fast"], max_tokens=1024),
        embedding_model="text-embedding-3-small" if provider == "openai" else "text-embedding-3-small",
        embedding_provider=provider if provider != "none" else "local",
    )


# Global configuration instance
_llm_config: LLMConfig | None = None


def get_llm_config() -> LLMConfig:
    """Get the global LLM configuration."""
    global _llm_config
    if _llm_config is None:
        _llm_config = create_default_config()
    return _llm_config


def set_llm_config(config: LLMConfig):
    """Set the global LLM configuration."""
    global _llm_config
    _llm_config = config


def init_llm_config(config: dict | None = None):
    """
    Initialize the LLM configuration from a config dictionary.

    Args:
        config: Configuration dictionary (typically from config.yaml)
    """
    global _llm_config
    if config:
        _llm_config = LLMConfig.from_config(config)
    else:
        _llm_config = create_default_config()


# ═══════════════════════════════════════════════════════════
# LITELLM COMPLETION WRAPPER
# ═══════════════════════════════════════════════════════════


class LiteLLMClient:
    """
    Wrapper around LiteLLM for unified LLM access.

    Provides a consistent interface regardless of the underlying provider.
    Handles both sync and async calls.
    """

    def __init__(self, model_config: ModelConfig | None = None):
        self.model_config = model_config or get_llm_config().agent_model
        self._check_litellm()

    def _check_litellm(self):
        """Check if litellm is available."""
        try:
            import litellm
            self._litellm = litellm
        except ImportError:
            raise ImportError(
                "litellm is required for multi-provider LLM support. "
                "Install it with: pip install litellm"
            )

    def completion(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict] | None = None,
        **kwargs
    ) -> Any:
        """
        Make a synchronous completion call.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Override the default model
            max_tokens: Override max tokens
            temperature: Override temperature
            tools: Optional list of tool definitions
            **kwargs: Additional arguments passed to litellm

        Returns:
            LiteLLM response object (OpenAI-compatible format)
        """
        model = model or self.model_config.model_id
        max_tokens = max_tokens or self.model_config.max_tokens
        temperature = temperature if temperature is not None else self.model_config.temperature

        call_kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }

        if tools:
            call_kwargs["tools"] = tools

        return self._litellm.completion(**call_kwargs)

    async def acompletion(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict] | None = None,
        **kwargs
    ) -> Any:
        """
        Make an asynchronous completion call.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Override the default model
            max_tokens: Override max tokens
            temperature: Override temperature
            tools: Optional list of tool definitions
            **kwargs: Additional arguments passed to litellm

        Returns:
            LiteLLM response object (OpenAI-compatible format)
        """
        model = model or self.model_config.model_id
        max_tokens = max_tokens or self.model_config.max_tokens
        temperature = temperature if temperature is not None else self.model_config.temperature

        call_kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }

        if tools:
            call_kwargs["tools"] = tools

        return await self._litellm.acompletion(**call_kwargs)


def get_client_for_use_case(
    use_case: Literal["coding", "research", "agent", "memory", "fast"]
) -> LiteLLMClient:
    """
    Get a LiteLLM client configured for a specific use case.

    Args:
        use_case: One of 'coding', 'research', 'agent', 'memory', 'fast'

    Returns:
        Configured LiteLLMClient
    """
    config = get_llm_config()

    model_map = {
        "coding": config.coding_model,
        "research": config.research_model,
        "agent": config.agent_model,
        "memory": config.memory_model,
        "fast": config.fast_model,
    }

    return LiteLLMClient(model_map.get(use_case, config.agent_model))
