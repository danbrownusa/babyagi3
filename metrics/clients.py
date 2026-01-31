"""
Instrumented API clients for transparent metrics collection.

These are drop-in replacements for the standard Anthropic and OpenAI clients.
They automatically track all API calls, emitting events that the MetricsCollector
can subscribe to.

Usage:
    # Instead of:
    from anthropic import Anthropic
    client = Anthropic()

    # Use:
    from metrics import InstrumentedAnthropic, set_event_emitter
    client = InstrumentedAnthropic()
    set_event_emitter(agent)  # Agent is an EventEmitter

    # Tag calls by source:
    from metrics import track_source
    with track_source("extraction"):
        response = client.messages.create(...)
"""

import time
from contextlib import contextmanager
from typing import Any

from .costs import calculate_cost, calculate_embedding_cost, estimate_tokens


# ═══════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════

# Event emitter reference (set during initialization)
_event_emitter = None

# Current source context for categorizing calls
_current_source = "agent"


def set_event_emitter(emitter):
    """
    Set the global event emitter for metrics.

    Call this once during agent initialization:
        set_event_emitter(agent)  # Agent inherits from EventEmitter

    Args:
        emitter: Object with an emit(event_name, data) method
    """
    global _event_emitter
    _event_emitter = emitter


def get_event_emitter():
    """Get the current event emitter."""
    return _event_emitter


@contextmanager
def track_source(source: str):
    """
    Context manager to set the source of API calls.

    All LLM/embedding calls made within this context will be tagged
    with the specified source for cost breakdown analysis.

    Args:
        source: Source identifier ("agent", "extraction", "summary", "retrieval")

    Usage:
        with track_source("extraction"):
            # All LLM calls here are tagged as "extraction"
            response = client.messages.create(...)
    """
    global _current_source
    previous = _current_source
    _current_source = source
    try:
        yield
    finally:
        _current_source = previous


def get_current_source() -> str:
    """Get the current source context."""
    return _current_source


# ═══════════════════════════════════════════════════════════
# INSTRUMENTED ANTHROPIC CLIENT
# ═══════════════════════════════════════════════════════════


class InstrumentedMessages:
    """
    Wrapper for anthropic.Anthropic().messages that tracks calls.

    Transparently intercepts create() calls to emit metrics events.
    """

    def __init__(self, messages):
        self._messages = messages

    def create(self, **kwargs) -> Any:
        """Instrumented messages.create() call."""
        start_time = time.time()

        # Make the actual API call
        response = self._messages.create(**kwargs)

        duration_ms = int((time.time() - start_time) * 1000)

        # Extract metrics
        model = kwargs.get("model", "unknown")
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = calculate_cost(model, input_tokens, output_tokens)

        # Emit event
        if _event_emitter:
            _event_emitter.emit("llm_call_end", {
                "source": _current_source,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "duration_ms": duration_ms,
                "stop_reason": response.stop_reason,
            })

        return response


class InstrumentedAsyncMessages:
    """
    Async wrapper for anthropic.AsyncAnthropic().messages.

    Same as InstrumentedMessages but for async clients.
    """

    def __init__(self, messages):
        self._messages = messages

    async def create(self, **kwargs) -> Any:
        """Instrumented async messages.create() call."""
        start_time = time.time()

        # Make the actual API call
        response = await self._messages.create(**kwargs)

        duration_ms = int((time.time() - start_time) * 1000)

        # Extract metrics
        model = kwargs.get("model", "unknown")
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = calculate_cost(model, input_tokens, output_tokens)

        # Emit event
        if _event_emitter:
            _event_emitter.emit("llm_call_end", {
                "source": _current_source,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "duration_ms": duration_ms,
                "stop_reason": response.stop_reason,
            })

        return response


class InstrumentedAnthropic:
    """
    Drop-in replacement for anthropic.Anthropic with automatic metrics.

    Usage:
        # Instead of:
        client = anthropic.Anthropic()

        # Use:
        client = InstrumentedAnthropic()

        # Everything else works the same
        response = client.messages.create(...)
    """

    def __init__(self, **kwargs):
        import anthropic
        self._client = anthropic.Anthropic(**kwargs)
        self._messages = InstrumentedMessages(self._client.messages)

    @property
    def messages(self):
        return self._messages


class InstrumentedAsyncAnthropic:
    """
    Drop-in replacement for anthropic.AsyncAnthropic with automatic metrics.

    Usage:
        client = InstrumentedAsyncAnthropic()
        response = await client.messages.create(...)
    """

    def __init__(self, **kwargs):
        import anthropic
        self._client = anthropic.AsyncAnthropic(**kwargs)
        self._messages = InstrumentedAsyncMessages(self._client.messages)

    @property
    def messages(self):
        return self._messages


# ═══════════════════════════════════════════════════════════
# INSTRUMENTED OPENAI CLIENT (for Embeddings)
# ═══════════════════════════════════════════════════════════


class InstrumentedEmbeddings:
    """
    Wrapper for openai.OpenAI().embeddings that tracks calls.

    Transparently intercepts create() calls to emit metrics events.
    """

    def __init__(self, embeddings):
        self._embeddings = embeddings

    def create(self, **kwargs) -> Any:
        """Instrumented embeddings.create() call."""
        start_time = time.time()

        # Make the actual API call
        response = self._embeddings.create(**kwargs)

        duration_ms = int((time.time() - start_time) * 1000)

        # Calculate metrics
        model = kwargs.get("model", "text-embedding-3-small")
        input_data = kwargs.get("input", "")

        if isinstance(input_data, list):
            text_count = len(input_data)
            token_estimate = sum(estimate_tokens(t) for t in input_data)
        else:
            text_count = 1
            token_estimate = estimate_tokens(input_data)

        cost = calculate_embedding_cost(model, token_estimate)

        # Emit event
        if _event_emitter:
            _event_emitter.emit("embedding_call_end", {
                "provider": "openai",
                "model": model,
                "text_count": text_count,
                "token_estimate": token_estimate,
                "cost_usd": cost,
                "duration_ms": duration_ms,
                "cached": False,
            })

        return response


class InstrumentedOpenAI:
    """
    Drop-in replacement for openai.OpenAI with automatic metrics.

    Currently only instruments the embeddings API.

    Usage:
        # Instead of:
        client = OpenAI()

        # Use:
        client = InstrumentedOpenAI()

        # Everything else works the same
        response = client.embeddings.create(...)
    """

    def __init__(self, **kwargs):
        from openai import OpenAI
        self._client = OpenAI(**kwargs)
        self._embeddings = InstrumentedEmbeddings(self._client.embeddings)

    @property
    def embeddings(self):
        return self._embeddings


# ═══════════════════════════════════════════════════════════
# CACHE-AWARE EMBEDDING TRACKING
# ═══════════════════════════════════════════════════════════


def emit_embedding_cache_hit(model: str, text_count: int, token_estimate: int):
    """
    Emit an event for cached embedding lookups.

    Call this when embeddings are retrieved from cache instead of API.
    This helps track cache hit rates for cost analysis.

    Args:
        model: The model that would have been used
        text_count: Number of texts that were cached
        token_estimate: Estimated tokens saved
    """
    if _event_emitter:
        _event_emitter.emit("embedding_call_end", {
            "provider": "cache",
            "model": model,
            "text_count": text_count,
            "token_estimate": token_estimate,
            "cost_usd": 0.0,
            "duration_ms": 0,
            "cached": True,
        })
