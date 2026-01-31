"""
Metrics system for BabyAGI.

Provides transparent instrumentation for all API calls (LLM, embeddings)
with cost tracking, latency measurement, and aggregation.

Usage:
    from metrics import (
        InstrumentedAnthropic,
        InstrumentedAsyncAnthropic,
        InstrumentedOpenAI,
        MetricsCollector,
        track_source,
        set_event_emitter,
    )

    # Swap clients at initialization
    client = InstrumentedAsyncAnthropic()
    set_event_emitter(agent)

    # Tag calls by source
    with track_source("extraction"):
        response = client.messages.create(...)
"""

from .clients import (
    InstrumentedAnthropic,
    InstrumentedAsyncAnthropic,
    InstrumentedOpenAI,
    set_event_emitter,
    track_source,
)
from .collector import MetricsCollector
from .costs import calculate_cost, calculate_embedding_cost, format_cost
from .models import EmbeddingCallMetric, LLMCallMetric, SessionMetrics, MetricsSummary

__all__ = [
    # Instrumented clients
    "InstrumentedAnthropic",
    "InstrumentedAsyncAnthropic",
    "InstrumentedOpenAI",
    # Configuration
    "set_event_emitter",
    "track_source",
    # Collector
    "MetricsCollector",
    # Cost calculation
    "calculate_cost",
    "calculate_embedding_cost",
    "format_cost",
    # Models
    "LLMCallMetric",
    "EmbeddingCallMetric",
    "SessionMetrics",
    "MetricsSummary",
]
