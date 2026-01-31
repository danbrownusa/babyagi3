# Metrics & Performance Tracking Implementation Plan

## Executive Summary

This plan describes an elegant, minimalist approach to adding comprehensive metrics tracking to BabyAGI. The design leverages the existing event system and memory infrastructure to track AI calls, tool executions, and session-level performance with minimal code changes.

---

## Current State Analysis

### What Exists

| Component | Location | Status |
|-----------|----------|--------|
| EventEmitter | `utils/events.py` | ✅ Robust, supports `tool_start`, `tool_end`, etc. |
| Tool Statistics | `memory/models.py:ToolDefinition` | ✅ Tracks usage_count, success_count, avg_duration_ms |
| Memory Hooks | `memory/integration.py` | ✅ Logs tool calls to SQLite |
| Memory Store | `memory/store.py` | ✅ SQLite backend with `record_tool_success/error` |

### What's Missing

| Feature | Impact |
|---------|--------|
| **AI/LLM Call Tracking** | No token counts, costs, or latency for Claude API calls |
| **Embedding Call Tracking** | No tracking of OpenAI/Voyage embedding API calls |
| **Memory System Call Tracking** | Extraction, summarization, and retrieval use Claude but aren't tracked |
| **Cost Calculation** | No infrastructure to calculate $ costs from tokens |
| **Session Metrics** | No per-session/thread aggregation |
| **Unified Query Interface** | Metrics scattered, no easy summary access |
| **Agent Introspection Tools** | Agent can't query its own performance |

### All API Call Locations (Complete Inventory)

| Component | File | Method | Model/API | Purpose |
|-----------|------|--------|-----------|---------|
| **Main Agent Loop** | `agent.py:771` | `run_async()` | Claude Sonnet | Primary agent reasoning |
| **Embeddings** | `memory/embeddings.py:234` | `OpenAIEmbeddings.embed()` | OpenAI text-embedding-3-small | Semantic vectors |
| **Embeddings** | `memory/embeddings.py:271` | `OpenAIEmbeddings.embed_batch()` | OpenAI text-embedding-3-small | Batch vectors |
| **Embeddings** | `memory/embeddings.py:304` | `AnthropicVoyageEmbeddings.embed()` | Voyage voyage-2 | Alt embeddings |
| **Extraction** | `memory/extraction.py:269` | `_llm_extract()` | Claude Sonnet | Entity/relationship extraction |
| **Extraction** | `memory/extraction.py:495` | `_cluster_entity_type_llm()` | Claude Haiku | Type classification |
| **Extraction** | `memory/extraction.py:541` | `_cluster_relation_type_llm()` | Claude Haiku | Relation classification |
| **Summaries** | `memory/summaries.py:263` | `_refresh_leaf_node()` | Claude Sonnet | Entity/topic summaries |
| **Summaries** | `memory/summaries.py:314` | `_refresh_branch_node()` | Claude Sonnet | Category synthesis |
| **Summaries** | `memory/summaries.py:367` | `_refresh_root_node()` | Claude Sonnet | High-level overview |
| **Deep Retrieval** | `memory/retrieval.py:811` | `DeepRetrievalAgent.run()` | Claude Sonnet | Agentic memory search |

---

## Design Philosophy

### Core Principles

1. **Event-Driven**: Leverage existing EventEmitter - don't create parallel tracking
2. **Single Source of Truth**: Store in memory system's SQLite, not separate database
3. **Lazy Aggregation**: Compute summaries on-demand, not continuously
4. **Non-Blocking**: Never slow down agent operations for metrics
5. **Agent-Accessible**: First-class tools for the agent to understand its performance

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ALL API CALL SOURCES                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐  │
│  │   Agent Loop     │  │  Memory System   │  │      Embeddings          │  │
│  │   (agent.py)     │  │                  │  │  (memory/embeddings.py)  │  │
│  │                  │  │  ┌────────────┐  │  │                          │  │
│  │  run_async()     │  │  │ Extraction │  │  │  OpenAI embed()          │  │
│  │  Claude Sonnet   │  │  │ Claude     │  │  │  Voyage embed()          │  │
│  │                  │  │  └────────────┘  │  │                          │  │
│  │  Tool Execution  │  │  ┌────────────┐  │  └──────────────────────────┘  │
│  │                  │  │  │ Summaries  │  │                                │
│  └────────┬─────────┘  │  │ Claude     │  │                                │
│           │            │  └────────────┘  │                                │
│           │            │  ┌────────────┐  │                                │
│           │            │  │ Retrieval  │  │                                │
│           │            │  │ Claude     │  │                                │
│           │            │  └────────────┘  │                                │
│           │            └────────┬─────────┘                                │
│           │                     │                                          │
│           └──────────┬──────────┘                                          │
│                      ▼                                                     │
│         ┌────────────────────────────────────────┐                         │
│         │         Instrumented API Client        │◀─── Key Innovation     │
│         │         (metrics/client.py)            │                         │
│         ├────────────────────────────────────────┤                         │
│         │  Wraps Anthropic/OpenAI clients        │                         │
│         │  Emits events for ALL API calls        │                         │
│         │  Tracks tokens, latency, cost          │                         │
│         │  Zero changes to calling code          │                         │
│         └────────────────┬───────────────────────┘                         │
│                          │ emit()                                          │
│                          ▼                                                 │
│         ┌────────────────────────────────────────┐                         │
│         │           EventEmitter                 │                         │
│         │  • llm_call_start / llm_call_end       │                         │
│         │  • embedding_call_start / end          │                         │
│         │  • tool_start / tool_end               │                         │
│         └────────────────┬───────────────────────┘                         │
└──────────────────────────│─────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        MetricsCollector (NEW)                                │
│                        metrics/collector.py                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│  • Subscribes to ALL metric events                                           │
│  • Categorizes by source: agent, extraction, summary, retrieval, embedding   │
│  • Stores to SQLite via memory.store                                         │
│  • Provides rich query/aggregation methods                                   │
└──────────────────────────┬───────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        SQLite (Extended Schema)                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  NEW TABLES:                                                                 │
│  • llm_calls: source, model, tokens_in, tokens_out, cost, latency_ms         │
│  • embedding_calls: provider, model, text_count, latency_ms, cached          │
│  • session_metrics: aggregated per-thread stats                              │
└──────────────────────────┬───────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Agent Tools (NEW)                                     │
│                        tools/metrics.py                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│  get_metrics(action, ...)                                                    │
│  • "summary" - Overall performance summary                                   │
│  • "costs" - Cost breakdown by model/source/period                           │
│  • "tools" - Tool performance stats                                          │
│  • "memory" - Memory system costs (extraction, summaries, retrieval)         │
│  • "embeddings" - Embedding API usage and cache hit rates                    │
│  • "session" - Current session metrics                                       │
│  • "slow" - Identify bottlenecks                                             │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decision: Instrumented Client Wrapper

Instead of modifying every file that makes API calls, we create **wrapper clients** that automatically track all calls. This is the most elegant approach because:

1. **Zero changes to existing code** - Just swap the client at initialization
2. **Impossible to miss calls** - All API traffic goes through the wrapper
3. **Single point of control** - Easy to add new metrics later
4. **Transparent** - Calling code doesn't know it's being tracked

---

## Implementation Plan

### Phase 0: Instrumented Client Wrappers (Foundation)

The most elegant approach is to create wrapper clients that transparently track all API calls. This requires minimal changes to existing code.

#### 0.1 Instrumented Anthropic Client

**File**: `metrics/clients.py` (NEW)

```python
"""
Instrumented API clients that automatically track all calls.

These wrappers are transparent - calling code doesn't know it's being tracked.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable
from functools import wraps

import anthropic

from .costs import calculate_cost


@dataclass
class LLMCallEvent:
    """Event data for an LLM API call."""
    source: str  # "agent", "extraction", "summary", "retrieval"
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_ms: int
    stop_reason: str
    metadata: dict | None = None


# Global event emitter reference (set during initialization)
_event_emitter = None
_current_source = "unknown"  # Context for categorizing calls


def set_event_emitter(emitter):
    """Set the global event emitter for metrics."""
    global _event_emitter
    _event_emitter = emitter


def set_call_source(source: str):
    """Set the current call source context."""
    global _current_source
    _current_source = source


class InstrumentedMessages:
    """Wrapper for anthropic.Anthropic().messages that tracks calls."""

    def __init__(self, messages, source: str = "unknown"):
        self._messages = messages
        self._source = source

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
    """Async wrapper for anthropic.AsyncAnthropic().messages."""

    def __init__(self, messages, source: str = "unknown"):
        self._messages = messages
        self._source = source

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
        self._client = anthropic.Anthropic(**kwargs)
        self._messages = InstrumentedMessages(self._client.messages)

    @property
    def messages(self):
        return self._messages


class InstrumentedAsyncAnthropic:
    """Async version of InstrumentedAnthropic."""

    def __init__(self, **kwargs):
        self._client = anthropic.AsyncAnthropic(**kwargs)
        self._messages = InstrumentedAsyncMessages(self._client.messages)

    @property
    def messages(self):
        return self._messages
```

#### 0.2 Instrumented OpenAI Client (for Embeddings)

**Add to**: `metrics/clients.py`

```python
from openai import OpenAI

# Embedding pricing per 1M tokens
EMBEDDING_PRICING = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


class InstrumentedEmbeddings:
    """Wrapper for openai.OpenAI().embeddings that tracks calls."""

    def __init__(self, embeddings):
        self._embeddings = embeddings

    def create(self, **kwargs) -> Any:
        """Instrumented embeddings.create() call."""
        start_time = time.time()

        # Make the actual API call
        response = self._embeddings.create(**kwargs)

        duration_ms = int((time.time() - start_time) * 1000)

        # Calculate cost
        model = kwargs.get("model", "text-embedding-3-small")
        input_text = kwargs.get("input", "")
        if isinstance(input_text, list):
            text_count = len(input_text)
            # Rough token estimate: ~4 chars per token
            token_estimate = sum(len(t) // 4 for t in input_text)
        else:
            text_count = 1
            token_estimate = len(input_text) // 4

        price_per_million = EMBEDDING_PRICING.get(model, 0.02)
        cost = (token_estimate / 1_000_000) * price_per_million

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
    """Drop-in replacement for openai.OpenAI with automatic metrics."""

    def __init__(self, **kwargs):
        self._client = OpenAI(**kwargs)
        self._embeddings = InstrumentedEmbeddings(self._client.embeddings)

    @property
    def embeddings(self):
        return self._embeddings
```

#### 0.3 Context Manager for Call Source

**Add to**: `metrics/clients.py`

```python
from contextlib import contextmanager


@contextmanager
def track_source(source: str):
    """
    Context manager to set the source of API calls.

    Usage:
        with track_source("extraction"):
            # All LLM calls in this block are tagged as "extraction"
            response = client.messages.create(...)
    """
    global _current_source
    previous = _current_source
    _current_source = source
    try:
        yield
    finally:
        _current_source = previous
```

#### 0.4 Integration Points

**Minimal changes required**:

1. **agent.py** (line ~167):
   ```python
   # Before:
   self.client = anthropic.AsyncAnthropic()

   # After:
   from metrics.clients import InstrumentedAsyncAnthropic, set_event_emitter
   self.client = InstrumentedAsyncAnthropic()
   set_event_emitter(self)  # Agent is an EventEmitter
   ```

2. **memory/extraction.py** (line ~57):
   ```python
   # Before:
   self._client = anthropic.Anthropic()

   # After:
   from metrics.clients import InstrumentedAnthropic, track_source
   self._client = InstrumentedAnthropic()

   # In _llm_extract:
   with track_source("extraction"):
       response = self.client.messages.create(...)
   ```

3. **memory/summaries.py** (line ~149):
   ```python
   # Before:
   self._client = anthropic.Anthropic()

   # After:
   from metrics.clients import InstrumentedAnthropic, track_source
   self._client = InstrumentedAnthropic()

   # In refresh methods:
   with track_source("summary"):
       response = self.client.messages.create(...)
   ```

4. **memory/retrieval.py** (line ~458):
   ```python
   # Before:
   self._client = anthropic.Anthropic()

   # After:
   from metrics.clients import InstrumentedAnthropic, track_source
   self._client = InstrumentedAnthropic()

   # In run():
   with track_source("retrieval"):
       response = self.client.messages.create(...)
   ```

5. **memory/embeddings.py** (line ~215):
   ```python
   # Before:
   self._client = OpenAI()

   # After:
   from metrics.clients import InstrumentedOpenAI
   self._client = InstrumentedOpenAI()
   ```

---

### Phase 1: Core Infrastructure

#### 1.1 Add AI Call Events to Agent

**File**: `agent.py` (modify `run_async` method around line 771)

```python
# Before API call
self.emit("ai_call_start", {
    "model": self.model,
    "thread_id": thread_id,
    "message_count": len(thread),
    "tool_count": len(self._tool_schemas()),
})

response = await self.client.messages.create(...)

# After API call
self.emit("ai_call_end", {
    "model": self.model,
    "thread_id": thread_id,
    "input_tokens": response.usage.input_tokens,
    "output_tokens": response.usage.output_tokens,
    "stop_reason": response.stop_reason,
    "duration_ms": duration_ms,
})
```

**Changes Required**:
- Wrap the `messages.create()` call with timing
- Extract token usage from response
- Emit events before/after

#### 1.2 Create Metrics Data Models

**File**: `metrics/models.py` (NEW)

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class AICallMetric:
    """Record of a single AI/LLM API call."""
    id: str
    timestamp: datetime

    # Call details
    model: str
    thread_id: str

    # Token usage
    input_tokens: int
    output_tokens: int
    total_tokens: int

    # Cost (calculated)
    cost_usd: float

    # Performance
    duration_ms: int
    tokens_per_second: float

    # Context
    stop_reason: str  # "end_turn", "tool_use", etc.
    tool_calls: int = 0  # How many tools called in this response


@dataclass
class ToolMetric:
    """Record of a single tool execution."""
    id: str
    timestamp: datetime

    tool_name: str
    thread_id: str

    duration_ms: int
    success: bool
    error_message: str | None = None

    # Input/output sizes (for understanding data flow)
    input_size_bytes: int = 0
    output_size_bytes: int = 0


@dataclass
class SessionMetrics:
    """Aggregated metrics for a session/thread."""
    thread_id: str
    started_at: datetime
    last_activity: datetime

    # AI calls
    ai_call_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_ai_cost_usd: float = 0.0
    avg_ai_latency_ms: float = 0.0

    # Tools
    tool_call_count: int = 0
    tool_success_count: int = 0
    tool_error_count: int = 0
    avg_tool_latency_ms: float = 0.0
    tools_used: dict[str, int] = field(default_factory=dict)  # tool -> count

    # Overall
    total_duration_ms: int = 0


@dataclass
class MetricsSummary:
    """High-level summary across all sessions."""
    period_start: datetime
    period_end: datetime

    # Totals
    total_ai_calls: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_tool_calls: int = 0

    # Averages
    avg_cost_per_session: float = 0.0
    avg_tokens_per_session: float = 0.0
    avg_ai_latency_ms: float = 0.0
    avg_tool_latency_ms: float = 0.0

    # Breakdowns
    cost_by_model: dict[str, float] = field(default_factory=dict)
    calls_by_tool: dict[str, int] = field(default_factory=dict)
    errors_by_tool: dict[str, int] = field(default_factory=dict)

    # Top items
    slowest_tools: list[tuple[str, float]] = field(default_factory=list)
    most_expensive_sessions: list[tuple[str, float]] = field(default_factory=list)
```

#### 1.3 Cost Calculator

**File**: `metrics/costs.py` (NEW)

```python
"""
Token cost calculation for various AI models.

Prices as of 2025 (update as needed).
"""

# Pricing per 1M tokens (input, output)
MODEL_PRICING = {
    # Claude 4
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-opus-4-20250514": (15.00, 75.00),

    # Claude 3.5
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-haiku-3-5-20241022": (0.80, 4.00),  # Alternate name

    # Claude 3
    "claude-3-opus-20240229": (15.00, 75.00),
    "claude-3-sonnet-20240229": (3.00, 15.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
}

# Embedding pricing per 1M tokens
EMBEDDING_PRICING = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
    "voyage-2": 0.10,
    "voyage-large-2": 0.12,
}

# Default fallback for unknown models
DEFAULT_PRICING = (3.00, 15.00)


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate cost in USD for an API call.

    Args:
        model: Model name/ID
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in USD (float)
    """
    input_price, output_price = MODEL_PRICING.get(model, DEFAULT_PRICING)

    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price

    return round(input_cost + output_cost, 6)


def format_cost(cost_usd: float) -> str:
    """Format cost for display."""
    if cost_usd < 0.01:
        return f"${cost_usd:.4f}"
    elif cost_usd < 1.00:
        return f"${cost_usd:.3f}"
    else:
        return f"${cost_usd:.2f}"
```

#### 1.4 Metrics Collector

**File**: `metrics/collector.py` (NEW)

```python
"""
Central metrics collection system.

Subscribes to agent events and stores metrics in the database.
Provides query methods for aggregation and analysis.
"""

import time
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from .models import AICallMetric, ToolMetric, SessionMetrics, MetricsSummary
from .costs import calculate_cost


class MetricsCollector:
    """
    Collects and stores metrics from agent events.

    Usage:
        collector = MetricsCollector(memory_store)
        collector.attach(agent)  # Subscribe to events

        # Later...
        summary = collector.get_summary()
        session = collector.get_session_metrics(thread_id)
    """

    def __init__(self, store):
        """
        Initialize collector with a memory store.

        Args:
            store: MemoryStore instance for persistence
        """
        self.store = store
        self._in_flight_ai_calls: dict[str, dict] = {}  # thread_id -> start data
        self._in_flight_tools: dict[str, dict] = {}  # tool_use_id -> start data

    def attach(self, agent):
        """
        Attach to an agent's event system.

        Args:
            agent: Agent instance with EventEmitter
        """
        agent.on("ai_call_start", self._on_ai_call_start)
        agent.on("ai_call_end", self._on_ai_call_end)
        agent.on("tool_start", self._on_tool_start)
        agent.on("tool_end", self._on_tool_end)

    def _on_ai_call_start(self, event: dict):
        """Track start of AI call for latency calculation."""
        thread_id = event.get("thread_id", "default")
        self._in_flight_ai_calls[thread_id] = {
            "start_time": time.time(),
            "model": event.get("model"),
            "message_count": event.get("message_count", 0),
        }

    def _on_ai_call_end(self, event: dict):
        """Record completed AI call."""
        thread_id = event.get("thread_id", "default")
        start_data = self._in_flight_ai_calls.pop(thread_id, {})

        # Calculate duration
        start_time = start_data.get("start_time", time.time())
        duration_ms = event.get("duration_ms", int((time.time() - start_time) * 1000))

        input_tokens = event.get("input_tokens", 0)
        output_tokens = event.get("output_tokens", 0)
        model = event.get("model", start_data.get("model", "unknown"))

        # Calculate cost
        cost = calculate_cost(model, input_tokens, output_tokens)

        # Calculate tokens per second
        total_tokens = input_tokens + output_tokens
        tps = (total_tokens / (duration_ms / 1000)) if duration_ms > 0 else 0

        # Create metric record
        metric = AICallMetric(
            id=str(uuid4()),
            timestamp=datetime.now(),
            model=model,
            thread_id=thread_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            duration_ms=duration_ms,
            tokens_per_second=tps,
            stop_reason=event.get("stop_reason", "unknown"),
        )

        # Store in database
        self.store.record_ai_call(metric)

    def _on_tool_start(self, event: dict):
        """Track start of tool execution."""
        # Use a unique key based on name + timestamp
        key = f"{event.get('name')}_{time.time()}"
        self._in_flight_tools[key] = {
            "start_time": time.time(),
            "name": event.get("name"),
            "input": event.get("input"),
            "thread_id": getattr(event.get("_agent"), "_current_thread_id", "default"),
        }
        # Store key for matching in tool_end
        event["_metrics_key"] = key

    def _on_tool_end(self, event: dict):
        """Record completed tool execution."""
        tool_name = event.get("name")
        duration_ms = event.get("duration_ms", 0)
        result = event.get("result")

        # Determine success/error
        is_error = False
        error_msg = None
        if isinstance(result, dict):
            if "error" in result:
                is_error = True
                error_msg = str(result["error"])
            elif result.get("status") == "error":
                is_error = True
                error_msg = result.get("message", "Unknown error")

        # Create metric record
        metric = ToolMetric(
            id=str(uuid4()),
            timestamp=datetime.now(),
            tool_name=tool_name,
            thread_id="default",  # TODO: Get from context
            duration_ms=duration_ms,
            success=not is_error,
            error_message=error_msg,
            input_size_bytes=len(str(event.get("input", ""))),
            output_size_bytes=len(str(result or "")),
        )

        # Store in database
        self.store.record_tool_metric(metric)

    # ═══════════════════════════════════════════════════════════
    # QUERY METHODS
    # ═══════════════════════════════════════════════════════════

    def get_summary(
        self,
        period: str = "all",  # "hour", "day", "week", "month", "all"
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> MetricsSummary:
        """
        Get aggregated metrics summary.

        Args:
            period: Time period to summarize
            start_time: Custom start time (overrides period)
            end_time: Custom end time (overrides period)

        Returns:
            MetricsSummary with aggregated data
        """
        # Calculate time bounds
        now = datetime.now()
        if start_time is None:
            if period == "hour":
                start_time = now - timedelta(hours=1)
            elif period == "day":
                start_time = now - timedelta(days=1)
            elif period == "week":
                start_time = now - timedelta(weeks=1)
            elif period == "month":
                start_time = now - timedelta(days=30)
            else:
                start_time = datetime.min
        if end_time is None:
            end_time = now

        return self.store.get_metrics_summary(start_time, end_time)

    def get_session_metrics(self, thread_id: str) -> SessionMetrics:
        """Get metrics for a specific session/thread."""
        return self.store.get_session_metrics(thread_id)

    def get_tool_metrics(
        self,
        tool_name: str | None = None,
        period: str = "all",
    ) -> dict:
        """
        Get tool performance metrics.

        Args:
            tool_name: Specific tool, or None for all tools
            period: Time period

        Returns:
            Dict with tool statistics
        """
        return self.store.get_tool_metrics(tool_name, period)

    def get_cost_breakdown(
        self,
        group_by: str = "model",  # "model", "day", "session"
        period: str = "all",
    ) -> dict:
        """Get cost breakdown by various dimensions."""
        return self.store.get_cost_breakdown(group_by, period)

    def get_slow_operations(
        self,
        threshold_ms: int = 5000,
        limit: int = 10,
    ) -> list[dict]:
        """Find slow AI calls and tool executions."""
        return self.store.get_slow_operations(threshold_ms, limit)
```

### Phase 2: Database Schema

#### 2.1 Extend Memory Store

**File**: `memory/store.py` (add new methods and tables)

```python
# Add to table creation in __init__
def _create_metrics_tables(self):
    """Create tables for metrics storage."""
    self.conn.executescript("""
        -- AI call metrics
        CREATE TABLE IF NOT EXISTS ai_calls (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            model TEXT NOT NULL,
            thread_id TEXT,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            total_tokens INTEGER NOT NULL,
            cost_usd REAL NOT NULL,
            duration_ms INTEGER NOT NULL,
            tokens_per_second REAL,
            stop_reason TEXT,
            tool_calls INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        -- Tool execution metrics (complements existing tool_records)
        CREATE TABLE IF NOT EXISTS tool_metrics (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            thread_id TEXT,
            duration_ms INTEGER NOT NULL,
            success INTEGER NOT NULL,
            error_message TEXT,
            input_size_bytes INTEGER,
            output_size_bytes INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        -- Session summaries (cached aggregations)
        CREATE TABLE IF NOT EXISTS session_metrics (
            thread_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            last_activity TEXT NOT NULL,
            ai_call_count INTEGER DEFAULT 0,
            total_input_tokens INTEGER DEFAULT 0,
            total_output_tokens INTEGER DEFAULT 0,
            total_ai_cost_usd REAL DEFAULT 0,
            tool_call_count INTEGER DEFAULT 0,
            tool_success_count INTEGER DEFAULT 0,
            tool_error_count INTEGER DEFAULT 0,
            tools_used TEXT,  -- JSON dict
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        -- Indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_ai_calls_timestamp ON ai_calls(timestamp);
        CREATE INDEX IF NOT EXISTS idx_ai_calls_thread ON ai_calls(thread_id);
        CREATE INDEX IF NOT EXISTS idx_ai_calls_model ON ai_calls(model);
        CREATE INDEX IF NOT EXISTS idx_tool_metrics_timestamp ON tool_metrics(timestamp);
        CREATE INDEX IF NOT EXISTS idx_tool_metrics_tool ON tool_metrics(tool_name);
    """)

# Add recording methods
def record_ai_call(self, metric: AICallMetric):
    """Record an AI call metric."""
    self.conn.execute("""
        INSERT INTO ai_calls
        (id, timestamp, model, thread_id, input_tokens, output_tokens,
         total_tokens, cost_usd, duration_ms, tokens_per_second, stop_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        metric.id, metric.timestamp.isoformat(), metric.model,
        metric.thread_id, metric.input_tokens, metric.output_tokens,
        metric.total_tokens, metric.cost_usd, metric.duration_ms,
        metric.tokens_per_second, metric.stop_reason
    ))
    self.conn.commit()

    # Update session metrics
    self._update_session_ai_metrics(metric)

def record_tool_metric(self, metric: ToolMetric):
    """Record a tool execution metric."""
    self.conn.execute("""
        INSERT INTO tool_metrics
        (id, timestamp, tool_name, thread_id, duration_ms, success,
         error_message, input_size_bytes, output_size_bytes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        metric.id, metric.timestamp.isoformat(), metric.tool_name,
        metric.thread_id, metric.duration_ms, int(metric.success),
        metric.error_message, metric.input_size_bytes, metric.output_size_bytes
    ))
    self.conn.commit()

def get_metrics_summary(self, start_time: datetime, end_time: datetime) -> MetricsSummary:
    """Get aggregated metrics for a time period."""
    # Implementation with SQL aggregation queries
    ...

def get_session_metrics(self, thread_id: str) -> SessionMetrics:
    """Get metrics for a specific session."""
    ...

def get_cost_breakdown(self, group_by: str, period: str) -> dict:
    """Get cost breakdown by dimension."""
    ...
```

### Phase 3: Agent Tools

#### 3.1 Metrics Tool

**File**: `tools/metrics.py` (NEW)

```python
"""
Metrics tool - allows the agent to introspect its own performance.

This tool gives the agent visibility into:
- API costs and token usage
- Tool execution performance
- Session statistics
- Bottleneck identification
"""

from tools import tool


@tool
def get_metrics(
    action: str,
    period: str = "day",
    source: str = None,
    tool_name: str = None,
    thread_id: str = None,
    limit: int = 10,
) -> dict:
    """
    Query performance metrics and costs across all system components.

    Actions:
    - "summary": Overall performance summary for the period
    - "costs": Cost breakdown by model, source, and usage
    - "tools": Tool performance statistics
    - "memory": Memory system costs (extraction, summaries, retrieval, embeddings)
    - "embeddings": Embedding API usage and cache hit rates
    - "session": Metrics for a specific session/thread
    - "slow": Find slow operations (bottlenecks)
    - "errors": Recent errors and failure patterns
    - "by_source": Breakdown by source (agent, extraction, summary, retrieval)

    Args:
        action: What metrics to retrieve
        period: Time period - "hour", "day", "week", "month", "all"
        source: Filter by source - "agent", "extraction", "summary", "retrieval", "embedding"
        tool_name: Filter to specific tool (for "tools" action)
        thread_id: Session ID (for "session" action)
        limit: Max results for list actions

    Returns:
        Dict with requested metrics

    Examples:
        get_metrics(action="summary")
        get_metrics(action="costs", period="week")
        get_metrics(action="memory")  # See memory system costs
        get_metrics(action="by_source")  # Breakdown by component
        get_metrics(action="tools", tool_name="web_search")
        get_metrics(action="slow", limit=5)
    """
    # Get collector from agent context (injected)
    from agent import _get_metrics_collector
    collector = _get_metrics_collector()

    if action == "summary":
        summary = collector.get_summary(period=period)
        return {
            "period": period,
            "ai_calls": summary.total_ai_calls,
            "total_tokens": summary.total_tokens,
            "total_cost": f"${summary.total_cost_usd:.4f}",
            "tool_calls": summary.total_tool_calls,
            "avg_ai_latency_ms": round(summary.avg_ai_latency_ms, 1),
            "avg_tool_latency_ms": round(summary.avg_tool_latency_ms, 1),
            "cost_by_model": {
                k: f"${v:.4f}" for k, v in summary.cost_by_model.items()
            },
            "top_tools": dict(list(summary.calls_by_tool.items())[:5]),
        }

    elif action == "costs":
        breakdown = collector.get_cost_breakdown(group_by="model", period=period)
        return {
            "period": period,
            "total_cost": f"${breakdown['total']:.4f}",
            "by_model": {
                k: f"${v:.4f}" for k, v in breakdown["by_model"].items()
            },
            "avg_cost_per_call": f"${breakdown['avg_per_call']:.6f}",
            "avg_tokens_per_call": breakdown["avg_tokens_per_call"],
        }

    elif action == "tools":
        stats = collector.get_tool_metrics(tool_name=tool_name, period=period)
        if tool_name:
            return {
                "tool": tool_name,
                "calls": stats["call_count"],
                "success_rate": f"{stats['success_rate']:.1f}%",
                "avg_duration_ms": round(stats["avg_duration_ms"], 1),
                "p95_duration_ms": round(stats.get("p95_duration_ms", 0), 1),
                "total_errors": stats["error_count"],
                "last_error": stats.get("last_error"),
            }
        return {
            "period": period,
            "tools": [
                {
                    "name": t["name"],
                    "calls": t["call_count"],
                    "success_rate": f"{t['success_rate']:.1f}%",
                    "avg_ms": round(t["avg_duration_ms"], 1),
                }
                for t in stats["tools"][:limit]
            ],
        }

    elif action == "session":
        if not thread_id:
            return {"error": "thread_id required for session metrics"}
        session = collector.get_session_metrics(thread_id)
        return {
            "thread_id": thread_id,
            "ai_calls": session.ai_call_count,
            "tokens_used": session.total_input_tokens + session.total_output_tokens,
            "cost": f"${session.total_ai_cost_usd:.4f}",
            "tool_calls": session.tool_call_count,
            "tool_errors": session.tool_error_count,
            "tools_used": session.tools_used,
            "duration_minutes": session.total_duration_ms / 60000,
        }

    elif action == "slow":
        slow = collector.get_slow_operations(threshold_ms=5000, limit=limit)
        return {
            "slow_operations": [
                {
                    "type": op["type"],
                    "name": op["name"],
                    "duration_ms": op["duration_ms"],
                    "timestamp": op["timestamp"],
                }
                for op in slow
            ],
        }

    elif action == "errors":
        errors = collector.get_recent_errors(limit=limit)
        return {
            "recent_errors": [
                {
                    "tool": err["tool_name"],
                    "error": err["error_message"][:200],
                    "timestamp": err["timestamp"],
                }
                for err in errors
            ],
            "error_counts_by_tool": collector.get_error_counts(),
        }

    elif action == "memory":
        # Memory system breakdown (extraction, summaries, retrieval, embeddings)
        memory_stats = collector.get_memory_system_metrics(period=period)
        return {
            "period": period,
            "extraction": {
                "calls": memory_stats["extraction"]["call_count"],
                "cost": f"${memory_stats['extraction']['cost']:.4f}",
                "avg_latency_ms": memory_stats["extraction"]["avg_latency_ms"],
            },
            "summaries": {
                "calls": memory_stats["summary"]["call_count"],
                "cost": f"${memory_stats['summary']['cost']:.4f}",
                "avg_latency_ms": memory_stats["summary"]["avg_latency_ms"],
            },
            "retrieval": {
                "calls": memory_stats["retrieval"]["call_count"],
                "cost": f"${memory_stats['retrieval']['cost']:.4f}",
                "avg_latency_ms": memory_stats["retrieval"]["avg_latency_ms"],
            },
            "embeddings": {
                "calls": memory_stats["embedding"]["call_count"],
                "cost": f"${memory_stats['embedding']['cost']:.4f}",
                "cache_hit_rate": f"{memory_stats['embedding']['cache_hit_rate']:.1f}%",
            },
            "total_memory_cost": f"${memory_stats['total_cost']:.4f}",
        }

    elif action == "embeddings":
        # Detailed embedding stats
        embed_stats = collector.get_embedding_metrics(period=period)
        return {
            "period": period,
            "total_calls": embed_stats["call_count"],
            "total_texts_embedded": embed_stats["text_count"],
            "cache_hit_rate": f"{embed_stats['cache_hit_rate']:.1f}%",
            "cost": f"${embed_stats['cost']:.4f}",
            "avg_latency_ms": embed_stats["avg_latency_ms"],
            "by_model": embed_stats.get("by_model", {}),
        }

    elif action == "by_source":
        # Breakdown by source (agent, extraction, summary, retrieval)
        source_stats = collector.get_metrics_by_source(period=period)
        return {
            "period": period,
            "sources": {
                name: {
                    "calls": stats["call_count"],
                    "tokens": stats["total_tokens"],
                    "cost": f"${stats['cost']:.4f}",
                    "avg_latency_ms": round(stats["avg_latency_ms"], 1),
                    "pct_of_total_cost": f"{stats['cost_pct']:.1f}%",
                }
                for name, stats in source_stats.items()
            },
            "total_cost": f"${sum(s['cost'] for s in source_stats.values()):.4f}",
        }

    return {"error": f"Unknown action: {action}"}
```

### Phase 4: Integration

#### 4.1 Wire Everything Together

**File**: `agent.py` (modifications)

```python
# Near the top, add import
from metrics.collector import MetricsCollector

# In Agent.__init__, initialize collector
self.metrics_collector = MetricsCollector(self.memory.store if self.memory else None)
self.metrics_collector.attach(self)

# In run_async, emit AI call events
async def run_async(self, user_input: str, thread_id: str = "default", context: dict = None):
    ...
    while True:
        # Emit AI call start
        self.emit("ai_call_start", {
            "model": self.model,
            "thread_id": thread_id,
            "message_count": len(thread),
        })

        start_time = time.time()
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=8096,
            system=system_prompt,
            tools=self._tool_schemas(),
            messages=thread
        )
        duration_ms = int((time.time() - start_time) * 1000)

        # Emit AI call end
        self.emit("ai_call_end", {
            "model": self.model,
            "thread_id": thread_id,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "stop_reason": response.stop_reason,
            "duration_ms": duration_ms,
        })
        ...

# Add global accessor for tools
_metrics_collector_instance = None

def _get_metrics_collector():
    return _metrics_collector_instance
```

**File**: `memory/integration.py` (add metrics setup)

```python
def setup_memory_hooks(agent, memory):
    """Set up event hooks including metrics collection."""
    # ... existing code ...

    # Set up metrics collector
    from metrics.collector import MetricsCollector
    collector = MetricsCollector(memory.store)
    collector.attach(agent)

    # Store reference for tool access
    import agent as agent_module
    agent_module._metrics_collector_instance = collector
```

---

## File Structure Summary

```
metrics/
├── __init__.py       # Package exports
├── clients.py        # InstrumentedAnthropic, InstrumentedOpenAI (KEY FILE)
├── models.py         # Data models (LLMCallMetric, EmbeddingCallMetric, etc.)
├── costs.py          # Token/embedding cost calculation for all providers
└── collector.py      # MetricsCollector class

tools/
└── metrics.py        # get_metrics tool (agent introspection)

memory/
├── store.py          # Extended with metrics tables and queries
├── extraction.py     # Modified: use InstrumentedAnthropic + track_source
├── summaries.py      # Modified: use InstrumentedAnthropic + track_source
├── retrieval.py      # Modified: use InstrumentedAnthropic + track_source
└── embeddings.py     # Modified: use InstrumentedOpenAI

agent.py              # Modified: use InstrumentedAsyncAnthropic
```

**Total new code**: ~700 lines
**Modified lines**: ~30 lines (just client swaps and track_source wraps)

---

## Usage Examples

### Agent Self-Introspection

The agent can now understand its own performance across ALL components:

```
User: How much have you cost me today?

Agent: Let me check my metrics.
[Calls get_metrics(action="summary", period="day")]

Agent: Today's usage across all systems:
- Total cost: $0.4521
- 47 agent LLM calls: $0.2341
- 12 memory extraction calls: $0.0892
- 8 summary refresh calls: $0.0634
- 3 deep retrieval searches: $0.0412
- 156 embedding calls: $0.0242
Average cost per agent call: $0.0050
```

```
User: What's my memory system costing me?

Agent: [Calls get_metrics(action="memory", period="week")]

Agent: Memory system costs this week:
- Extraction (entity/relationship): $2.34 (89 calls)
- Summary generation: $1.87 (42 refreshes)
- Deep retrieval: $0.92 (18 searches)
- Embeddings: $0.31 (1,234 texts, 94% cache hit rate)
Total memory overhead: $5.44 (23% of total spend)
```

```
User: Which tools are slowest?

Agent: [Calls get_metrics(action="slow")]

Agent: The slowest operations today:
1. browse_url: 12,450ms - navigating to complex page
2. deep_retrieval: 8,890ms - complex memory search
3. web_search: 8,234ms - rate limited
4. summary_refresh: 6,120ms - refreshing root node
```

```
User: Show me cost breakdown by component

Agent: [Calls get_metrics(action="by_source")]

Agent: Cost breakdown by source:
- Agent (main reasoning): $0.23 (51% of total)
- Extraction (NLP): $0.11 (24%)
- Summaries: $0.07 (15%)
- Retrieval: $0.03 (7%)
- Embeddings: $0.01 (3%)
```

### Programmatic Access

```python
# Get summary
summary = agent.metrics_collector.get_summary(period="week")
print(f"Weekly cost: ${summary.total_cost_usd:.2f}")
print(f"Token efficiency: {summary.avg_tokens_per_call} tokens/call")

# Get session breakdown
session = agent.metrics_collector.get_session_metrics("thread_123")
print(f"Session cost: ${session.total_ai_cost_usd:.4f}")
```

---

## Design Decisions & Trade-offs

### Why Event-Driven?

**Pros**:
- Leverages existing infrastructure
- Decoupled from core agent logic
- Easy to add more subscribers (logging, UI, etc.)
- Non-blocking by default

**Cons**:
- Slight overhead for event emission
- Need to manage in-flight tracking for latency

**Alternative Considered**: Decorator-based tracking
- Would require wrapping every API call
- Less flexible, harder to extend

### Why SQLite?

**Pros**:
- Already used by memory system
- No additional dependencies
- Good performance for this scale
- Easy aggregation queries

**Cons**:
- Not suitable for multi-process scenarios
- Limited to single-machine deployment

**Alternative Considered**: In-memory only
- Would lose data on restart
- No historical analysis possible

### Why Lazy Aggregation?

**Pros**:
- Minimal overhead during operation
- Fresh data on query
- Simple implementation

**Cons**:
- Slower queries on large datasets
- May need caching for dashboards

**Alternative Considered**: Pre-computed snapshots
- Added complexity
- Stale data issues
- Can add later if needed

---

## Future Enhancements

1. **Streaming Metrics**: WebSocket endpoint for real-time dashboards
2. **Alerts**: Configurable thresholds (cost > $X, latency > Yms)
3. **Export**: CSV/JSON export for external analysis
4. **Comparison**: Compare performance across time periods
5. **Prediction**: Estimate costs before executing complex tasks
6. **Budget Limits**: Stop execution when cost threshold reached

---

## Implementation Checklist

### Phase 0: Instrumented Client Wrappers (Foundation)
- [ ] Create `metrics/__init__.py` with package exports
- [ ] Create `metrics/clients.py` with InstrumentedAnthropic, InstrumentedAsyncAnthropic
- [ ] Add InstrumentedOpenAI for embeddings
- [ ] Implement `track_source()` context manager
- [ ] Add `set_event_emitter()` global configuration

### Phase 1: Core Infrastructure
- [ ] Create `metrics/models.py` with data models (LLMCallMetric, EmbeddingCallMetric, etc.)
- [ ] Create `metrics/costs.py` with pricing logic for all models
- [ ] Create `metrics/collector.py` with MetricsCollector
- [ ] Subscribe collector to `llm_call_end`, `embedding_call_end`, `tool_start/end`

### Phase 2: Integration Points (Minimal Changes)
- [ ] Update `agent.py` to use InstrumentedAsyncAnthropic
- [ ] Update `memory/extraction.py` to use InstrumentedAnthropic + track_source("extraction")
- [ ] Update `memory/summaries.py` to use InstrumentedAnthropic + track_source("summary")
- [ ] Update `memory/retrieval.py` to use InstrumentedAnthropic + track_source("retrieval")
- [ ] Update `memory/embeddings.py` to use InstrumentedOpenAI

### Phase 3: Database
- [ ] Add `llm_calls` table to `memory/store.py`
- [ ] Add `embedding_calls` table
- [ ] Add `session_metrics` table
- [ ] Implement `record_llm_call()` method
- [ ] Implement `record_embedding_call()` method
- [ ] Implement aggregation query methods (by_source, by_period, etc.)

### Phase 4: Agent Tools
- [ ] Create `tools/metrics.py` with `get_metrics` tool
- [ ] Implement all actions: summary, costs, memory, embeddings, by_source, tools, slow, errors
- [ ] Register tool in agent

### Phase 5: Final Integration
- [ ] Wire MetricsCollector in `memory/integration.py`
- [ ] Add global accessor for tools
- [ ] Update CLI to show session cost summary on exit
- [ ] Add verbose mode integration (show costs in real-time)

### Testing
- [ ] Unit tests for cost calculation (all models)
- [ ] Unit tests for instrumented clients
- [ ] Unit tests for collector
- [ ] Integration tests for full flow (agent + memory system)
- [ ] Performance tests (ensure overhead < 1ms per call)

---

## Conclusion

This design provides a powerful yet elegant metrics system that:

1. **Tracks EVERYTHING** - Main agent, extraction, summaries, retrieval, embeddings
2. **Minimizes code changes** - ~700 lines new, ~30 lines modified (just client swaps)
3. **Leverages existing infrastructure** - Events, SQLite, memory system
4. **Enables agent self-awareness** - First-class introspection tools
5. **Uses transparent instrumentation** - Wrapper clients capture all API calls automatically
6. **Supports future growth** - Easy to add dashboards, alerts, exports

### Key Innovation: Instrumented Client Wrappers

The most elegant aspect of this design is the **instrumented client wrapper** approach:

```python
# Instead of modifying every file that calls Claude:
self._client = anthropic.Anthropic()

# Just swap the client at initialization:
self._client = InstrumentedAnthropic()

# Everything else stays the same - metrics are captured automatically!
```

This means:
- No need to hunt down every API call
- Impossible to miss calls (all traffic goes through the wrapper)
- Easy to add new metrics later
- Existing code doesn't know it's being tracked

The implementation follows existing codebase patterns and can be completed incrementally, with each phase providing immediate value. Start with Phase 0 (instrumented clients) to get visibility into all API costs immediately.
