"""
Cost calculation for AI API calls.

Provides pricing for Claude models and embedding providers.
Update prices as needed when providers change their pricing.
"""

# ═══════════════════════════════════════════════════════════
# LLM PRICING (per 1M tokens: input, output)
# ═══════════════════════════════════════════════════════════

LLM_PRICING: dict[str, tuple[float, float]] = {
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

# Default fallback for unknown models
DEFAULT_LLM_PRICING = (3.00, 15.00)


# ═══════════════════════════════════════════════════════════
# EMBEDDING PRICING (per 1M tokens)
# ═══════════════════════════════════════════════════════════

EMBEDDING_PRICING: dict[str, float] = {
    # OpenAI
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
    # Voyage AI
    "voyage-2": 0.10,
    "voyage-large-2": 0.12,
    "voyage-code-2": 0.12,
}

DEFAULT_EMBEDDING_PRICING = 0.02


# ═══════════════════════════════════════════════════════════
# COST CALCULATION FUNCTIONS
# ═══════════════════════════════════════════════════════════


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate cost in USD for an LLM API call.

    Args:
        model: Model name/ID (e.g., "claude-sonnet-4-20250514")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in USD (float, rounded to 6 decimal places)
    """
    input_price, output_price = LLM_PRICING.get(model, DEFAULT_LLM_PRICING)

    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price

    return round(input_cost + output_cost, 6)


def calculate_embedding_cost(model: str, token_count: int) -> float:
    """
    Calculate cost in USD for an embedding API call.

    Args:
        model: Model name/ID (e.g., "text-embedding-3-small")
        token_count: Number of tokens (or estimate)

    Returns:
        Cost in USD (float, rounded to 6 decimal places)
    """
    price_per_million = EMBEDDING_PRICING.get(model, DEFAULT_EMBEDDING_PRICING)
    return round((token_count / 1_000_000) * price_per_million, 6)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text length.

    Uses a simple heuristic of ~4 characters per token.
    This is a rough estimate - actual token counts vary by model.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    return max(1, len(text) // 4)


def format_cost(cost_usd: float) -> str:
    """
    Format cost for human-readable display.

    Args:
        cost_usd: Cost in USD

    Returns:
        Formatted string (e.g., "$0.0023", "$1.50")
    """
    if cost_usd < 0.0001:
        return f"${cost_usd:.6f}"
    elif cost_usd < 0.01:
        return f"${cost_usd:.4f}"
    elif cost_usd < 1.00:
        return f"${cost_usd:.3f}"
    else:
        return f"${cost_usd:.2f}"


def get_model_info(model: str) -> dict:
    """
    Get pricing info for a model.

    Args:
        model: Model name/ID

    Returns:
        Dict with pricing details
    """
    if model in LLM_PRICING:
        input_price, output_price = LLM_PRICING[model]
        return {
            "type": "llm",
            "model": model,
            "input_price_per_million": input_price,
            "output_price_per_million": output_price,
        }
    elif model in EMBEDDING_PRICING:
        return {
            "type": "embedding",
            "model": model,
            "price_per_million": EMBEDDING_PRICING[model],
        }
    else:
        return {
            "type": "unknown",
            "model": model,
            "note": "Using default pricing",
        }
