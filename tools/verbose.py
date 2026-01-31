"""
Verbose Mode Control Tool

Allows the agent to control verbose output level at runtime.

Verbose Levels:
    - OFF (0): No verbose output, just user/agent messages
    - LIGHT (1): Key operations only (tool names, task starts)
    - DEEP (2): Everything (tool inputs/outputs, full details)

Note: Current verbose level is shown in the system prompt context.
"""

from tools import tool
from utils.console import console, parse_verbose_level


@tool
def set_verbose(level: str) -> dict:
    """Set the verbose output level.

    Controls how much detail is shown in the console output.
    Changes take effect immediately.

    Args:
        level: The verbose level to set. Options:
               - "off" or "0": No verbose output
               - "light" or "1": Key operations only
               - "deep" or "2": Full details
    """
    # Use shared parsing logic
    parsed = parse_verbose_level(level)
    console.set_verbose(parsed)

    return {
        "success": True,
        "level": console.get_verbose().name.lower()
    }
