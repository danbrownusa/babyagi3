"""
Interactive Initialization Flow for BabyAGI.

Self-contained onboarding module that walks the user through first-time setup.
Designed so that none of this context persists past initialization - the entire
module is only imported/called once during first run, then never loaded again.

Flow:
    1. Detect first run (no ~/.babyagi/initialized marker)
    2. Explain how BabyAGI works
    3. Collect owner info (name, email, phone, timezone)
    4. Walk through AgentMail setup (email channel)
    5. Walk through SendBlue setup (SMS/iMessage channel)
    6. Write config values and env vars
    7. Schedule two daily tasks (stats report + self-improvement)
    8. Write marker file so init never runs again
"""

import os
import re
import sys
import json
import uuid
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Marker file location
INIT_MARKER = Path(os.path.expanduser("~/.babyagi/initialized"))
INIT_STATE_FILE = Path(os.path.expanduser("~/.babyagi/init_state.json"))

# ANSI colors for terminal output
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_RESET = "\033[0m"


# =============================================================================
# Public API
# =============================================================================

def needs_initialization() -> bool:
    """Check if BabyAGI needs first-time initialization."""
    return not INIT_MARKER.exists()


def run_initialization(config: dict) -> dict:
    """Run the interactive initialization flow.

    This is the single entry point. It handles the full onboarding
    interactively via stdin/stdout, updates config in-place, and
    returns the modified config. After this returns, the module is
    never imported again.

    Args:
        config: The loaded config dict (will be modified in-place).

    Returns:
        Updated config dict with user-provided values applied.
    """
    _print_welcome()
    _print_system_overview()

    # Step 1: Collect owner information
    owner_info = _collect_owner_info(config)
    _apply_owner_info(config, owner_info)

    # Step 2: Set up AgentMail (email channel)
    agentmail_info = _setup_agentmail(config, owner_info)

    # Step 3: Set up SendBlue (SMS/iMessage channel)
    sendblue_info = _setup_sendblue(config, owner_info)

    # Step 4: Summary and confirmation
    _print_setup_summary(owner_info, agentmail_info, sendblue_info)

    # Step 5: Save init state for post-agent task scheduling
    _save_init_state(owner_info, agentmail_info, sendblue_info)

    # Step 6: Mark initialization as complete
    _write_marker(owner_info)

    _print_completion()

    return config


def schedule_post_init_tasks(agent):
    """Schedule the two daily tasks after agent is fully initialized.

    Called from main.py after the agent is created. Reads saved init state
    and sets up:
      1. Daily stats analysis email (5 min after init, then every 24h)
      2. Daily self-improvement task (1h after init, then every 24h)

    After scheduling, deletes the init state file so this is truly one-time.
    """
    if not INIT_STATE_FILE.exists():
        return

    try:
        with open(INIT_STATE_FILE) as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    owner_email = state.get("owner_email", "")
    if not owner_email:
        # No email to send stats to - skip stats task but still do self-improvement
        logger.info("No owner email found, skipping daily stats task")

    from scheduler import Schedule, ScheduledTask

    now = datetime.now(timezone.utc)

    # --- Task 1: Daily Stats Analysis ---
    if owner_email:
        stats_anchor = (now + timedelta(minutes=5)).isoformat()
        stats_task = ScheduledTask(
            id=uuid.uuid4().hex[:8],
            name="Daily Stats Report",
            goal=_DAILY_STATS_GOAL.format(owner_email=owner_email),
            schedule=Schedule(
                kind="every",
                every="24h",
                anchor=stats_anchor,
            ),
        )
        agent.scheduler.add(stats_task)
        logger.info("Scheduled daily stats report (first run in ~5 minutes)")

    # --- Task 2: Daily Self-Improvement ---
    improvement_anchor = (now + timedelta(hours=1)).isoformat()
    improvement_task = ScheduledTask(
        id=uuid.uuid4().hex[:8],
        name="Daily Self-Improvement",
        goal=_DAILY_IMPROVEMENT_GOAL,
        schedule=Schedule(
            kind="every",
            every="24h",
            anchor=improvement_anchor,
        ),
    )
    agent.scheduler.add(improvement_task)
    logger.info("Scheduled daily self-improvement (first run in ~1 hour)")

    # Clean up init state file - no longer needed
    try:
        INIT_STATE_FILE.unlink()
    except OSError:
        pass


# =============================================================================
# Goal Templates (kept here so the context lives in this file only)
# =============================================================================

_DAILY_STATS_GOAL = """Compile and email a detailed daily statistics report to {owner_email}.

Gather the following data:
1. TOOL USAGE: Use get_cost_summary to get overall metrics. List every tool used in the last 24 hours with call counts, success rates, and average latency.
2. MEMORY & EXTRACTION: Check memory system stats - how many entities extracted, relationships found, summaries generated, and events recorded in the last 24 hours.
3. LLM COSTS: Break down costs by model and by source (agent, extraction, summary, retrieval). Show total tokens and total spend.
4. SCHEDULED TASKS: List all scheduled tasks with their last run status and next run time.
5. ERRORS: List any errors from the last 24 hours with tool name and error message.

Format this as a clean, readable email with sections and send it using send_email to {owner_email} with subject "BabyAGI Daily Stats Report - [today's date]".

Keep the email concise but thorough. Use plain text formatting with clear section headers."""

_DAILY_IMPROVEMENT_GOAL = """Think of ONE concrete thing to do right now to be more helpful to your owner.

IMPORTANT: First, check your memory for notes tagged "self_improvement_log" to see what you've done in previous runs. Do NOT repeat recent actions - vary your approach.

Choose ONE of these action types (rotate between them):

A) CREATE A USEFUL SKILL: Think about what your owner frequently asks for or might need. Create and save a new tool/skill that would help them. Test it to make sure it works.

B) SET UP A HELPFUL SCHEDULED TASK: Think of something useful to monitor or do regularly that isn't already scheduled. Examples: check for interesting news in owner's field, review and organize recent memories, check if any API keys are expiring, test that all channels are working.

C) ASK YOUR OWNER A QUESTION: Come up with one thoughtful question to better understand your owner's needs, preferences, or workflows. Send it via the most appropriate channel (email or SMS). Make it specific and actionable, not generic.

After completing your chosen action, store a note in memory with tag "self_improvement_log" recording:
- Date
- Action type (A/B/C)
- What you did
- Brief rationale

This helps you track what you've done and vary your approach. Keep the log entry under 100 words to prevent context bloat."""


# =============================================================================
# Internal: Welcome and system explanation
# =============================================================================

def _print_welcome():
    """Print the welcome banner."""
    print(f"""
{_BOLD}{_CYAN}{'=' * 60}
   Welcome to BabyAGI - First-Time Setup
{'=' * 60}{_RESET}

{_DIM}This wizard will walk you through setting up your personal
AI agent. It takes about 2 minutes.{_RESET}
""")


def _print_system_overview():
    """Explain how BabyAGI works before collecting info."""
    print(f"""{_BOLD}How BabyAGI Works{_RESET}
{_DIM}{'─' * 40}{_RESET}

BabyAGI is a persistent AI agent that runs in the background
and communicates with you through multiple channels:

  {_CYAN}Chat{_RESET}      Talk to it right here in the terminal
  {_CYAN}Email{_RESET}     Gets its own email address via AgentMail
  {_CYAN}SMS{_RESET}       Text it via SendBlue (iMessage/SMS)

It remembers everything - conversations, people, facts - in a
three-layer memory system (event log, knowledge graph, summaries).

It can schedule tasks, do research in the background, search the
web, run code, create new tools on the fly, and learn from your
feedback over time.

{_BOLD}Let's set it up.{_RESET}
""")


# =============================================================================
# Internal: Owner info collection
# =============================================================================

def _collect_owner_info(config: dict) -> dict:
    """Interactively collect owner information."""
    print(f"{_BOLD}Step 1: About You{_RESET}")
    print(f"{_DIM}{'─' * 40}{_RESET}")
    print(f"{_DIM}This helps your agent personalize responses and know how to reach you.{_RESET}\n")

    existing_owner = config.get("owner", {})

    name = _prompt(
        "Your name",
        default=existing_owner.get("name") or os.environ.get("OWNER_NAME", ""),
    )

    email = _prompt(
        "Your email address",
        default=existing_owner.get("email") or os.environ.get("OWNER_EMAIL", ""),
        validator=_validate_email,
    )

    phone = _prompt(
        "Your phone number (for SMS, e.g. +15551234567)",
        default=existing_owner.get("phone") or os.environ.get("OWNER_PHONE", ""),
        required=False,
        validator=_validate_phone,
    )

    timezone = _prompt(
        "Your timezone (e.g. America/New_York, US/Pacific, UTC)",
        default=existing_owner.get("timezone") or os.environ.get("OWNER_TIMEZONE", ""),
        required=False,
    )

    agent_name = _prompt(
        "Name for your agent",
        default=config.get("agent", {}).get("name") or os.environ.get("AGENT_NAME", "Assistant"),
    )

    print()
    return {
        "name": name,
        "email": email,
        "phone": phone,
        "timezone": timezone,
        "agent_name": agent_name,
    }


def _apply_owner_info(config: dict, info: dict):
    """Apply owner info to config and environment variables."""
    # Update config
    if "owner" not in config:
        config["owner"] = {}
    config["owner"]["name"] = info["name"]
    config["owner"]["email"] = info["email"]
    config["owner"]["phone"] = info.get("phone", "")
    config["owner"]["timezone"] = info.get("timezone", "")
    config["owner"]["contacts"] = {
        "email": info["email"],
        "phone": info.get("phone", ""),
    }

    if "agent" not in config:
        config["agent"] = {}
    config["agent"]["name"] = info["agent_name"]

    # Set environment variables for immediate use
    os.environ["OWNER_NAME"] = info["name"]
    os.environ["OWNER_EMAIL"] = info["email"]
    if info.get("phone"):
        os.environ["OWNER_PHONE"] = info["phone"]
    if info.get("timezone"):
        os.environ["OWNER_TIMEZONE"] = info["timezone"]
    os.environ["AGENT_NAME"] = info["agent_name"]


# =============================================================================
# Internal: AgentMail setup
# =============================================================================

def _setup_agentmail(config: dict, owner_info: dict) -> dict:
    """Walk through AgentMail email channel setup."""
    print(f"{_BOLD}Step 2: Email Channel (AgentMail){_RESET}")
    print(f"{_DIM}{'─' * 40}{_RESET}")
    print(f"""
AgentMail gives your agent its own email address. It can then:
  - Receive and respond to emails on your behalf
  - Send you summaries and reports
  - Handle email-based workflows (signups, verifications, etc.)

{_DIM}Get a free API key at: {_CYAN}https://agentmail.to{_RESET}
""")

    # Check if already configured
    existing_key = os.environ.get("AGENTMAIL_API_KEY", "")
    if existing_key:
        print(f"  {_GREEN}AgentMail API key already detected.{_RESET}")
        setup = _prompt_yn("Reconfigure AgentMail?", default=False)
        if not setup:
            return {"configured": True, "source": "existing"}

    api_key = _prompt(
        "AgentMail API key",
        default=existing_key,
        required=False,
        secret=True,
    )

    if not api_key:
        print(f"  {_YELLOW}Skipping email setup. You can add it later in config.yaml or as AGENTMAIL_API_KEY.{_RESET}\n")
        return {"configured": False}

    # Set immediately
    os.environ["AGENTMAIL_API_KEY"] = api_key

    # Try to get or create an inbox
    inbox_id = None
    try:
        from agentmail import AgentMail
        client = AgentMail(api_key=api_key)

        # Check for existing inboxes
        inboxes_response = client.inboxes.list()
        inboxes = getattr(inboxes_response, 'inboxes', None) or inboxes_response or []

        if inboxes:
            inbox = inboxes[0]
            inbox_id = getattr(inbox, 'inbox_id', None) or getattr(inbox, 'id', None)
            print(f"  {_GREEN}Found existing inbox: {inbox_id}{_RESET}")
        else:
            # Create new inbox
            inbox = client.inboxes.create()
            inbox_id = getattr(inbox, 'inbox_id', None) or getattr(inbox, 'id', None)
            print(f"  {_GREEN}Created new inbox: {inbox_id}{_RESET}")

        if inbox_id:
            os.environ["AGENTMAIL_INBOX_ID"] = inbox_id
    except ImportError:
        print(f"  {_YELLOW}agentmail package not installed. Run: pip install agentmail{_RESET}")
        print(f"  {_DIM}The API key has been saved and will work once the package is installed.{_RESET}")
    except Exception as e:
        print(f"  {_YELLOW}Could not connect to AgentMail: {e}{_RESET}")
        print(f"  {_DIM}The API key has been saved. The agent will retry on startup.{_RESET}")

    # Update config
    if "channels" not in config:
        config["channels"] = {}
    if "email" not in config["channels"]:
        config["channels"]["email"] = {}
    config["channels"]["email"]["enabled"] = True
    config["channels"]["email"]["api_key"] = api_key

    print()
    return {
        "configured": True,
        "api_key_set": True,
        "inbox_id": inbox_id,
    }


# =============================================================================
# Internal: SendBlue setup
# =============================================================================

def _setup_sendblue(config: dict, owner_info: dict) -> dict:
    """Walk through SendBlue SMS/iMessage channel setup."""
    print(f"{_BOLD}Step 3: SMS/iMessage Channel (SendBlue){_RESET}")
    print(f"{_DIM}{'─' * 40}{_RESET}")
    print(f"""
SendBlue lets your agent send and receive text messages (SMS and
iMessage). You can text your agent from your phone and get replies.

{_DIM}Get API credentials at: {_CYAN}https://sendblue.co{_RESET}
""")

    # Check if already configured
    existing_key = os.environ.get("SENDBLUE_API_KEY", "")
    existing_secret = os.environ.get("SENDBLUE_API_SECRET", "")
    if existing_key and existing_secret:
        print(f"  {_GREEN}SendBlue credentials already detected.{_RESET}")
        setup = _prompt_yn("Reconfigure SendBlue?", default=False)
        if not setup:
            return {"configured": True, "source": "existing"}

    api_key = _prompt(
        "SendBlue API Key",
        default=existing_key,
        required=False,
        secret=True,
    )

    if not api_key:
        print(f"  {_YELLOW}Skipping SMS setup. You can add it later in config.yaml or as SENDBLUE_API_KEY.{_RESET}\n")
        return {"configured": False}

    api_secret = _prompt(
        "SendBlue API Secret",
        default=existing_secret,
        required=False,
        secret=True,
    )

    if not api_secret:
        print(f"  {_YELLOW}Both API Key and Secret are required for SendBlue. Skipping.{_RESET}\n")
        return {"configured": False}

    # Set immediately
    os.environ["SENDBLUE_API_KEY"] = api_key
    os.environ["SENDBLUE_API_SECRET"] = api_secret

    # Optionally collect the SendBlue phone number
    from_number = _prompt(
        "Your SendBlue phone number (the number messages come from)",
        default=os.environ.get("SENDBLUE_PHONE_NUMBER", ""),
        required=False,
    )
    if from_number:
        os.environ["SENDBLUE_PHONE_NUMBER"] = from_number

    # Update config
    if "channels" not in config:
        config["channels"] = {}
    if "sendblue" not in config["channels"]:
        config["channels"]["sendblue"] = {}
    config["channels"]["sendblue"]["enabled"] = True
    config["channels"]["sendblue"]["api_key"] = api_key
    config["channels"]["sendblue"]["api_secret"] = api_secret

    print(f"  {_GREEN}SendBlue configured successfully.{_RESET}")

    # Suggest setting owner phone if not already set
    if not owner_info.get("phone"):
        print(f"\n  {_DIM}Tip: Set your phone number (Step 1) so the agent knows which texts are from you.{_RESET}")

    print()
    return {
        "configured": True,
        "api_key_set": True,
        "api_secret_set": True,
    }


# =============================================================================
# Internal: Summary and completion
# =============================================================================

def _print_setup_summary(owner_info: dict, agentmail_info: dict, sendblue_info: dict):
    """Print a summary of what was configured."""
    print(f"""{_BOLD}Setup Summary{_RESET}
{_DIM}{'─' * 40}{_RESET}
""")
    print(f"  {_BOLD}Owner:{_RESET}      {owner_info['name']} ({owner_info['email']})")
    print(f"  {_BOLD}Agent:{_RESET}      {owner_info['agent_name']}")

    if owner_info.get("timezone"):
        print(f"  {_BOLD}Timezone:{_RESET}   {owner_info['timezone']}")

    # Channels
    channels = []
    channels.append("CLI (always on)")
    if agentmail_info.get("configured"):
        inbox = agentmail_info.get("inbox_id", "pending")
        channels.append(f"Email ({inbox})")
    if sendblue_info.get("configured"):
        channels.append("SMS/iMessage (SendBlue)")

    print(f"  {_BOLD}Channels:{_RESET}   {', '.join(channels)}")
    print()


def _print_completion():
    """Print the completion message."""
    print(f"""{_BOLD}{_GREEN}{'=' * 60}
   Setup Complete!
{'=' * 60}{_RESET}

Your agent is now configured and ready to go. Two daily tasks
have been scheduled:

  {_CYAN}1. Daily Stats Report{_RESET}
     Emails you detailed usage statistics every 24 hours.
     First report arrives in ~5 minutes.

  {_CYAN}2. Daily Self-Improvement{_RESET}
     Your agent will think of one thing to do each day to
     be more helpful - creating skills, setting up tasks,
     or asking you questions to understand you better.

You can reconfigure anytime by deleting ~/.babyagi/initialized
and restarting, or by editing config.yaml directly.

{_BOLD}Starting your agent...{_RESET}
""")


# =============================================================================
# Internal: Persistence
# =============================================================================

def _save_init_state(owner_info: dict, agentmail_info: dict, sendblue_info: dict):
    """Save minimal state needed for post-agent task scheduling.

    This file is read once by schedule_post_init_tasks() then deleted.
    """
    state = {
        "owner_email": owner_info.get("email", ""),
        "owner_name": owner_info.get("name", ""),
        "initialized_at": datetime.now(timezone.utc).isoformat(),
        "agentmail_configured": agentmail_info.get("configured", False),
        "sendblue_configured": sendblue_info.get("configured", False),
    }

    INIT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INIT_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _write_marker(owner_info: dict):
    """Write the initialization marker file."""
    INIT_MARKER.parent.mkdir(parents=True, exist_ok=True)
    marker_data = {
        "initialized_at": datetime.now(timezone.utc).isoformat(),
        "owner": owner_info.get("name", ""),
        "version": "0.3.0",
    }
    with open(INIT_MARKER, "w") as f:
        json.dump(marker_data, f, indent=2)


# =============================================================================
# Internal: Input helpers
# =============================================================================

def _prompt(
    label: str,
    default: str = "",
    required: bool = True,
    validator=None,
    secret: bool = False,
) -> str:
    """Prompt the user for input with optional default and validation.

    Args:
        label: The prompt label shown to the user.
        default: Default value (shown in brackets).
        required: Whether a non-empty value is required.
        validator: Optional callable(value) -> str|None. Returns error msg or None.
        secret: If True, mask the default display.
    """
    default_display = ""
    if default:
        if secret:
            default_display = f" [{default[:4]}...{default[-4:]}]" if len(default) > 8 else f" [****]"
        else:
            default_display = f" [{default}]"

    while True:
        try:
            value = input(f"  {_BLUE}{label}{default_display}: {_RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if not value and default:
            value = default

        if required and not value:
            print(f"    {_YELLOW}This field is required.{_RESET}")
            continue

        if not value:
            return ""

        if validator:
            error = validator(value)
            if error:
                print(f"    {_YELLOW}{error}{_RESET}")
                continue

        return value


def _prompt_yn(label: str, default: bool = True) -> bool:
    """Prompt for yes/no."""
    hint = "[Y/n]" if default else "[y/N]"
    try:
        value = input(f"  {_BLUE}{label} {hint}: {_RESET}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)

    if not value:
        return default
    return value in ("y", "yes")


def _validate_email(value: str) -> str | None:
    """Basic email validation."""
    if "@" not in value or "." not in value:
        return "Please enter a valid email address."
    return None


def _validate_phone(value: str) -> str | None:
    """Basic phone number validation."""
    if not value:
        return None
    # Strip common formatting
    cleaned = re.sub(r"[\s\-\(\)\.]+", "", value)
    if not re.match(r"^\+?\d{10,15}$", cleaned):
        return "Please enter a valid phone number (e.g. +15551234567)."
    return None
