"""
Web Tools

Provides web search and browser automation capabilities.

- web_search: Quick DuckDuckGo search (no API key needed)
- browse: Full browser automation via Browser Use Cloud API
- browse_session: Create reusable browser sessions
- browse_profile: Manage persistent browser profiles
- browse_control: Pause/resume/stop running tasks
- fetch_url: Simple URL fetching and parsing
"""

import os
import sys
import time
from tools import tool, tool_error

# Browser Use Cloud API configuration
BROWSER_USE_API_URL = "https://api.browser-use.com/api/v1"

# Available LLM models for browser automation
BROWSER_USE_MODELS = {
    "default": None,  # Use API default
    "gpt-4.1": "gpt-4.1",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "claude-sonnet": "claude-sonnet-4-20250514",
    "gemini-2.5-flash": "gemini-2.5-flash",
}


def _get_headers() -> dict:
    """Get API headers with authentication."""
    api_key = os.environ.get("BROWSER_USE_API_KEY")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }


def _show_live_url(url: str, task: str):
    """Display live browser URL if running interactively."""
    if sys.stderr.isatty():
        # Truncate task for display
        short_task = task[:50] + "..." if len(task) > 50 else task
        print(f"\033[36m🌐 Browser session: {url}\033[0m", file=sys.stderr, flush=True)
        print(f"\033[36m   Task: {short_task}\033[0m", file=sys.stderr, flush=True)


@tool(packages=["ddgs"])
def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web using DuckDuckGo.

    Fast, free search with no API key required. Use this for quick lookups,
    finding documentation, or discovering services.

    Args:
        query: What to search for
        max_results: Maximum number of results to return (default 5)
    """
    try:
        from ddgs import DDGS
    except ImportError:
        return tool_error("ddgs not installed", fix="pip install ddgs")

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return {
                "query": query,
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")
                    }
                    for r in results
                ]
            }
    except Exception as e:
        return {"error": str(e)}


@tool(packages=["httpx"], env=["BROWSER_USE_API_KEY"])
def browse(
    task: str,
    url: str = None,
    max_steps: int = 25,
    model: str = None,
    output_schema: dict = None,
    session_id: str = None,
    save_session: bool = False,
    agent=None
) -> dict:
    """Control a browser to complete a task autonomously via Browser Use Cloud.

    This is a powerful tool that can:
    - Navigate websites and fill forms
    - Sign up for services using the agent's email
    - Extract information from pages (including API keys from dashboards!)
    - Click buttons, follow links, handle popups
    - Return structured data matching a schema

    For API key retrieval, describe the task like:
    "Go to the API keys page, copy the API key shown"

    Requires BROWSER_USE_API_KEY environment variable.
    Get your key at: https://cloud.browser-use.com

    Args:
        task: Natural language description of what to do in the browser
        url: Optional starting URL
        max_steps: Maximum number of agent steps (default 25)
        model: LLM model to use (gpt-4.1, gpt-4.1-mini, claude-sonnet, gemini-2.5-flash)
        output_schema: JSON schema for structured output extraction
        session_id: Reuse an existing browser session (preserves cookies/state)
        save_session: Keep session alive after task for reuse (returns session_id)
    """
    import httpx

    api_key = os.environ.get("BROWSER_USE_API_KEY")
    if not api_key:
        return tool_error(
            "BROWSER_USE_API_KEY not set",
            fix="Get your API key at https://cloud.browser-use.com and set BROWSER_USE_API_KEY"
        )

    headers = _get_headers()

    # Build request payload
    payload = {
        "task": task,
        "maxSteps": max_steps,
    }
    if url:
        payload["startUrl"] = url

    # Add model selection
    if model:
        model_id = BROWSER_USE_MODELS.get(model, model)
        if model_id:
            payload["model"] = model_id

    # Add structured output schema
    if output_schema:
        payload["outputSchema"] = output_schema

    # Add session reuse
    if session_id:
        payload["sessionId"] = session_id

    # Configure session persistence
    if save_session:
        payload["keepSessionAlive"] = True

    try:
        # Create task
        with httpx.Client(timeout=30) as client:
            response = client.post(
                f"{BROWSER_USE_API_URL}/run-task",
                headers=headers,
                json=payload
            )

            if response.status_code not in (200, 201, 202):
                return {
                    "error": f"Failed to create task: {response.status_code}",
                    "details": response.text
                }

            task_data = response.json()
            task_id = task_data.get("id")
            live_url = task_data.get("liveUrl")
            returned_session_id = task_data.get("sessionId") or session_id

            if not task_id:
                return {"error": "No task ID returned", "response": task_data}

            # Show live URL for interactive debugging
            if live_url:
                _show_live_url(live_url, task)

            # Poll for completion (max 5 minutes)
            max_wait = 300
            poll_interval = 3
            waited = 0

            while waited < max_wait:
                time.sleep(poll_interval)
                waited += poll_interval

                status_response = client.get(
                    f"{BROWSER_USE_API_URL}/task/{task_id}",
                    headers=headers
                )

                if status_response.status_code != 200:
                    continue

                status_data = status_response.json()
                status = status_data.get("status", "").lower()

                if status in ("finished", "completed", "done"):
                    # Stop the session to save credits (unless saving for reuse)
                    if not save_session and returned_session_id:
                        try:
                            client.post(
                                f"{BROWSER_USE_API_URL}/session/{returned_session_id}/stop",
                                headers=headers
                            )
                        except Exception:
                            pass

                    result = {
                        "task": task,
                        "status": "completed",
                        "output": status_data.get("output"),
                        "steps": status_data.get("steps", []),
                        "task_id": task_id,
                        "success": True
                    }

                    # Include session ID for reuse if saved
                    if save_session and returned_session_id:
                        result["session_id"] = returned_session_id

                    return result

                elif status in ("failed", "error"):
                    return {
                        "task": task,
                        "status": "failed",
                        "error": status_data.get("error") or status_data.get("output"),
                        "task_id": task_id,
                        "success": False
                    }

            # Timeout
            return {
                "task": task,
                "status": "timeout",
                "error": f"Task did not complete within {max_wait} seconds",
                "task_id": task_id,
                "session_id": returned_session_id,
                "success": False
            }

    except Exception as e:
        return {"error": str(e), "task": task}


@tool(packages=["httpx", "bs4"])
def fetch_url(url: str, extract: str = None) -> dict:
    """Fetch a URL and optionally extract specific information.

    Simpler than browse() - just fetches and parses a page.
    Use browse() for interactive tasks, this for quick reads.

    Args:
        url: The URL to fetch
        extract: Optional description of what to extract (e.g., "the main article text")
    """
    try:
        import httpx
        from bs4 import BeautifulSoup
    except ImportError:
        return tool_error("httpx and beautifulsoup4 not installed", fix="pip install httpx beautifulsoup4")

    try:
        response = httpx.get(url, follow_redirects=True, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Get text content
        text = soup.get_text(separator="\n", strip=True)

        # Truncate if too long
        if len(text) > 10000:
            text = text[:10000] + "\n...[truncated]"

        return {
            "url": url,
            "title": soup.title.string if soup.title else None,
            "content": text,
            "extract_hint": extract
        }

    except Exception as e:
        return {"error": str(e), "url": url}


# =============================================================================
# Browser Session & Control Tools
# =============================================================================

@tool(packages=["httpx"], env=["BROWSER_USE_API_KEY"])
def browse_control(task_id: str, action: str) -> dict:
    """Control a running browser task - pause, resume, or stop.

    Use this to intervene in long-running tasks. Useful when:
    - You need to manually handle a CAPTCHA (pause, solve, resume)
    - A task is stuck and needs to be stopped
    - You want to take over control temporarily

    Args:
        task_id: The task ID returned from browse()
        action: Control action - "pause", "resume", or "stop"
    """
    import httpx

    api_key = os.environ.get("BROWSER_USE_API_KEY")
    if not api_key:
        return tool_error(
            "BROWSER_USE_API_KEY not set",
            fix="Get your API key at https://cloud.browser-use.com and set BROWSER_USE_API_KEY"
        )

    if action not in ("pause", "resume", "stop"):
        return tool_error(
            f"Invalid action: {action}",
            fix="Use one of: pause, resume, stop"
        )

    headers = _get_headers()

    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                f"{BROWSER_USE_API_URL}/task/{task_id}/{action}",
                headers=headers
            )

            if response.status_code not in (200, 201, 202):
                return {
                    "error": f"Failed to {action} task: {response.status_code}",
                    "details": response.text
                }

            return {
                "task_id": task_id,
                "action": action,
                "status": "success",
                "message": f"Task {action}d successfully"
            }

    except Exception as e:
        return {"error": str(e), "task_id": task_id}


@tool(packages=["httpx"], env=["BROWSER_USE_API_KEY"])
def browse_session(action: str, session_id: str = None, profile_id: str = None) -> dict:
    """Manage browser sessions for multi-task workflows.

    Sessions preserve browser state (cookies, localStorage, auth) between tasks.
    Create a session, run multiple browse() tasks with that session_id, then stop.

    Workflow example:
    1. create session -> get session_id
    2. browse(task="login", session_id=session_id, save_session=True)
    3. browse(task="do something", session_id=session_id, save_session=True)
    4. browse_session(action="stop", session_id=session_id)

    Args:
        action: Session action - "create" or "stop"
        session_id: Required for "stop" action
        profile_id: Optional profile ID to use with "create" (for persistent auth)
    """
    import httpx

    api_key = os.environ.get("BROWSER_USE_API_KEY")
    if not api_key:
        return tool_error(
            "BROWSER_USE_API_KEY not set",
            fix="Get your API key at https://cloud.browser-use.com and set BROWSER_USE_API_KEY"
        )

    if action not in ("create", "stop"):
        return tool_error(
            f"Invalid action: {action}",
            fix="Use one of: create, stop"
        )

    headers = _get_headers()

    try:
        with httpx.Client(timeout=30) as client:
            if action == "create":
                payload = {}
                if profile_id:
                    payload["profileId"] = profile_id

                response = client.post(
                    f"{BROWSER_USE_API_URL}/sessions",
                    headers=headers,
                    json=payload if payload else None
                )

                if response.status_code not in (200, 201, 202):
                    return {
                        "error": f"Failed to create session: {response.status_code}",
                        "details": response.text
                    }

                data = response.json()
                live_url = data.get("liveUrl")

                # Show live URL for interactive debugging
                if live_url:
                    _show_live_url(live_url, "New browser session")

                return {
                    "action": "create",
                    "session_id": data.get("id"),
                    "live_url": live_url,
                    "status": "created",
                    "message": "Session created. Use session_id in browse() calls."
                }

            elif action == "stop":
                if not session_id:
                    return tool_error(
                        "session_id required for stop action",
                        fix="Provide the session_id to stop"
                    )

                response = client.post(
                    f"{BROWSER_USE_API_URL}/session/{session_id}/stop",
                    headers=headers
                )

                if response.status_code not in (200, 201, 202, 204):
                    return {
                        "error": f"Failed to stop session: {response.status_code}",
                        "details": response.text
                    }

                return {
                    "action": "stop",
                    "session_id": session_id,
                    "status": "stopped",
                    "message": "Session stopped. Browser resources released."
                }

    except Exception as e:
        return {"error": str(e)}


@tool(packages=["httpx"], env=["BROWSER_USE_API_KEY"])
def browse_profile(action: str, name: str = None, profile_id: str = None) -> dict:
    """Manage browser profiles for persistent authentication.

    Profiles save browser fingerprints, cookies, and auth state permanently.
    Unlike sessions (temporary), profiles persist across restarts.

    Use profiles when:
    - You need to stay logged into services long-term
    - You want consistent browser identity (for bot detection)
    - You're automating workflows that require authentication

    Workflow:
    1. Create profile: browse_profile(action="create", name="my-service")
    2. Login once: browse(task="login to service", save_session=True)
    3. Future tasks: browse_session(action="create", profile_id=profile_id)
       then browse() with that session_id - already logged in!

    Args:
        action: Profile action - "create", "list", or "delete"
        name: Profile name (required for create)
        profile_id: Profile ID (required for delete)
    """
    import httpx

    api_key = os.environ.get("BROWSER_USE_API_KEY")
    if not api_key:
        return tool_error(
            "BROWSER_USE_API_KEY not set",
            fix="Get your API key at https://cloud.browser-use.com and set BROWSER_USE_API_KEY"
        )

    if action not in ("create", "list", "delete"):
        return tool_error(
            f"Invalid action: {action}",
            fix="Use one of: create, list, delete"
        )

    headers = _get_headers()

    try:
        with httpx.Client(timeout=30) as client:
            if action == "create":
                if not name:
                    return tool_error(
                        "name required for create action",
                        fix="Provide a name for the profile"
                    )

                response = client.post(
                    f"{BROWSER_USE_API_URL}/profiles",
                    headers=headers,
                    json={"name": name}
                )

                if response.status_code not in (200, 201, 202):
                    return {
                        "error": f"Failed to create profile: {response.status_code}",
                        "details": response.text
                    }

                data = response.json()
                return {
                    "action": "create",
                    "profile_id": data.get("id"),
                    "name": name,
                    "status": "created",
                    "message": "Profile created. Use profile_id with browse_session(action='create')."
                }

            elif action == "list":
                response = client.get(
                    f"{BROWSER_USE_API_URL}/profiles",
                    headers=headers
                )

                if response.status_code != 200:
                    return {
                        "error": f"Failed to list profiles: {response.status_code}",
                        "details": response.text
                    }

                data = response.json()
                profiles = data.get("profiles", data) if isinstance(data, dict) else data
                return {
                    "action": "list",
                    "profiles": [
                        {"id": p.get("id"), "name": p.get("name")}
                        for p in (profiles if isinstance(profiles, list) else [])
                    ],
                    "count": len(profiles) if isinstance(profiles, list) else 0
                }

            elif action == "delete":
                if not profile_id:
                    return tool_error(
                        "profile_id required for delete action",
                        fix="Provide the profile_id to delete"
                    )

                response = client.delete(
                    f"{BROWSER_USE_API_URL}/profiles/{profile_id}",
                    headers=headers
                )

                if response.status_code not in (200, 201, 202, 204):
                    return {
                        "error": f"Failed to delete profile: {response.status_code}",
                        "details": response.text
                    }

                return {
                    "action": "delete",
                    "profile_id": profile_id,
                    "status": "deleted",
                    "message": "Profile deleted."
                }

    except Exception as e:
        return {"error": str(e)}
