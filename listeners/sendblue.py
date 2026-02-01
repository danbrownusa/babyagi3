"""
SendBlue Listener - Poll for SMS/iMessage messages.

IMPORTANT: This polling-based listener only sees OUTBOUND messages (messages YOU sent).
For INBOUND messages (messages others send to you), use the webhook endpoint instead:

    POST /webhooks/sendblue

Configure your SendBlue dashboard to send webhooks to:
    https://your-domain.com/webhooks/sendblue

See server.py for the webhook implementation.

This listener can still be useful for:
- Monitoring outbound message status
- Debugging API connectivity
- Legacy compatibility

Requires SENDBLUE_API_KEY and SENDBLUE_API_SECRET environment variables.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

# SendBlue API base URL
SENDBLUE_API_BASE = "https://api.sendblue.co"


def _normalize_phone(phone: str) -> str:
    """Normalize phone number for comparison.

    Strips formatting and ensures consistent format.
    """
    if not phone:
        return ""
    # Remove common formatting
    cleaned = phone.replace(" ", "").replace("-", "").replace("(", "").replace(")", "").replace(".", "")
    # Ensure + prefix for comparison
    if not cleaned.startswith("+"):
        if cleaned.startswith("1") and len(cleaned) == 11:
            cleaned = "+" + cleaned
        elif len(cleaned) == 10:
            cleaned = "+1" + cleaned
    return cleaned.lower()


async def run_sendblue_listener(agent, config: dict = None):
    """Run the SendBlue listener.

    Args:
        agent: The Agent instance
        config: Configuration dict with:
            - owner_phone: Owner's phone number (for owner detection)
            - poll_interval: Seconds between message checks (default: 30)
            - api_key: SendBlue API key (or use SENDBLUE_API_KEY env)
            - api_secret: SendBlue API secret (or use SENDBLUE_API_SECRET env)
    """
    config = config or {}
    owner_phone = config.get("owner_phone") or os.environ.get("OWNER_PHONE")
    poll_interval = config.get("poll_interval", 30)

    # Get API credentials
    api_key = config.get("api_key") or os.environ.get("SENDBLUE_API_KEY")
    api_secret = config.get("api_secret") or os.environ.get("SENDBLUE_API_SECRET")

    if not api_key or not api_secret:
        logger.warning("SendBlue listener disabled: SENDBLUE_API_KEY or SENDBLUE_API_SECRET not set")
        return

    logger.info("SendBlue listener started")
    logger.warning(
        "NOTE: Polling only sees OUTBOUND messages. "
        "For INBOUND messages, configure webhooks to POST /webhooks/sendblue"
    )

    # Track processed message IDs to avoid reprocessing
    processed_ids = set()

    # Track the last check time to only get new messages
    last_check_time = datetime.now(timezone.utc)

    # Normalize owner phone for comparison
    owner_phone_normalized = _normalize_phone(owner_phone) if owner_phone else None

    headers = {
        "sb-api-key-id": api_key,
        "sb-api-secret-key": api_secret,
        "Content-Type": "application/json"
    }

    while True:
        try:
            async with httpx.AsyncClient() as client:
                # Fetch recent messages
                # The API returns messages in reverse chronological order
                response = await client.get(
                    f"{SENDBLUE_API_BASE}/api/v2/messages",
                    headers=headers,
                    params={
                        "limit": 50  # Fetch recent messages
                    },
                    timeout=30.0
                )

                if response.status_code != 200:
                    logger.error(f"SendBlue API error: {response.status_code} - {response.text}")
                    await asyncio.sleep(poll_interval)
                    continue

                data = response.json()
                messages = data.get("messages", [])

                # Log poll results for visibility
                inbound_count = sum(1 for m in messages if not m.get("is_outbound", False))
                outbound_count = len(messages) - inbound_count
                logger.debug(
                    f"SendBlue poll: {len(messages)} messages "
                    f"({inbound_count} inbound, {outbound_count} outbound)"
                )

                # Process messages (filter for inbound only)
                for msg in messages:
                    msg_id = msg.get("message_handle") or msg.get("id")

                    # Skip if already processed
                    if msg_id in processed_ids:
                        continue

                    # Skip outbound messages (we only want inbound)
                    if msg.get("is_outbound", False):
                        processed_ids.add(msg_id)
                        continue

                    # Get message details
                    from_number = msg.get("from_number", "")
                    content = msg.get("content", "")
                    media_url = msg.get("media_url")
                    date_sent = msg.get("date_sent") or msg.get("date_created")

                    # Skip empty messages
                    if not content and not media_url:
                        processed_ids.add(msg_id)
                        continue

                    # Check if this is a new message (after listener started)
                    # This prevents processing old messages on startup
                    if date_sent:
                        try:
                            msg_time = datetime.fromisoformat(date_sent.replace("Z", "+00:00"))
                            # Skip messages older than when listener started (first run only)
                            if len(processed_ids) == 0 and msg_time < last_check_time:
                                processed_ids.add(msg_id)
                                continue
                        except (ValueError, TypeError):
                            pass

                    # Normalize sender for comparison
                    from_number_normalized = _normalize_phone(from_number)

                    # Determine if owner
                    is_owner = bool(
                        owner_phone_normalized and
                        from_number_normalized == owner_phone_normalized
                    )

                    # Build thread ID
                    if is_owner:
                        # Owner messages use a single thread for continuity
                        thread_id = "sendblue:owner"
                    else:
                        # External messages get separate thread per sender
                        thread_id = f"sendblue:{from_number_normalized}"

                    # Build context
                    context = {
                        "channel": "sendblue",
                        "is_owner": is_owner,
                        "sender": from_number,
                        "message_id": msg_id,
                    }

                    # Format input with message context
                    sender_type = "Owner" if is_owner else "External"
                    message_input = f"[Text from {sender_type}: {from_number}]\n\n{content}"

                    if media_url:
                        message_input += f"\n\n[Media attached: {media_url}]"

                    logger.info(f"Processing text from {from_number} (owner={is_owner}): {content[:50]}...")

                    # Process through agent
                    response_text = await agent.run_async(
                        message_input,
                        thread_id=thread_id,
                        context=context
                    )

                    # Auto-reply for owner messages
                    if is_owner and response_text:
                        try:
                            # Use the sendblue sender if registered
                            if "sendblue" in agent.senders:
                                await agent.senders["sendblue"].send(
                                    to=from_number,
                                    content=response_text
                                )
                                logger.info(f"Replied to {from_number}")
                            else:
                                logger.warning("SendBlue sender not registered, cannot auto-reply")
                        except Exception as e:
                            logger.error(f"Failed to reply: {e}")

                    # Mark as processed
                    processed_ids.add(msg_id)

        except httpx.TimeoutException:
            logger.warning("SendBlue API timeout, will retry")
        except Exception as e:
            logger.error(f"SendBlue listener error: {e}")

        await asyncio.sleep(poll_interval)
