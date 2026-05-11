"""Slack webhook integration.

Provides webhook handling for Slack bot integration,
supporting Block Kit and interactive components.
"""

import hashlib
import hmac
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SlackBlockType(str, Enum):
    """Types of Slack blocks."""

    SECTION = "section"
    DIVIDER = "divider"
    CONTEXT = "context"
    ACTIONS = "actions"
    HEADER = "header"


@dataclass
class SlackMessage:
    """Slack message with blocks.

    Attributes:
        text: Fallback text for notifications.
        blocks: Block Kit blocks.
        thread_ts: Thread timestamp for replies.
        channel: Channel ID.
    """

    text: str = ""
    blocks: list[dict[str, Any]] = field(default_factory=list)
    thread_ts: str = ""
    channel: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to Slack API format."""
        message: dict[str, Any] = {}

        if self.text:
            message["text"] = self.text

        if self.blocks:
            message["blocks"] = self.blocks

        if self.thread_ts:
            message["thread_ts"] = self.thread_ts

        if self.channel:
            message["channel"] = self.channel

        return message


class SlackBlockBuilder:
    """Builder for Slack Block Kit blocks."""

    def __init__(self) -> None:
        """Initialize builder."""
        self._blocks: list[dict[str, Any]] = []

    @property
    def blocks(self) -> list[dict[str, Any]]:
        """Get the built blocks."""
        return self._blocks

    def add_header(self, text: str) -> "SlackBlockBuilder":
        """Add a header block.

        Args:
            text: Header text.

        Returns:
            Self for chaining.
        """
        self._blocks.append(
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": text,
                    "emoji": True,
                },
            }
        )
        return self

    def add_section(
        self,
        text: str,
        markdown: bool = True,
        accessory: dict[str, Any] | None = None,
    ) -> "SlackBlockBuilder":
        """Add a section block.

        Args:
            text: Section text.
            markdown: Whether text is markdown.
            accessory: Optional accessory element.

        Returns:
            Self for chaining.
        """
        block: dict[str, Any] = {
            "type": "section",
            "text": {
                "type": "mrkdwn" if markdown else "plain_text",
                "text": text,
            },
        }
        if accessory:
            block["accessory"] = accessory

        self._blocks.append(block)
        return self

    def add_section_fields(
        self,
        fields: list[tuple[str, str]],
        markdown: bool = True,
    ) -> "SlackBlockBuilder":
        """Add a section with fields.

        Args:
            fields: List of (label, value) tuples.
            markdown: Whether values are markdown.

        Returns:
            Self for chaining.
        """
        self._blocks.append(
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn" if markdown else "plain_text",
                        "text": f"*{label}*\n{value}",
                    }
                    for label, value in fields
                ],
            }
        )
        return self

    def add_divider(self) -> "SlackBlockBuilder":
        """Add a divider block.

        Returns:
            Self for chaining.
        """
        self._blocks.append({"type": "divider"})
        return self

    def add_context(self, elements: list[str]) -> "SlackBlockBuilder":
        """Add a context block.

        Args:
            elements: List of context texts.

        Returns:
            Self for chaining.
        """
        self._blocks.append(
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": text} for text in elements],
            }
        )
        return self

    def add_button(
        self,
        text: str,
        action_id: str,
        value: str = "",
        style: str = "",
        url: str = "",
    ) -> "SlackBlockBuilder":
        """Add an actions block with a button.

        Args:
            text: Button text.
            action_id: Action identifier.
            value: Button value.
            style: Button style (primary, danger).
            url: URL for link button.

        Returns:
            Self for chaining.
        """
        button: dict[str, Any] = {
            "type": "button",
            "text": {
                "type": "plain_text",
                "text": text,
                "emoji": True,
            },
            "action_id": action_id,
        }

        if value:
            button["value"] = value
        if style:
            button["style"] = style
        if url:
            button["url"] = url

        self._blocks.append(
            {
                "type": "actions",
                "elements": [button],
            }
        )
        return self

    def build(self) -> list[dict[str, Any]]:
        """Build and return the blocks.

        Returns:
            List of Slack blocks.
        """
        return self._blocks


@dataclass
class SlackEvent:
    """Slack event from Events API.

    Attributes:
        type: Event type.
        user: User ID.
        channel: Channel ID.
        text: Message text.
        ts: Timestamp.
        thread_ts: Thread timestamp.
        team: Team ID.
        raw: Raw event data.
    """

    type: str = ""
    user: str = ""
    channel: str = ""
    text: str = ""
    ts: str = ""
    thread_ts: str = ""
    team: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SlackEvent":
        """Create from raw event data."""
        event = data.get("event", data)

        return cls(
            type=event.get("type", ""),
            user=event.get("user", ""),
            channel=event.get("channel", ""),
            text=event.get("text", ""),
            ts=event.get("ts", ""),
            thread_ts=event.get("thread_ts", ""),
            team=data.get("team_id", ""),
            raw=data,
        )


class SlackWebhookHandler:
    """Handler for Slack webhook requests.

    Processes incoming Slack events and generates appropriate responses.
    """

    def __init__(
        self,
        signing_secret: str | None = None,
        bot_token: str | None = None,
    ) -> None:
        """Initialize handler.

        Args:
            signing_secret: Slack signing secret for verification.
            bot_token: Bot token for API calls.
        """
        self._signing_secret = signing_secret or os.getenv("SLACK_SIGNING_SECRET", "")
        self._bot_token = bot_token or os.getenv("SLACK_BOT_TOKEN", "")

    def verify_signature(
        self,
        body: bytes,
        timestamp: str,
        signature: str,
    ) -> bool:
        """Verify Slack request signature.

        Args:
            body: Raw request body.
            timestamp: X-Slack-Request-Timestamp header.
            signature: X-Slack-Signature header.

        Returns:
            True if signature is valid.
        """
        if not self._signing_secret:
            logger.warning("No signing secret configured, skipping verification")
            return True

        # Check timestamp to prevent replay attacks
        if abs(time.time() - int(timestamp)) > 60 * 5:
            logger.warning("Slack request timestamp too old")
            return False

        # Compute signature
        sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
        computed_signature = (
            "v0="
            + hmac.new(
                self._signing_secret.encode("utf-8"),
                sig_basestring.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
        )

        return hmac.compare_digest(computed_signature, signature)

    def parse_event(self, body: dict[str, Any]) -> SlackEvent:
        """Parse incoming event.

        Args:
            body: Request body.

        Returns:
            Parsed event.
        """
        return SlackEvent.from_dict(body)

    def create_response(
        self,
        text: str,
        blocks: list[dict[str, Any]] | None = None,
        thread_ts: str = "",
        channel: str = "",
    ) -> SlackMessage:
        """Create a response message.

        Args:
            text: Message text.
            blocks: Optional Block Kit blocks.
            thread_ts: Thread timestamp for reply.
            channel: Channel ID.

        Returns:
            Slack message.
        """
        return SlackMessage(
            text=text,
            blocks=blocks or [],
            thread_ts=thread_ts,
            channel=channel,
        )

    def create_error_response(
        self,
        error_message: str,
        thread_ts: str = "",
    ) -> SlackMessage:
        """Create an error response.

        Args:
            error_message: Error description.
            thread_ts: Thread timestamp for reply.

        Returns:
            Error message.
        """
        builder = SlackBlockBuilder()
        builder.add_section(f":x: *Error*\n{error_message}")

        return SlackMessage(
            text=f"Error: {error_message}",
            blocks=builder.blocks,
            thread_ts=thread_ts,
        )

    def format_agent_response(
        self,
        agent_response: str,
        agent_type: str = "AI Agent",
        session_id: str = "",
        thread_ts: str = "",
    ) -> SlackMessage:
        """Format an agent response for Slack.

        Args:
            agent_response: Response from the AI agent.
            agent_type: Type of agent that responded.
            session_id: Session ID.
            thread_ts: Thread timestamp for reply.

        Returns:
            Formatted Slack message.
        """
        builder = SlackBlockBuilder()

        # Add main response
        builder.add_section(agent_response)

        # Add metadata context
        if session_id:
            builder.add_divider()
            builder.add_context(
                [
                    f":robot_face: *{agent_type}*",
                    f":id: `{session_id[:8]}...`",
                    f":clock1: {datetime.now().strftime('%H:%M:%S')}",
                ]
            )

        return SlackMessage(
            text=agent_response,
            blocks=builder.blocks,
            thread_ts=thread_ts,
        )

    def create_welcome_message(self) -> SlackMessage:
        """Create a welcome message.

        Returns:
            Welcome message.
        """
        builder = SlackBlockBuilder()
        builder.add_header("AI Agent Ready")
        builder.add_section("Hello! :wave: I'm your AI assistant. How can I help you today?")
        builder.add_divider()
        builder.add_section_fields(
            [
                ("IT Helpdesk", "Password resets, software help"),
                ("ServiceNow", "Tickets and change requests"),
            ]
        )

        return SlackMessage(
            text="AI Agent is ready to help!",
            blocks=builder.blocks,
        )


def verify_slack_signature(
    body: bytes,
    timestamp: str,
    signature: str,
    signing_secret: str | None = None,
) -> bool:
    """Verify Slack request signature.

    Standalone function for use in middleware or dependencies.

    Args:
        body: Raw request body.
        timestamp: X-Slack-Request-Timestamp header.
        signature: X-Slack-Signature header.
        signing_secret: Slack signing secret.

    Returns:
        True if signature is valid.
    """
    secret = signing_secret or os.getenv("SLACK_SIGNING_SECRET", "")
    if not secret:
        return True

    if abs(time.time() - int(timestamp)) > 60 * 5:
        return False

    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    computed = (
        "v0="
        + hmac.new(
            secret.encode("utf-8"),
            sig_basestring.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
    )

    return hmac.compare_digest(computed, signature)


async def process_slack_webhook(
    body: dict[str, Any],
    agent_callback: Any,
    handler: SlackWebhookHandler | None = None,
) -> dict[str, Any]:
    """Process a Slack webhook request.

    Args:
        body: Request body from Slack.
        agent_callback: Async callback for agent processing.
        handler: Optional handler instance.

    Returns:
        Response to send back to Slack.
    """
    if handler is None:
        handler = SlackWebhookHandler()

    try:
        # Handle URL verification challenge
        if body.get("type") == "url_verification":
            return {"challenge": body.get("challenge", "")}

        # Handle event callback
        if body.get("type") == "event_callback":
            event = handler.parse_event(body)

            # Skip bot messages to avoid loops
            if body.get("event", {}).get("bot_id"):
                return {"ok": True}

            if event.type in ("message", "app_mention"):
                user_message = event.text

                # Remove bot mention from message
                if event.type == "app_mention":
                    # Remove <@BOT_ID> mentions
                    import re

                    user_message = re.sub(r"<@[A-Z0-9]+>\s*", "", user_message).strip()

                if not user_message:
                    return {"ok": True}

                # Call agent
                try:
                    response = await agent_callback(
                        message=user_message,
                        user_id=event.user,
                        conversation_id=event.channel,
                        thread_ts=event.thread_ts or event.ts,
                        channel="slack",
                    )

                    if isinstance(response, dict):
                        agent_response = response.get("response", str(response))
                        session_id = response.get("session_id", "")
                        agent_type = response.get("agent_type", "AI Agent")
                    else:
                        agent_response = str(response)
                        session_id = ""
                        agent_type = "AI Agent"

                    message = handler.format_agent_response(
                        agent_response=agent_response,
                        agent_type=agent_type,
                        session_id=session_id,
                        thread_ts=event.thread_ts or event.ts,
                    )

                    return message.to_dict()

                except Exception as e:
                    logger.error(f"Agent error: {e}")
                    error_msg = handler.create_error_response(
                        str(e),
                        thread_ts=event.thread_ts or event.ts,
                    )
                    return error_msg.to_dict()

        return {"ok": True}

    except Exception as e:
        logger.error(f"Slack webhook error: {e}")
        return {"error": str(e)}
