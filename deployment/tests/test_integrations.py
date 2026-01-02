"""Unit tests for the integrations module.

Tests cover:
- Microsoft Teams webhook handling
- Slack webhook handling
- Block/Card builders
- Signature verification
"""

import hashlib
import hmac
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.integrations.teams_webhook import (
    TeamsActivity,
    TeamsAdaptiveCard,
    TeamsMessageCard,
    TeamsWebhookHandler,
    process_teams_webhook,
)
from app.integrations.slack_webhook import (
    SlackBlockBuilder,
    SlackEvent,
    SlackMessage,
    SlackWebhookHandler,
    process_slack_webhook,
    verify_slack_signature,
)


# =============================================================================
# Teams MessageCard Tests
# =============================================================================


class TestTeamsMessageCard:
    """Tests for TeamsMessageCard."""

    def test_basic_card(self):
        """Test creating a basic message card."""
        card = TeamsMessageCard(
            title="Test Title",
            text="Test message content",
        )
        data = card.to_dict()

        assert data["@type"] == "MessageCard"
        assert data["title"] == "Test Title"
        assert data["text"] == "Test message content"
        assert data["themeColor"] == "0078D7"

    def test_card_with_custom_color(self):
        """Test card with custom theme color."""
        card = TeamsMessageCard(
            title="Alert",
            text="Warning message",
            theme_color="FF0000",
        )
        data = card.to_dict()

        assert data["themeColor"] == "FF0000"

    def test_add_section(self):
        """Test adding sections to card."""
        card = TeamsMessageCard(title="Report")
        card.add_section(
            activity_title="Section Title",
            text="Section content",
            facts=[("Status", "Active"), ("Priority", "High")],
        )

        data = card.to_dict()
        assert len(data["sections"]) == 1
        assert data["sections"][0]["activityTitle"] == "Section Title"
        assert len(data["sections"][0]["facts"]) == 2

    def test_chained_sections(self):
        """Test chaining multiple sections."""
        card = (
            TeamsMessageCard(title="Multi-Section")
            .add_section(activity_title="First")
            .add_section(activity_title="Second")
            .add_section(activity_title="Third")
        )

        data = card.to_dict()
        assert len(data["sections"]) == 3


# =============================================================================
# Teams AdaptiveCard Tests
# =============================================================================


class TestTeamsAdaptiveCard:
    """Tests for TeamsAdaptiveCard."""

    def test_basic_adaptive_card(self):
        """Test creating a basic adaptive card."""
        card = TeamsAdaptiveCard()
        data = card.to_dict()

        assert data["type"] == "AdaptiveCard"
        assert data["version"] == "1.4"
        assert data["body"] == []
        assert data["actions"] == []

    def test_add_text_block(self):
        """Test adding text blocks."""
        card = TeamsAdaptiveCard()
        card.add_text_block("Hello World", size="large", weight="bolder")

        data = card.to_dict()
        assert len(data["body"]) == 1
        assert data["body"][0]["type"] == "TextBlock"
        assert data["body"][0]["text"] == "Hello World"
        assert data["body"][0]["size"] == "large"
        assert data["body"][0]["weight"] == "bolder"

    def test_add_fact_set(self):
        """Test adding fact sets."""
        card = TeamsAdaptiveCard()
        card.add_fact_set([
            ("Name", "John Doe"),
            ("Email", "john@example.com"),
        ])

        data = card.to_dict()
        assert len(data["body"]) == 1
        assert data["body"][0]["type"] == "FactSet"
        assert len(data["body"][0]["facts"]) == 2

    def test_add_action_button(self):
        """Test adding action buttons."""
        card = TeamsAdaptiveCard()
        card.add_action_button("Open Link", url="https://example.com")
        card.add_action_button("Submit", data={"action": "submit"})

        data = card.to_dict()
        assert len(data["actions"]) == 2
        assert data["actions"][0]["type"] == "Action.OpenUrl"
        assert data["actions"][1]["type"] == "Action.Submit"


# =============================================================================
# Teams Activity Tests
# =============================================================================


class TestTeamsActivity:
    """Tests for TeamsActivity parsing."""

    def test_parse_message_activity(self):
        """Test parsing a message activity."""
        raw = {
            "type": "message",
            "id": "activity-123",
            "timestamp": "2024-01-01T12:00:00Z",
            "channelId": "msteams",
            "conversation": {"id": "conv-456"},
            "from": {"id": "user-789", "name": "John Doe"},
            "text": "Hello bot",
        }

        activity = TeamsActivity.from_dict(raw)

        assert activity.type == "message"
        assert activity.id == "activity-123"
        assert activity.conversation_id == "conv-456"
        assert activity.from_id == "user-789"
        assert activity.from_name == "John Doe"
        assert activity.text == "Hello bot"

    def test_parse_conversation_update(self):
        """Test parsing a conversation update activity."""
        raw = {
            "type": "conversationUpdate",
            "conversation": {"id": "conv-123"},
        }

        activity = TeamsActivity.from_dict(raw)
        assert activity.type == "conversationUpdate"


# =============================================================================
# Teams WebhookHandler Tests
# =============================================================================


class TestTeamsWebhookHandler:
    """Tests for TeamsWebhookHandler."""

    @pytest.fixture
    def handler(self):
        """Create handler instance."""
        return TeamsWebhookHandler()

    def test_parse_activity(self, handler):
        """Test parsing activity."""
        body = {
            "type": "message",
            "text": "Test message",
            "conversation": {"id": "conv-1"},
            "from": {"id": "user-1"},
        }

        activity = handler.parse_activity(body)
        assert activity.type == "message"
        assert activity.text == "Test message"

    def test_create_response(self, handler):
        """Test creating response."""
        response = handler.create_response("Hello!")

        assert response["type"] == "message"
        assert response["text"] == "Hello!"

    def test_create_response_with_card(self, handler):
        """Test creating response with card."""
        card = TeamsMessageCard(title="Test", text="Content")
        response = handler.create_response("Hello!", card=card)

        assert "attachments" in response
        assert len(response["attachments"]) == 1

    def test_create_error_response(self, handler):
        """Test creating error response."""
        response = handler.create_error_response("Something went wrong")

        assert "Error" in response["text"]
        assert "attachments" in response

    def test_format_agent_response(self, handler):
        """Test formatting agent response."""
        response = handler.format_agent_response(
            agent_response="Here is your answer.",
            agent_type="IT Helpdesk",
            session_id="session-12345678",
        )

        assert response["text"] == "Here is your answer."
        assert "attachments" in response


# =============================================================================
# Teams Webhook Processing Tests
# =============================================================================


class TestTeamsWebhookProcessing:
    """Tests for process_teams_webhook."""

    @pytest.mark.asyncio
    async def test_process_message(self):
        """Test processing a message activity."""
        body = {
            "type": "message",
            "text": "Hello",
            "conversation": {"id": "conv-1"},
            "from": {"id": "user-1"},
        }

        async def mock_callback(**kwargs):
            return {"response": "Hi there!"}

        result = await process_teams_webhook(body, mock_callback)

        assert result["type"] == "message"
        assert "Hi there!" in result["text"]

    @pytest.mark.asyncio
    async def test_process_conversation_update(self):
        """Test processing conversation update."""
        body = {
            "type": "conversationUpdate",
            "conversation": {"id": "conv-1"},
        }

        async def mock_callback(**kwargs):
            return {"response": ""}

        result = await process_teams_webhook(body, mock_callback)

        assert "Hello" in result["text"]

    @pytest.mark.asyncio
    async def test_process_empty_message(self):
        """Test processing empty message."""
        body = {
            "type": "message",
            "text": "",
            "conversation": {"id": "conv-1"},
            "from": {"id": "user-1"},
        }

        async def mock_callback(**kwargs):
            return {"response": ""}

        result = await process_teams_webhook(body, mock_callback)

        assert "didn't receive" in result["text"]


# =============================================================================
# Slack Message Tests
# =============================================================================


class TestSlackMessage:
    """Tests for SlackMessage."""

    def test_basic_message(self):
        """Test creating a basic message."""
        msg = SlackMessage(text="Hello Slack!")
        data = msg.to_dict()

        assert data["text"] == "Hello Slack!"

    def test_message_with_blocks(self):
        """Test message with blocks."""
        msg = SlackMessage(
            text="Fallback",
            blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": "Hello"}}],
        )
        data = msg.to_dict()

        assert len(data["blocks"]) == 1
        assert data["text"] == "Fallback"

    def test_threaded_message(self):
        """Test threaded message."""
        msg = SlackMessage(
            text="Reply",
            thread_ts="1234567890.123456",
        )
        data = msg.to_dict()

        assert data["thread_ts"] == "1234567890.123456"


# =============================================================================
# Slack BlockBuilder Tests
# =============================================================================


class TestSlackBlockBuilder:
    """Tests for SlackBlockBuilder."""

    def test_add_header(self):
        """Test adding header."""
        builder = SlackBlockBuilder()
        builder.add_header("Test Header")

        blocks = builder.blocks
        assert len(blocks) == 1
        assert blocks[0]["type"] == "header"
        assert blocks[0]["text"]["text"] == "Test Header"

    def test_add_section(self):
        """Test adding section."""
        builder = SlackBlockBuilder()
        builder.add_section("Some *markdown* text")

        blocks = builder.blocks
        assert len(blocks) == 1
        assert blocks[0]["type"] == "section"
        assert blocks[0]["text"]["type"] == "mrkdwn"

    def test_add_section_fields(self):
        """Test adding section with fields."""
        builder = SlackBlockBuilder()
        builder.add_section_fields([
            ("Field 1", "Value 1"),
            ("Field 2", "Value 2"),
        ])

        blocks = builder.blocks
        assert len(blocks) == 1
        assert len(blocks[0]["fields"]) == 2

    def test_add_divider(self):
        """Test adding divider."""
        builder = SlackBlockBuilder()
        builder.add_divider()

        blocks = builder.blocks
        assert blocks[0]["type"] == "divider"

    def test_add_context(self):
        """Test adding context."""
        builder = SlackBlockBuilder()
        builder.add_context(["Context 1", "Context 2"])

        blocks = builder.blocks
        assert blocks[0]["type"] == "context"
        assert len(blocks[0]["elements"]) == 2

    def test_add_button(self):
        """Test adding button."""
        builder = SlackBlockBuilder()
        builder.add_button("Click Me", "btn-action", value="test")

        blocks = builder.blocks
        assert blocks[0]["type"] == "actions"
        assert blocks[0]["elements"][0]["type"] == "button"

    def test_chaining(self):
        """Test chaining methods."""
        blocks = (
            SlackBlockBuilder()
            .add_header("Title")
            .add_section("Content")
            .add_divider()
            .add_context(["Footer"])
            .build()
        )

        assert len(blocks) == 4


# =============================================================================
# Slack Event Tests
# =============================================================================


class TestSlackEvent:
    """Tests for SlackEvent parsing."""

    def test_parse_message_event(self):
        """Test parsing a message event."""
        raw = {
            "type": "event_callback",
            "team_id": "T12345",
            "event": {
                "type": "message",
                "user": "U12345",
                "channel": "C12345",
                "text": "Hello bot",
                "ts": "1234567890.123456",
            },
        }

        event = SlackEvent.from_dict(raw)

        assert event.type == "message"
        assert event.user == "U12345"
        assert event.channel == "C12345"
        assert event.text == "Hello bot"
        assert event.team == "T12345"

    def test_parse_app_mention(self):
        """Test parsing app mention event."""
        raw = {
            "event": {
                "type": "app_mention",
                "user": "U12345",
                "text": "<@BOTID> help me",
                "channel": "C12345",
            },
        }

        event = SlackEvent.from_dict(raw)
        assert event.type == "app_mention"


# =============================================================================
# Slack Signature Verification Tests
# =============================================================================


class TestSlackSignatureVerification:
    """Tests for Slack signature verification."""

    def test_valid_signature(self):
        """Test valid signature verification."""
        signing_secret = "test_secret"
        timestamp = str(int(time.time()))
        body = b'{"test": "data"}'

        sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
        signature = (
            "v0=" +
            hmac.new(
                signing_secret.encode("utf-8"),
                sig_basestring.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
        )

        result = verify_slack_signature(
            body=body,
            timestamp=timestamp,
            signature=signature,
            signing_secret=signing_secret,
        )

        assert result is True

    def test_invalid_signature(self):
        """Test invalid signature rejection."""
        result = verify_slack_signature(
            body=b'{"test": "data"}',
            timestamp=str(int(time.time())),
            signature="v0=invalid",
            signing_secret="test_secret",
        )

        assert result is False

    def test_old_timestamp_rejection(self):
        """Test old timestamp rejection."""
        old_timestamp = str(int(time.time()) - 600)  # 10 minutes ago

        result = verify_slack_signature(
            body=b'{"test": "data"}',
            timestamp=old_timestamp,
            signature="v0=anything",
            signing_secret="test_secret",
        )

        assert result is False

    def test_no_secret_passes(self):
        """Test that missing secret allows through."""
        result = verify_slack_signature(
            body=b'{"test": "data"}',
            timestamp=str(int(time.time())),
            signature="v0=anything",
            signing_secret="",
        )

        assert result is True


# =============================================================================
# Slack WebhookHandler Tests
# =============================================================================


class TestSlackWebhookHandler:
    """Tests for SlackWebhookHandler."""

    @pytest.fixture
    def handler(self):
        """Create handler instance."""
        return SlackWebhookHandler()

    def test_parse_event(self, handler):
        """Test parsing event."""
        body = {
            "event": {
                "type": "message",
                "text": "Hello",
                "user": "U123",
                "channel": "C123",
            },
        }

        event = handler.parse_event(body)
        assert event.type == "message"
        assert event.text == "Hello"

    def test_create_response(self, handler):
        """Test creating response."""
        response = handler.create_response("Hello!")

        assert response.text == "Hello!"

    def test_create_error_response(self, handler):
        """Test creating error response."""
        response = handler.create_error_response("Error occurred")

        assert "Error" in response.text
        assert len(response.blocks) > 0

    def test_format_agent_response(self, handler):
        """Test formatting agent response."""
        response = handler.format_agent_response(
            agent_response="Here is your answer.",
            agent_type="IT Helpdesk",
            session_id="session-12345678",
        )

        assert response.text == "Here is your answer."
        assert len(response.blocks) > 0

    def test_create_welcome_message(self, handler):
        """Test creating welcome message."""
        response = handler.create_welcome_message()

        assert "AI Agent" in response.text or "ready" in response.text.lower()
        assert len(response.blocks) > 0


# =============================================================================
# Slack Webhook Processing Tests
# =============================================================================


class TestSlackWebhookProcessing:
    """Tests for process_slack_webhook."""

    @pytest.mark.asyncio
    async def test_url_verification(self):
        """Test URL verification challenge."""
        body = {
            "type": "url_verification",
            "challenge": "test-challenge-token",
        }

        async def mock_callback(**kwargs):
            return {"response": ""}

        result = await process_slack_webhook(body, mock_callback)

        assert result["challenge"] == "test-challenge-token"

    @pytest.mark.asyncio
    async def test_process_message_event(self):
        """Test processing message event."""
        body = {
            "type": "event_callback",
            "event": {
                "type": "message",
                "text": "Hello",
                "user": "U123",
                "channel": "C123",
                "ts": "1234567890.123456",
            },
        }

        async def mock_callback(**kwargs):
            return {"response": "Hi there!"}

        result = await process_slack_webhook(body, mock_callback)

        assert "Hi there!" in result.get("text", "")

    @pytest.mark.asyncio
    async def test_skip_bot_messages(self):
        """Test that bot messages are skipped."""
        body = {
            "type": "event_callback",
            "event": {
                "type": "message",
                "text": "Bot response",
                "bot_id": "B123",
            },
        }

        async def mock_callback(**kwargs):
            pytest.fail("Callback should not be called for bot messages")

        result = await process_slack_webhook(body, mock_callback)

        assert result.get("ok") is True

    @pytest.mark.asyncio
    async def test_process_app_mention(self):
        """Test processing app mention."""
        body = {
            "type": "event_callback",
            "event": {
                "type": "app_mention",
                "text": "<@BOTID> help me",
                "user": "U123",
                "channel": "C123",
                "ts": "1234567890.123456",
            },
        }

        async def mock_callback(**kwargs):
            # Should receive "help me" without bot mention
            assert "help me" in kwargs.get("message", "")
            return {"response": "How can I help?"}

        result = await process_slack_webhook(body, mock_callback)

        assert "help" in result.get("text", "").lower()


# =============================================================================
# Integration Module Tests
# =============================================================================


class TestIntegrationModule:
    """Tests for integration module exports."""

    def test_module_exports(self):
        """Test that all exports are available."""
        from app.integrations import (
            TeamsWebhookHandler,
            TeamsMessageCard,
            TeamsAdaptiveCard,
            process_teams_webhook,
            SlackWebhookHandler,
            SlackBlockBuilder,
            SlackMessage,
            process_slack_webhook,
            verify_slack_signature,
            setup_integration_routes,
        )

        assert TeamsWebhookHandler is not None
        assert SlackWebhookHandler is not None
        assert setup_integration_routes is not None
