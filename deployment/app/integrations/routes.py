"""FastAPI routes for external integrations.

Provides webhook endpoints for Microsoft Teams and Slack integrations.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict

from app.integrations.teams_webhook import (
    TeamsWebhookHandler,
    process_teams_webhook,
)
from app.integrations.slack_webhook import (
    SlackWebhookHandler,
    process_slack_webhook,
    verify_slack_signature,
)

logger = logging.getLogger(__name__)

# Router for integration endpoints
router = APIRouter(prefix="/api/integrations", tags=["integrations"])


# Request/Response models
class TeamsWebhookRequest(BaseModel):
    """Teams webhook request body."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    type: str = ""
    id: str = ""
    timestamp: str = ""
    channelId: str = ""
    conversation: dict[str, Any] = {}
    from_: dict[str, Any] = {}
    text: str = ""
    value: dict[str, Any] = {}


class SlackEventRequest(BaseModel):
    """Slack event request body."""

    model_config = ConfigDict(extra="allow")

    type: str = ""
    token: str = ""
    challenge: str = ""
    team_id: str = ""
    event: dict[str, Any] = {}


class IntegrationResponse(BaseModel):
    """Generic integration response."""

    success: bool = True
    message: str = ""
    data: dict[str, Any] = {}


# Default agent callback (can be replaced)
async def default_agent_callback(
    message: str,
    user_id: str = "",
    conversation_id: str = "",
    thread_ts: str = "",
    channel: str = "",
) -> dict[str, Any]:
    """Default agent callback for testing.

    In production, this should be replaced with actual agent logic.

    Args:
        message: User message.
        user_id: User identifier.
        conversation_id: Conversation/channel ID.
        thread_ts: Thread timestamp (Slack).
        channel: Source channel (teams/slack).

    Returns:
        Agent response.
    """
    # Import the conversation manager for actual agent processing
    try:
        from app.agents.conversation_manager import get_conversation_manager

        manager = get_conversation_manager()

        # Create or get session
        session_info = manager.start_session(
            agent_type="it_helpdesk",
            user_id=user_id or f"{channel}-{conversation_id}",
        )

        # Process message
        result = await manager.chat_async(
            session_id=session_info["session_id"],
            message=message,
        )

        return {
            "response": result.get("response", ""),
            "session_id": session_info["session_id"],
            "agent_type": session_info.get("agent_type", "IT Helpdesk"),
        }

    except ImportError:
        # Fallback response if conversation manager not available
        return {
            "response": f"Received: {message}. Agent integration pending.",
            "session_id": "",
            "agent_type": "Echo Agent",
        }
    except Exception as e:
        logger.error(f"Agent callback error: {e}")
        return {
            "response": f"Error processing request: {str(e)}",
            "session_id": "",
            "agent_type": "Error",
        }


# Store for agent callback
_agent_callback = default_agent_callback


def set_agent_callback(callback: Any) -> None:
    """Set the agent callback function.

    Args:
        callback: Async function to process messages.
    """
    global _agent_callback
    _agent_callback = callback


# Teams endpoint
@router.post("/teams/webhook")
async def teams_webhook(request: Request) -> Response:
    """Handle Microsoft Teams webhook.

    This endpoint receives messages from Teams and processes them
    through the configured AI agent.

    Args:
        request: FastAPI request.

    Returns:
        Teams-formatted response.
    """
    try:
        body = await request.json()
        logger.info(f"Teams webhook received: type={body.get('type', 'unknown')}")

        handler = TeamsWebhookHandler()
        response = await process_teams_webhook(
            body=body,
            agent_callback=_agent_callback,
            handler=handler,
        )

        return Response(
            content=str(response) if isinstance(response, str) else str(response),
            media_type="application/json",
        )

    except Exception as e:
        logger.error(f"Teams webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Slack endpoints
@router.post("/slack/events")
async def slack_events(
    request: Request,
    x_slack_request_timestamp: str = Header(None, alias="X-Slack-Request-Timestamp"),
    x_slack_signature: str = Header(None, alias="X-Slack-Signature"),
) -> dict[str, Any]:
    """Handle Slack Events API webhook.

    This endpoint receives events from Slack's Events API and processes
    messages through the configured AI agent.

    Args:
        request: FastAPI request.
        x_slack_request_timestamp: Slack timestamp header.
        x_slack_signature: Slack signature header.

    Returns:
        Slack-formatted response.
    """
    try:
        # Get raw body for signature verification
        raw_body = await request.body()
        body = await request.json()

        logger.info(f"Slack webhook received: type={body.get('type', 'unknown')}")

        # Verify signature in production
        if x_slack_request_timestamp and x_slack_signature:
            if not verify_slack_signature(
                body=raw_body,
                timestamp=x_slack_request_timestamp,
                signature=x_slack_signature,
            ):
                raise HTTPException(status_code=401, detail="Invalid signature")

        # Handle URL verification
        if body.get("type") == "url_verification":
            return {"challenge": body.get("challenge", "")}

        handler = SlackWebhookHandler()
        response = await process_slack_webhook(
            body=body,
            agent_callback=_agent_callback,
            handler=handler,
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Slack webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/slack/commands")
async def slack_commands(request: Request) -> dict[str, Any]:
    """Handle Slack slash commands.

    Args:
        request: FastAPI request with form data.

    Returns:
        Slack response.
    """
    try:
        form = await request.form()
        command = form.get("command", "")
        text = form.get("text", "")
        user_id = form.get("user_id", "")
        channel_id = form.get("channel_id", "")

        logger.info(f"Slack command received: {command} {text}")

        # Process command
        message = f"{command} {text}".strip()
        if not message:
            return {
                "response_type": "ephemeral",
                "text": "Please provide a message.",
            }

        response = await _agent_callback(
            message=text or "help",
            user_id=user_id,
            conversation_id=channel_id,
            channel="slack",
        )

        return {
            "response_type": "in_channel",
            "text": response.get("response", "No response"),
        }

    except Exception as e:
        logger.error(f"Slack command error: {e}")
        return {
            "response_type": "ephemeral",
            "text": f"Error: {str(e)}",
        }


@router.post("/slack/interactive")
async def slack_interactive(request: Request) -> dict[str, Any]:
    """Handle Slack interactive components.

    Args:
        request: FastAPI request with form data.

    Returns:
        Slack response.
    """
    try:
        form = await request.form()
        payload = form.get("payload", "{}")

        import json
        data = json.loads(payload) if isinstance(payload, str) else payload

        action_type = data.get("type", "")
        logger.info(f"Slack interactive: {action_type}")

        if action_type == "block_actions":
            actions = data.get("actions", [])
            if actions:
                action = actions[0]
                action_id = action.get("action_id", "")
                value = action.get("value", "")

                # Process action
                return {
                    "response_type": "ephemeral",
                    "text": f"Action received: {action_id}",
                }

        return {"ok": True}

    except Exception as e:
        logger.error(f"Slack interactive error: {e}")
        return {"ok": False, "error": str(e)}


# Health check
@router.get("/health")
async def integrations_health() -> dict[str, Any]:
    """Health check for integrations.

    Returns:
        Health status.
    """
    return {
        "status": "healthy",
        "integrations": {
            "teams": "available",
            "slack": "available",
        },
    }


def setup_integration_routes(app: Any) -> None:
    """Set up integration routes on a FastAPI app.

    Args:
        app: FastAPI application instance.
    """
    app.include_router(router)
    logger.info("Integration routes registered at /api/integrations")
