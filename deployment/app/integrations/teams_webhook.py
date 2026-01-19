"""Microsoft Teams webhook integration.

Provides webhook handling for Microsoft Teams bot integration,
supporting Adaptive Cards, message formatting, and JWT verification.
"""

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# =============================================================================
# JWT Verification Configuration
# =============================================================================

# Microsoft Bot Framework OpenID configuration URLs
OPENID_METADATA_URL = "https://login.botframework.com/v1/.well-known/openidconfiguration"
EMULATOR_OPENID_URL = "https://login.microsoftonline.com/botframework.com/v2.0/.well-known/openid-configuration"

# Valid token issuers for Microsoft Bot Framework
VALID_TOKEN_ISSUERS = [
    "https://api.botframework.com",
    "https://sts.windows.net/d6d49420-f39b-4df7-a1dc-d59a935871db/",
    "https://login.microsoftonline.com/d6d49420-f39b-4df7-a1dc-d59a935871db/v2.0",
    "https://sts.windows.net/f8cdef31-a31e-4b4a-93e4-5f571e91255a/",
    "https://login.microsoftonline.com/f8cdef31-a31e-4b4a-93e4-5f571e91255a/v2.0",
]

# JWT verification enabled by default for production
TEAMS_JWT_VERIFICATION_ENABLED = os.getenv("TEAMS_JWT_VERIFICATION", "true").lower() == "true"


@dataclass
class JWTVerificationResult:
    """Result of JWT verification.

    Attributes:
        valid: Whether the token is valid.
        error: Error message if invalid.
        claims: Decoded token claims if valid.
    """

    valid: bool
    error: str = ""
    claims: dict[str, Any] = field(default_factory=dict)


class TeamsJWTVerifier:
    """JWT verifier for Microsoft Teams Bot Framework tokens.

    Validates incoming requests from Microsoft Teams using JWT verification
    per the Bot Framework authentication specification.
    """

    def __init__(
        self,
        app_id: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        """Initialize verifier.

        Args:
            app_id: Microsoft App ID (Bot registration).
            tenant_id: Azure AD tenant ID (optional, for single-tenant bots).
        """
        self._app_id = app_id or os.getenv("TEAMS_APP_ID", "")
        self._tenant_id = tenant_id or os.getenv("AZURE_TENANT_ID", "")
        self._jwks_cache: dict[str, Any] = {}
        self._jwks_cache_time: float = 0
        self._jwks_cache_ttl: int = 3600  # 1 hour

    def _get_jwks(self) -> dict[str, Any] | None:
        """Get JSON Web Key Set from Microsoft.

        Returns:
            JWKS dictionary or None if unavailable.
        """
        try:
            import requests

            # Check cache
            if self._jwks_cache and (time.time() - self._jwks_cache_time) < self._jwks_cache_ttl:
                return self._jwks_cache

            # Fetch OpenID configuration
            response = requests.get(OPENID_METADATA_URL, timeout=10)
            response.raise_for_status()
            openid_config = response.json()

            # Fetch JWKS
            jwks_uri = openid_config.get("jwks_uri")
            if not jwks_uri:
                logger.error("No jwks_uri in OpenID configuration")
                return None

            jwks_response = requests.get(jwks_uri, timeout=10)
            jwks_response.raise_for_status()

            self._jwks_cache = jwks_response.json()
            self._jwks_cache_time = time.time()
            return self._jwks_cache

        except ImportError:
            logger.warning("requests package not installed for JWT verification")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch JWKS: {e}")
            return None

    def verify_token(
        self,
        token: str,
        service_url: str | None = None,
    ) -> JWTVerificationResult:
        """Verify a JWT token from Teams/Bot Framework.

        Args:
            token: The JWT token (from Authorization header).
            service_url: The service URL from the activity (optional).

        Returns:
            Verification result with claims if valid.
        """
        if not token:
            return JWTVerificationResult(valid=False, error="No token provided")

        # Remove "Bearer " prefix if present
        if token.startswith("Bearer "):
            token = token[7:]

        try:
            import jwt
            from jwt import PyJWKClient

            # Get JWKS for verification
            jwks = self._get_jwks()
            if not jwks:
                # SECURITY: Fail closed - reject token if we can't verify signature
                logger.error("SECURITY: Cannot fetch JWKS - rejecting JWT token")
                return JWTVerificationResult(
                    valid=False,
                    claims=None,
                    error="JWKS unavailable - cannot verify token signature",
                )

            # Create JWKS client
            jwks_client = PyJWKClient(jwks_uri=OPENID_METADATA_URL.replace(".well-known/openidconfiguration", ".well-known/keys"))

            # Get signing key
            signing_key = jwks_client.get_signing_key_from_jwt(token)

            # Verify and decode token
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                options={
                    "verify_aud": bool(self._app_id),
                    "verify_iss": True,
                },
                audience=self._app_id if self._app_id else None,
                issuer=VALID_TOKEN_ISSUERS,
            )

            # Additional validation: check service URL if provided
            if service_url and "serviceurl" in claims:
                token_service_url = claims["serviceurl"]
                if not self._validate_service_url(service_url, token_service_url):
                    return JWTVerificationResult(
                        valid=False,
                        error=f"Service URL mismatch: {service_url} vs {token_service_url}",
                    )

            # Check token expiration manually for extra safety
            exp = claims.get("exp")
            if exp and time.time() > exp:
                return JWTVerificationResult(valid=False, error="Token expired")

            return JWTVerificationResult(valid=True, claims=claims)

        except ImportError:
            logger.warning("PyJWT package not installed, JWT verification disabled")
            return JWTVerificationResult(
                valid=True,
                error="JWT verification skipped (PyJWT not installed)",
            )
        except jwt.ExpiredSignatureError:
            return JWTVerificationResult(valid=False, error="Token expired")
        except jwt.InvalidAudienceError:
            return JWTVerificationResult(valid=False, error="Invalid audience")
        except jwt.InvalidIssuerError:
            return JWTVerificationResult(valid=False, error="Invalid issuer")
        except jwt.InvalidTokenError as e:
            return JWTVerificationResult(valid=False, error=f"Invalid token: {e}")
        except Exception as e:
            logger.error(f"JWT verification error: {e}")
            return JWTVerificationResult(valid=False, error=str(e))

    def _validate_service_url(self, activity_url: str, token_url: str) -> bool:
        """Validate service URL matches between activity and token.

        Args:
            activity_url: Service URL from the activity.
            token_url: Service URL from the token.

        Returns:
            True if URLs match.
        """
        try:
            activity_parsed = urlparse(activity_url)
            token_parsed = urlparse(token_url)

            # Compare scheme and host (case-insensitive)
            return (
                activity_parsed.scheme.lower() == token_parsed.scheme.lower()
                and activity_parsed.netloc.lower() == token_parsed.netloc.lower()
            )
        except Exception:
            return False


def verify_teams_request(
    authorization_header: str | None,
    body: dict[str, Any],
    app_id: str | None = None,
) -> JWTVerificationResult:
    """Verify an incoming Teams webhook request.

    Args:
        authorization_header: The Authorization header value.
        body: The request body (activity).
        app_id: Optional App ID override.

    Returns:
        Verification result.
    """
    if not TEAMS_JWT_VERIFICATION_ENABLED:
        return JWTVerificationResult(
            valid=True,
            error="JWT verification disabled",
        )

    if not authorization_header:
        return JWTVerificationResult(
            valid=False,
            error="Missing Authorization header",
        )

    verifier = TeamsJWTVerifier(app_id=app_id)
    service_url = body.get("serviceUrl")

    return verifier.verify_token(authorization_header, service_url)


class TeamsCardType(str, Enum):
    """Types of Teams cards."""

    MESSAGE = "message"
    ADAPTIVE = "adaptive"
    HERO = "hero"
    THUMBNAIL = "thumbnail"


@dataclass
class TeamsMessageCard:
    """Simple Teams message card.

    Attributes:
        title: Card title.
        text: Main message text.
        theme_color: Accent color (hex without #).
        sections: Additional sections.
    """

    title: str = ""
    text: str = ""
    theme_color: str = "0078D7"  # Microsoft Blue
    sections: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to Teams-compatible dictionary."""
        card = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": self.theme_color,
            "summary": self.title or "AI Agent Response",
        }

        if self.title:
            card["title"] = self.title

        if self.text:
            card["text"] = self.text

        if self.sections:
            card["sections"] = self.sections

        return card

    def add_section(
        self,
        activity_title: str = "",
        activity_subtitle: str = "",
        text: str = "",
        facts: list[tuple[str, str]] | None = None,
    ) -> "TeamsMessageCard":
        """Add a section to the card.

        Args:
            activity_title: Section title.
            activity_subtitle: Section subtitle.
            text: Section text.
            facts: List of (name, value) tuples.

        Returns:
            Self for chaining.
        """
        section: dict[str, Any] = {}

        if activity_title:
            section["activityTitle"] = activity_title
        if activity_subtitle:
            section["activitySubtitle"] = activity_subtitle
        if text:
            section["text"] = text
        if facts:
            section["facts"] = [
                {"name": name, "value": value} for name, value in facts
            ]

        self.sections.append(section)
        return self


@dataclass
class TeamsAdaptiveCard:
    """Teams Adaptive Card with rich formatting.

    Attributes:
        version: Adaptive Card schema version.
        body: Card body elements.
        actions: Card actions.
    """

    version: str = "1.4"
    body: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to Adaptive Card format."""
        return {
            "type": "AdaptiveCard",
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "version": self.version,
            "body": self.body,
            "actions": self.actions,
        }

    def add_text_block(
        self,
        text: str,
        size: str = "default",
        weight: str = "default",
        wrap: bool = True,
        color: str = "default",
    ) -> "TeamsAdaptiveCard":
        """Add a text block.

        Args:
            text: Text content.
            size: Text size (small, default, medium, large, extraLarge).
            weight: Font weight (lighter, default, bolder).
            wrap: Whether to wrap text.
            color: Text color.

        Returns:
            Self for chaining.
        """
        block = {
            "type": "TextBlock",
            "text": text,
            "wrap": wrap,
        }
        if size != "default":
            block["size"] = size
        if weight != "default":
            block["weight"] = weight
        if color != "default":
            block["color"] = color

        self.body.append(block)
        return self

    def add_fact_set(self, facts: list[tuple[str, str]]) -> "TeamsAdaptiveCard":
        """Add a fact set.

        Args:
            facts: List of (title, value) tuples.

        Returns:
            Self for chaining.
        """
        self.body.append({
            "type": "FactSet",
            "facts": [
                {"title": title, "value": value}
                for title, value in facts
            ],
        })
        return self

    def add_action_button(
        self,
        title: str,
        url: str | None = None,
        data: dict | None = None,
    ) -> "TeamsAdaptiveCard":
        """Add an action button.

        Args:
            title: Button label.
            url: URL for OpenUrl action.
            data: Data for Submit action.

        Returns:
            Self for chaining.
        """
        if url:
            self.actions.append({
                "type": "Action.OpenUrl",
                "title": title,
                "url": url,
            })
        elif data:
            self.actions.append({
                "type": "Action.Submit",
                "title": title,
                "data": data,
            })
        return self


@dataclass
class TeamsActivity:
    """Teams Bot Framework activity.

    Represents an incoming activity from Teams.
    """

    type: str = ""
    id: str = ""
    timestamp: str = ""
    channel_id: str = ""
    conversation_id: str = ""
    from_id: str = ""
    from_name: str = ""
    text: str = ""
    value: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TeamsActivity":
        """Create from raw activity data."""
        conversation = data.get("conversation", {})
        from_user = data.get("from", {})

        return cls(
            type=data.get("type", ""),
            id=data.get("id", ""),
            timestamp=data.get("timestamp", ""),
            channel_id=data.get("channelId", ""),
            conversation_id=conversation.get("id", ""),
            from_id=from_user.get("id", ""),
            from_name=from_user.get("name", ""),
            text=data.get("text", ""),
            value=data.get("value", {}),
            raw=data,
        )


class TeamsWebhookHandler:
    """Handler for Teams webhook requests.

    Processes incoming Teams messages and generates appropriate responses.
    """

    def __init__(
        self,
        app_id: str | None = None,
        app_password: str | None = None,
    ) -> None:
        """Initialize handler.

        Args:
            app_id: Microsoft App ID.
            app_password: Microsoft App Password.
        """
        self._app_id = app_id or os.getenv("TEAMS_APP_ID", "")
        self._app_password = app_password or os.getenv("TEAMS_APP_PASSWORD", "")

    def parse_activity(self, body: dict[str, Any]) -> TeamsActivity:
        """Parse incoming activity.

        Args:
            body: Raw request body.

        Returns:
            Parsed activity.
        """
        return TeamsActivity.from_dict(body)

    def create_response(
        self,
        text: str,
        card: TeamsMessageCard | TeamsAdaptiveCard | None = None,
    ) -> dict[str, Any]:
        """Create a response payload.

        Args:
            text: Response text.
            card: Optional card attachment.

        Returns:
            Response dictionary.
        """
        response: dict[str, Any] = {
            "type": "message",
            "text": text,
        }

        if card:
            if isinstance(card, TeamsAdaptiveCard):
                response["attachments"] = [{
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": card.to_dict(),
                }]
            elif isinstance(card, TeamsMessageCard):
                response["attachments"] = [{
                    "contentType": "application/vnd.microsoft.teams.card.o365connector",
                    "content": card.to_dict(),
                }]

        return response

    def create_typing_activity(self) -> dict[str, Any]:
        """Create a typing indicator activity.

        Returns:
            Typing indicator payload.
        """
        return {"type": "typing"}

    def create_error_response(self, error_message: str) -> dict[str, Any]:
        """Create an error response.

        Args:
            error_message: Error description.

        Returns:
            Error response with card.
        """
        card = TeamsMessageCard(
            title="Error",
            text=error_message,
            theme_color="FF0000",
        )
        return self.create_response(
            text=f"Error: {error_message}",
            card=card,
        )

    def format_agent_response(
        self,
        agent_response: str,
        agent_type: str = "AI Agent",
        session_id: str = "",
    ) -> dict[str, Any]:
        """Format an agent response for Teams.

        Args:
            agent_response: Response from the AI agent.
            agent_type: Type of agent that responded.
            session_id: Session ID.

        Returns:
            Formatted Teams response.
        """
        card = TeamsAdaptiveCard()
        card.add_text_block(
            agent_response,
            wrap=True,
        )

        if session_id:
            card.add_fact_set([
                ("Agent", agent_type),
                ("Session", session_id[:8] + "..."),
                ("Time", datetime.now().strftime("%H:%M:%S")),
            ])

        return self.create_response(agent_response, card)


async def process_teams_webhook(
    body: dict[str, Any],
    agent_callback: Any,
    handler: TeamsWebhookHandler | None = None,
    authorization_header: str | None = None,
) -> dict[str, Any]:
    """Process a Teams webhook request.

    Args:
        body: Request body from Teams.
        agent_callback: Async callback for agent processing.
        handler: Optional handler instance.
        authorization_header: Authorization header for JWT verification.

    Returns:
        Response to send back to Teams.
    """
    if handler is None:
        handler = TeamsWebhookHandler()

    # Verify JWT token if enabled
    if TEAMS_JWT_VERIFICATION_ENABLED:
        verification = verify_teams_request(authorization_header, body)
        if not verification.valid:
            logger.warning(f"Teams JWT verification failed: {verification.error}")
            return handler.create_error_response(
                f"Authentication failed: {verification.error}"
            )

    try:
        activity = handler.parse_activity(body)

        if activity.type == "message":
            # Get user message
            user_message = activity.text

            if not user_message:
                return handler.create_response("I didn't receive any message.")

            # Call agent
            try:
                response = await agent_callback(
                    message=user_message,
                    user_id=activity.from_id,
                    conversation_id=activity.conversation_id,
                    channel="teams",
                )

                if isinstance(response, dict):
                    agent_response = response.get("response", str(response))
                    session_id = response.get("session_id", "")
                    agent_type = response.get("agent_type", "AI Agent")
                else:
                    agent_response = str(response)
                    session_id = ""
                    agent_type = "AI Agent"

                return handler.format_agent_response(
                    agent_response=agent_response,
                    agent_type=agent_type,
                    session_id=session_id,
                )

            except Exception as e:
                logger.error(f"Agent error: {e}")
                return handler.create_error_response(str(e))

        elif activity.type == "conversationUpdate":
            # Handle conversation update (bot added/removed)
            return handler.create_response(
                "Hello! I'm your AI assistant. How can I help you today?"
            )

        else:
            # Acknowledge other activity types
            return {"type": "message", "text": ""}

    except Exception as e:
        logger.error(f"Teams webhook error: {e}")
        return handler.create_error_response(f"Processing error: {str(e)}")
