"""External integrations module for Teams and Slack webhooks.

Provides webhook handlers for Microsoft Teams and Slack bot integrations,
allowing the AI agents to be used directly within collaboration platforms.

Usage:
    from app.integrations import (
        # Teams integration
        TeamsWebhookHandler,
        TeamsMessageCard,
        process_teams_webhook,

        # Slack integration
        SlackWebhookHandler,
        SlackBlockBuilder,
        process_slack_webhook,

        # Router setup
        setup_integration_routes,
    )

    # Add routes to FastAPI app
    setup_integration_routes(app)
"""

from app.integrations.teams_webhook import (
    TeamsWebhookHandler,
    TeamsMessageCard,
    TeamsAdaptiveCard,
    process_teams_webhook,
)
from app.integrations.slack_webhook import (
    SlackWebhookHandler,
    SlackBlockBuilder,
    SlackMessage,
    process_slack_webhook,
    verify_slack_signature,
)
from app.integrations.routes import setup_integration_routes

__all__ = [
    # Teams
    "TeamsWebhookHandler",
    "TeamsMessageCard",
    "TeamsAdaptiveCard",
    "process_teams_webhook",
    # Slack
    "SlackWebhookHandler",
    "SlackBlockBuilder",
    "SlackMessage",
    "process_slack_webhook",
    "verify_slack_signature",
    # Routes
    "setup_integration_routes",
]
