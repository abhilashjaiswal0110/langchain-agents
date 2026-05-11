"""Exponential-backoff retry handler with a Dead Letter Queue for outbound webhooks.

Failed deliveries are stored in an in-process DLQ that can be inspected via the
admin endpoint ``GET /api/integrations/dlq``.

Usage:
    from app.integrations.retry_handler import get_retry_handler

    handler = get_retry_handler()
    ok = await handler.send_with_retry(url, payload, headers)
    if not ok:
        # Message is now in the DLQ
        entries = handler.get_dlq()
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_BACKOFF = [1, 5, 25]  # seconds between attempts


class WebhookRetryHandler:
    """Sends outbound webhook payloads with exponential back-off.

    Attempts delivery up to ``len(backoff_seconds) + 1`` times (one initial
    attempt followed by one retry per entry). On total failure the entry is
    appended to the in-memory Dead Letter Queue.

    Args:
        backoff_seconds: Wait times between successive attempts.  Pass
            ``[0, 0, 0]`` in tests to avoid real sleeps.
    """

    def __init__(self, backoff_seconds: list[int] | None = None) -> None:
        self._backoff = backoff_seconds if backoff_seconds is not None else _DEFAULT_BACKOFF
        self._dlq: list[dict[str, Any]] = []

    async def send_with_retry(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> bool:
        """Deliver *payload* to *url* with retries.

        4xx responses are not retried (client errors are permanent).
        5xx responses and network exceptions are retried up to
        ``len(backoff_seconds)`` additional times.

        Args:
            url: Destination webhook URL.
            payload: JSON-serialisable request body.
            headers: HTTP headers to include.

        Returns:
            ``True`` on successful delivery, ``False`` after all attempts fail.
        """
        attempts = len(self._backoff) + 1

        for attempt in range(attempts):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=payload, headers=headers, timeout=10)
                    if response.status_code < 500:
                        if response.status_code < 400:
                            logger.debug("Webhook delivered to %s (attempt %d)", url, attempt + 1)
                            return True
                        # 4xx — permanent client error, no retry
                        logger.warning(
                            "Webhook %s returned %d (client error); not retrying",
                            url,
                            response.status_code,
                        )
                        self._add_to_dlq(url, payload, f"HTTP {response.status_code}")
                        return False
                    logger.warning(
                        "Webhook %s returned %d on attempt %d",
                        url,
                        response.status_code,
                        attempt + 1,
                    )
            except Exception as exc:
                logger.warning(
                    "Webhook %s raised %s on attempt %d",
                    url,
                    exc,
                    attempt + 1,
                )

            if attempt < len(self._backoff):
                await asyncio.sleep(self._backoff[attempt])

        self._add_to_dlq(url, payload, "max retries exceeded")
        return False

    def _add_to_dlq(self, url: str, payload: dict[str, Any], reason: str) -> None:
        entry = {
            "url": url,
            "payload": payload,
            "reason": reason,
            "failed_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._dlq.append(entry)
        logger.error("Webhook delivery failed, added to DLQ: %s — %s", url, reason)

    def get_dlq(self) -> list[dict[str, Any]]:
        """Return all DLQ entries.

        Returns:
            List of failed delivery records.
        """
        return list(self._dlq)

    def clear_dlq(self) -> int:
        """Clear all DLQ entries and return the count removed.

        Returns:
            Number of entries removed.
        """
        count = len(self._dlq)
        self._dlq.clear()
        return count


_handler_instance: WebhookRetryHandler | None = None


def get_retry_handler() -> WebhookRetryHandler:
    """Return the module-level retry handler singleton.

    Returns:
        Shared WebhookRetryHandler instance.
    """
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = WebhookRetryHandler()
    return _handler_instance
