"""Tests for webhook retry handler and Dead Letter Queue."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.integrations.retry_handler import WebhookRetryHandler


@pytest.fixture()
def handler() -> WebhookRetryHandler:
    return WebhookRetryHandler(backoff_seconds=[0, 0, 0])  # instant retries in tests


class TestWebhookRetryHandlerSuccess:
    @pytest.mark.asyncio
    async def test_returns_true_on_first_success(self, handler: WebhookRetryHandler) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await handler.send_with_retry(
                "https://example.com/hook", {"text": "hello"}, {}
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_true_on_second_attempt(self, handler: WebhookRetryHandler) -> None:
        mock_500 = MagicMock()
        mock_500.status_code = 500
        mock_200 = MagicMock()
        mock_200.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[mock_500, mock_200])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await handler.send_with_retry(
                "https://example.com/hook", {"text": "hello"}, {}
            )

        assert result is True


class TestWebhookRetryHandlerFailure:
    @pytest.mark.asyncio
    async def test_returns_false_after_all_retries_fail(self, handler: WebhookRetryHandler) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await handler.send_with_retry(
                "https://example.com/hook", {"text": "fail"}, {}
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_adds_to_dlq_after_all_retries_fail(self, handler: WebhookRetryHandler) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            await handler.send_with_retry(
                "https://example.com/hook", {"key": "val"}, {}
            )

        dlq = handler.get_dlq()
        assert len(dlq) == 1
        assert dlq[0]["url"] == "https://example.com/hook"
        assert "failed_at" in dlq[0]

    @pytest.mark.asyncio
    async def test_returns_false_on_exception(self, handler: WebhookRetryHandler) -> None:
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=ConnectionError("timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await handler.send_with_retry(
                "https://example.com/hook", {}, {}
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_dlq_accumulates_multiple_failures(self, handler: WebhookRetryHandler) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            await handler.send_with_retry("https://a.com", {}, {})
            await handler.send_with_retry("https://b.com", {}, {})

        assert len(handler.get_dlq()) == 2


class TestDLQManagement:
    def test_get_dlq_returns_list(self) -> None:
        handler = WebhookRetryHandler()
        assert isinstance(handler.get_dlq(), list)

    def test_clear_dlq(self) -> None:
        handler = WebhookRetryHandler(backoff_seconds=[0])
        handler._dlq.append({"url": "test", "payload": {}, "failed_at": "now"})
        assert len(handler.get_dlq()) == 1
        handler.clear_dlq()
        assert len(handler.get_dlq()) == 0

    def test_dlq_entry_has_required_fields(self) -> None:
        handler = WebhookRetryHandler(backoff_seconds=[0])
        handler._dlq.append({"url": "test", "payload": {"k": "v"}, "failed_at": "2026-01-01"})
        entry = handler.get_dlq()[0]
        assert "url" in entry
        assert "payload" in entry
        assert "failed_at" in entry

    def test_4xx_stops_retrying(self) -> None:
        """A 4xx response is a client error — no point retrying."""
        handler = WebhookRetryHandler(backoff_seconds=[0, 0])
        mock_response = MagicMock()
        mock_response.status_code = 400

        call_count = 0

        async def run():
            nonlocal call_count
            with patch("httpx.AsyncClient") as mock_client_cls:
                async def fake_post(*a, **kw):
                    nonlocal call_count
                    call_count += 1
                    return mock_response

                mock_client = AsyncMock()
                mock_client.post = fake_post
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_cls.return_value = mock_client

                await handler.send_with_retry("https://x.com", {}, {})

        asyncio.run(run())
        assert call_count == 1
