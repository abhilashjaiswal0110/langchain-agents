"""Conversational IT helpdesk load scenario (multi-turn sessions)."""
import random

from locust import between, task
from locust.contrib.fasthttp import FastHttpUser

_TURNS = [
    "I can't log in to my laptop",
    "My password expired this morning",
    "I need to reset my VPN credentials",
    "Outlook is not syncing my calendar",
    "Can you raise a ticket for a software install?",
]


class ConversationUser(FastHttpUser):
    """Simulates a multi-turn IT helpdesk conversation session."""

    wait_time = between(1, 4)
    weight = 2

    _session_id: str | None = None

    def on_start(self) -> None:
        with self.client.post(
            "/api/conversation/start",
            json={"agent_type": "it_helpdesk", "user_id": "load-test-user"},
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                self._session_id = resp.json().get("session_id")
                resp.success()
            else:
                resp.failure(f"Conversation start failed: {resp.status_code}")

    @task(4)
    def send_message(self) -> None:
        if not self._session_id:
            return
        msg = random.choice(_TURNS)
        with self.client.post(
            "/api/conversation/chat",
            json={"session_id": self._session_id, "message": msg},
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Chat failed: {resp.status_code}")

    @task(1)
    def check_health(self) -> None:
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Health check failed: {resp.status_code}")
