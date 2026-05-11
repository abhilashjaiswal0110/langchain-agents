"""Deep agent (IT Operations) load scenario."""
import random

from locust import between, task
from locust.contrib.fasthttp import FastHttpUser

_MESSAGES = [
    "Analyse the latest incidents from last week",
    "What is the MTTR for P1 incidents in Q1?",
    "Run a root cause analysis for the login service outage",
    "Show change requests pending approval",
    "Check SLA compliance for the database team",
]


class DeepAgentUser(FastHttpUser):
    """Simulates deep agent chat sessions (start + multi-turn chat)."""

    wait_time = between(3, 8)
    weight = 1

    _session_id: str | None = None

    def on_start(self) -> None:
        with self.client.post(
            "/api/deepagent/start",
            json={"agent_type": "it_operations", "user_id": "load-test"},
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                self._session_id = resp.json().get("session_id")
                resp.success()
            else:
                resp.failure(f"Deep agent start failed: {resp.status_code}")

    @task
    def chat(self) -> None:
        if not self._session_id:
            return
        msg = random.choice(_MESSAGES)
        with self.client.post(
            "/api/deepagent/chat",
            json={"session_id": self._session_id, "message": msg},
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Deep agent chat failed: {resp.status_code}")
