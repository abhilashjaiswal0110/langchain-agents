"""Locust load testing entrypoint for the Enterprise Agents Platform.

Combines all scenario users into a single file so Locust can run them
together with realistic task weights.

Run:
    locust -f tests/load/locustfile.py --host http://localhost:8000
    locust -f tests/load/locustfile.py --host http://localhost:8000 \\
           --headless -u 50 -r 5 --run-time 60s
"""
import random

from locust import HttpUser, between, task


class AgentLoadUser(HttpUser):
    """General-purpose user that exercises all major agent entry points."""

    wait_time = between(1, 3)

    # ---- Research agent (highest weight — most common) ----

    @task(3)
    def research_query(self) -> None:
        queries = [
            "AI trends in enterprise software 2026",
            "How does LangGraph differ from classic LangChain agents?",
            "Best practices for RAG in production",
        ]
        self.client.post(
            "/api/enterprise/research/invoke",
            json={"input": random.choice(queries)},
            name="/api/enterprise/research/invoke",
        )

    # ---- IT helpdesk conversation (stateless single-turn) ----

    @task(2)
    def helpdesk_conversation(self) -> None:
        r = self.client.post(
            "/api/conversation/start",
            json={"agent_type": "it_helpdesk", "user_id": "load-test"},
            name="/api/conversation/start",
        )
        if r.status_code != 200:
            return
        session_id = r.json().get("session_id")
        if session_id:
            self.client.post(
                "/api/conversation/chat",
                json={"session_id": session_id, "message": "reset my password"},
                name="/api/conversation/chat",
            )

    # ---- Content generation ----

    @task(1)
    def content_generation(self) -> None:
        self.client.post(
            "/api/enterprise/content/invoke",
            json={"input": "Write a LinkedIn post about AI in IT operations"},
            name="/api/enterprise/content/invoke",
        )

    # ---- Analytics metrics (lightweight) ----

    @task(2)
    def analytics_metrics(self) -> None:
        self.client.get("/api/analytics/metrics", name="/api/analytics/metrics")

    # ---- Health check ----

    @task(1)
    def health_check(self) -> None:
        self.client.get("/health", name="/health")

    # ---- Domain agent (cloud) ----

    @task(1)
    def domain_agent_cloud(self) -> None:
        self.client.post(
            "/api/domain/cloud/invoke",
            json={"message": "list my Azure VMs"},
            name="/api/domain/{domain}/invoke",
        )
