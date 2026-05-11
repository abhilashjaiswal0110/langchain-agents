"""Research agent load scenario."""
import random

from locust import between, task
from locust.contrib.fasthttp import FastHttpUser

_QUERIES = [
    "What are the latest trends in enterprise AI?",
    "Summarise recent advances in LangChain and LangGraph.",
    "What is retrieval-augmented generation and when should I use it?",
    "Compare OpenAI GPT-4o and Anthropic Claude 3 for enterprise use.",
    "How do vector databases improve AI application performance?",
]


class ResearchAgentUser(FastHttpUser):
    """Simulates users invoking the Research Enterprise Agent."""

    wait_time = between(2, 5)
    weight = 3

    @task
    def invoke_research(self) -> None:
        query = random.choice(_QUERIES)
        with self.client.post(
            "/api/enterprise/research/invoke",
            json={"input": query},
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Unexpected status {resp.status_code}")
