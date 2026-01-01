"""Conversation summarization for token efficiency.

Provides:
- Sliding window summarization for long conversations
- Token budget management
- Summary storage in semantic memory
"""

import os
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


# Default summarization prompt
SUMMARIZATION_PROMPT = """You are a conversation summarizer. Your task is to create a concise summary of the conversation that captures:

1. Key topics discussed
2. Important decisions or conclusions reached
3. Any action items or follow-ups mentioned
4. User preferences or requirements expressed

Conversation to summarize:
{conversation}

Provide a concise summary (2-4 sentences) that preserves the essential context needed to continue the conversation later."""


class ConversationSummarizer:
    """Summarizes conversations for token efficiency and memory storage."""

    def __init__(
        self,
        llm=None,
        summary_threshold: int | None = None,
        max_tokens_before_summary: int = 4000,
    ):
        """Initialize the summarizer.

        Args:
            llm: Language model for summarization. If None, creates default.
            summary_threshold: Number of messages before triggering summary.
                             If None, reads from CONVERSATION_SUMMARY_THRESHOLD env var.
            max_tokens_before_summary: Approximate token limit before summarization.
        """
        self._llm = llm
        self.summary_threshold = summary_threshold or int(
            os.getenv("CONVERSATION_SUMMARY_THRESHOLD", "10")
        )
        self.max_tokens_before_summary = max_tokens_before_summary
        self._prompt = ChatPromptTemplate.from_template(SUMMARIZATION_PROMPT)

    @property
    def llm(self):
        """Get or create the LLM for summarization."""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def _create_llm(self):
        """Create default LLM for summarization."""
        # Try OpenAI first (faster for summarization)
        try:
            from langchain_openai import ChatOpenAI

            if os.getenv("OPENAI_API_KEY"):
                return ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    max_tokens=500,
                )
        except ImportError:
            pass

        # Fall back to Anthropic
        try:
            from langchain_anthropic import ChatAnthropic

            if os.getenv("ANTHROPIC_API_KEY"):
                return ChatAnthropic(
                    model="claude-3-haiku-20240307",
                    temperature=0,
                    max_tokens=500,
                )
        except ImportError:
            pass

        raise ImportError(
            "No LLM available for summarization. "
            "Install langchain-openai or langchain-anthropic."
        )

    def should_summarize(self, messages: list[BaseMessage]) -> bool:
        """Check if conversation should be summarized.

        Args:
            messages: List of conversation messages.

        Returns:
            True if summarization is recommended.
        """
        # Check message count threshold
        if len(messages) >= self.summary_threshold:
            return True

        # Check approximate token count
        total_chars = sum(len(str(m.content)) for m in messages)
        # Rough estimate: 4 chars per token
        estimated_tokens = total_chars / 4
        if estimated_tokens > self.max_tokens_before_summary:
            return True

        return False

    def format_messages_for_summary(self, messages: list[BaseMessage]) -> str:
        """Format messages into a string for summarization.

        Args:
            messages: List of messages to format.

        Returns:
            Formatted conversation string.
        """
        lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "User"
            elif isinstance(msg, AIMessage):
                role = "Assistant"
            elif isinstance(msg, SystemMessage):
                continue  # Skip system messages
            else:
                role = msg.__class__.__name__

            content = str(msg.content)
            # Truncate very long messages
            if len(content) > 500:
                content = content[:500] + "..."

            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def summarize(self, messages: list[BaseMessage]) -> str:
        """Create a summary of the conversation.

        Args:
            messages: List of messages to summarize.

        Returns:
            Summary string.
        """
        if not messages:
            return ""

        conversation_text = self.format_messages_for_summary(messages)

        try:
            chain = self._prompt | self.llm
            result = chain.invoke({"conversation": conversation_text})
            return result.content if hasattr(result, "content") else str(result)
        except Exception as e:
            print(f"Warning: Summarization failed: {e}")
            # Fallback: Create a simple summary
            return self._create_fallback_summary(messages)

    def _create_fallback_summary(self, messages: list[BaseMessage]) -> str:
        """Create a simple summary without LLM.

        Args:
            messages: List of messages to summarize.

        Returns:
            Basic summary string.
        """
        human_messages = [m for m in messages if isinstance(m, HumanMessage)]
        if not human_messages:
            return "Conversation with no user messages."

        # Get first and last user message topics
        first = str(human_messages[0].content)[:100]
        last = str(human_messages[-1].content)[:100]

        if len(human_messages) == 1:
            return f"User discussed: {first}"

        return f"Conversation started with: {first}... Most recent topic: {last}"

    def summarize_and_compress(
        self,
        messages: list[BaseMessage],
        keep_recent: int = 3,
    ) -> tuple[list[BaseMessage], str | None]:
        """Summarize older messages and keep recent ones.

        This creates a sliding window effect where older messages
        are summarized while recent messages are kept intact.

        Args:
            messages: Full list of conversation messages.
            keep_recent: Number of recent messages to keep intact.

        Returns:
            Tuple of (compressed messages list, summary if created)
        """
        if not self.should_summarize(messages):
            return messages, None

        # Separate system messages (keep all)
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]

        if len(non_system) <= keep_recent:
            return messages, None

        # Split into old (to summarize) and recent (to keep)
        old_messages = non_system[:-keep_recent]
        recent_messages = non_system[-keep_recent:]

        # Create summary of old messages
        summary = self.summarize(old_messages)

        # Create a system message with the summary
        summary_message = SystemMessage(
            content=f"## Previous Conversation Summary\n{summary}\n\n"
            f"(Summarized from {len(old_messages)} messages)"
        )

        # Reconstruct message list
        compressed = system_messages + [summary_message] + recent_messages

        return compressed, summary

    async def asummarize(self, messages: list[BaseMessage]) -> str:
        """Async version of summarize.

        Args:
            messages: List of messages to summarize.

        Returns:
            Summary string.
        """
        if not messages:
            return ""

        conversation_text = self.format_messages_for_summary(messages)

        try:
            chain = self._prompt | self.llm
            result = await chain.ainvoke({"conversation": conversation_text})
            return result.content if hasattr(result, "content") else str(result)
        except Exception as e:
            print(f"Warning: Async summarization failed: {e}")
            return self._create_fallback_summary(messages)


# Global summarizer instance
_summarizer: ConversationSummarizer | None = None


def get_summarizer() -> ConversationSummarizer:
    """Get or create the global summarizer instance."""
    global _summarizer
    if _summarizer is None:
        _summarizer = ConversationSummarizer()
    return _summarizer


def reset_summarizer() -> None:
    """Reset the global summarizer instance."""
    global _summarizer
    _summarizer = None
