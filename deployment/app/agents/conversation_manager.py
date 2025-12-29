"""Conversation Manager for multi-agent session handling.

Provides unified interface for managing conversations across different agents
with session persistence, history tracking, and integration support.
"""

import uuid
from datetime import datetime
from typing import Any, Literal
from langsmith import traceable


# =============================================================================
# Session Storage (In-memory for demo, use Redis/DB for production)
# =============================================================================

class SessionStore:
    """In-memory session storage."""

    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}

    def create_session(
        self,
        agent_type: str,
        user_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Create a new session."""
        session_id = str(uuid.uuid4())

        self.sessions[session_id] = {
            "id": session_id,
            "agent_type": agent_type,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "messages": [],
            "context": {},
        }

        return session_id

    def get_session(self, session_id: str) -> dict | None:
        """Get session by ID."""
        return self.sessions.get(session_id)

    def update_session(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: dict | None = None,
    ) -> None:
        """Update session with new messages."""
        session = self.sessions.get(session_id)
        if not session:
            return

        session["messages"].append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat(),
        })
        session["messages"].append({
            "role": "assistant",
            "content": assistant_message,
            "timestamp": datetime.now().isoformat(),
        })
        session["updated_at"] = datetime.now().isoformat()

        if metadata:
            session["metadata"].update(metadata)

    def get_history(self, session_id: str, limit: int = 50) -> list[dict]:
        """Get conversation history."""
        session = self.sessions.get(session_id)
        if not session:
            return []
        return session["messages"][-limit:]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def list_sessions(
        self,
        user_id: str | None = None,
        agent_type: str | None = None,
    ) -> list[dict]:
        """List sessions with optional filters."""
        results = []
        for session in self.sessions.values():
            if user_id and session.get("user_id") != user_id:
                continue
            if agent_type and session.get("agent_type") != agent_type:
                continue
            results.append({
                "id": session["id"],
                "agent_type": session["agent_type"],
                "user_id": session["user_id"],
                "created_at": session["created_at"],
                "updated_at": session["updated_at"],
                "message_count": len(session["messages"]),
            })
        return results


# =============================================================================
# Conversation Manager
# =============================================================================

class ConversationManager:
    """Unified conversation manager for all IT Support agents."""

    AVAILABLE_AGENTS = {
        "it_helpdesk": "IT Helpdesk Agent - General IT support, password resets, troubleshooting",
        "servicenow": "ServiceNow Agent - Ticket management, change requests, CMDB",
    }

    def __init__(self) -> None:
        """Initialize conversation manager."""
        self.session_store = SessionStore()
        self._agents: dict[str, Any] = {}
        self._load_agents()

    def _load_agents(self) -> None:
        """Lazy load agents."""
        try:
            from app.agents.it_helpdesk import ITHelpdeskAgent
            self._agents["it_helpdesk"] = ITHelpdeskAgent(model_provider="auto")
        except Exception as e:
            print(f"Failed to load IT Helpdesk Agent: {e}")

        try:
            from app.agents.servicenow_agent import ServiceNowAgent
            self._agents["servicenow"] = ServiceNowAgent(model_provider="auto")
        except Exception as e:
            print(f"Failed to load ServiceNow Agent: {e}")

    def get_available_agents(self) -> dict[str, str]:
        """Get list of available agents."""
        return {k: v for k, v in self.AVAILABLE_AGENTS.items() if k in self._agents}

    @traceable(name="conversation_start", tags=["conversation", "session"])
    def start_conversation(
        self,
        agent_type: Literal["it_helpdesk", "servicenow"],
        user_id: str | None = None,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        """Start a new conversation with an agent.

        Args:
            agent_type: Type of agent to use.
            user_id: Optional user identifier.
            metadata: Optional metadata for the session.

        Returns:
            Session information including ID and welcome message.
        """
        if agent_type not in self._agents:
            return {
                "error": f"Agent '{agent_type}' not available",
                "available_agents": list(self.get_available_agents().keys()),
            }

        session_id = self.session_store.create_session(
            agent_type=agent_type,
            user_id=user_id,
            metadata=metadata,
        )

        # Welcome messages
        welcome_messages = {
            "it_helpdesk": """Welcome to IT Support! I'm your IT Helpdesk Assistant.

I can help you with:
- Password resets and account issues
- Software installation and troubleshooting
- Network and VPN connectivity
- Hardware problems
- Creating support tickets
- Checking system status

How can I assist you today?""",
            "servicenow": """Welcome to ServiceNow Support! I'm your ITSM Assistant.

I can help you with:
- Searching and creating incidents
- Checking ticket status
- Viewing upcoming changes
- Searching the CMDB
- Tracking your tickets

What would you like to do?""",
        }

        return {
            "session_id": session_id,
            "agent_type": agent_type,
            "welcome_message": welcome_messages.get(agent_type, "How can I help you?"),
            "available_commands": [
                "/history - View conversation history",
                "/clear - Clear conversation",
                "/switch <agent> - Switch to different agent",
                "/status - Check system status",
                "/help - Show help",
            ],
        }

    @traceable(name="conversation_chat", tags=["conversation", "chat"])
    def chat(
        self,
        session_id: str,
        message: str,
    ) -> dict[str, Any]:
        """Send a message in an existing conversation.

        Args:
            session_id: The session ID.
            message: User's message.

        Returns:
            Agent's response and metadata.
        """
        session = self.session_store.get_session(session_id)
        if not session:
            return {
                "error": "Session not found. Please start a new conversation.",
                "session_id": None,
            }

        agent_type = session["agent_type"]
        agent = self._agents.get(agent_type)

        if not agent:
            return {
                "error": f"Agent '{agent_type}' not available.",
                "session_id": session_id,
            }

        # Handle special commands
        if message.startswith("/"):
            return self._handle_command(session_id, message)

        # Chat with agent
        try:
            result = agent.chat(message, thread_id=session_id)

            # Update session
            self.session_store.update_session(
                session_id=session_id,
                user_message=message,
                assistant_message=result["response"],
            )

            return {
                "session_id": session_id,
                "response": result["response"],
                "agent_type": agent_type,
                "tool_calls": result.get("tool_calls", []),
            }

        except Exception as e:
            return {
                "error": f"Error processing message: {str(e)}",
                "session_id": session_id,
            }

    async def achat(
        self,
        session_id: str,
        message: str,
    ) -> dict[str, Any]:
        """Async version of chat."""
        session = self.session_store.get_session(session_id)
        if not session:
            return {
                "error": "Session not found.",
                "session_id": None,
            }

        agent_type = session["agent_type"]
        agent = self._agents.get(agent_type)

        if not agent:
            return {
                "error": f"Agent '{agent_type}' not available.",
                "session_id": session_id,
            }

        if message.startswith("/"):
            return self._handle_command(session_id, message)

        try:
            result = await agent.achat(message, thread_id=session_id)

            self.session_store.update_session(
                session_id=session_id,
                user_message=message,
                assistant_message=result["response"],
            )

            return {
                "session_id": session_id,
                "response": result["response"],
                "agent_type": agent_type,
                "tool_calls": result.get("tool_calls", []),
            }

        except Exception as e:
            return {
                "error": f"Error: {str(e)}",
                "session_id": session_id,
            }

    def _handle_command(self, session_id: str, command: str) -> dict[str, Any]:
        """Handle special commands."""
        cmd = command.lower().strip()
        session = self.session_store.get_session(session_id)

        if cmd == "/history":
            history = self.session_store.get_history(session_id)
            if not history:
                return {
                    "session_id": session_id,
                    "response": "No conversation history yet.",
                    "is_command": True,
                }
            formatted = []
            for msg in history[-10:]:  # Last 10 messages
                role = "You" if msg["role"] == "user" else "Agent"
                formatted.append(f"**{role}:** {msg['content'][:100]}...")
            return {
                "session_id": session_id,
                "response": "**Recent History:**\n\n" + "\n\n".join(formatted),
                "is_command": True,
            }

        elif cmd == "/clear":
            if session:
                session["messages"] = []
            return {
                "session_id": session_id,
                "response": "Conversation cleared. How can I help you?",
                "is_command": True,
            }

        elif cmd == "/status":
            # Import and call system status
            try:
                from app.agents.it_helpdesk import check_system_status
                status = check_system_status.invoke({})
                return {
                    "session_id": session_id,
                    "response": status,
                    "is_command": True,
                }
            except Exception:
                return {
                    "session_id": session_id,
                    "response": "System status check unavailable.",
                    "is_command": True,
                }

        elif cmd.startswith("/switch"):
            parts = cmd.split()
            if len(parts) < 2:
                agents = ", ".join(self._agents.keys())
                return {
                    "session_id": session_id,
                    "response": f"Usage: /switch <agent>\nAvailable agents: {agents}",
                    "is_command": True,
                }
            new_agent = parts[1]
            if new_agent in self._agents:
                if session:
                    session["agent_type"] = new_agent
                return {
                    "session_id": session_id,
                    "response": f"Switched to {new_agent} agent. How can I help?",
                    "is_command": True,
                    "agent_type": new_agent,
                }
            return {
                "session_id": session_id,
                "response": f"Agent '{new_agent}' not found.",
                "is_command": True,
            }

        elif cmd == "/help":
            return {
                "session_id": session_id,
                "response": """**Available Commands:**

**/history** - View recent conversation history
**/clear** - Clear current conversation
**/switch <agent>** - Switch to a different agent
  - it_helpdesk: General IT support
  - servicenow: Ticket management
**/status** - Check system status
**/help** - Show this help message

Just type your question to chat with the current agent.""",
                "is_command": True,
            }

        return {
            "session_id": session_id,
            "response": f"Unknown command: {command}. Type /help for available commands.",
            "is_command": True,
        }

    def get_session_info(self, session_id: str) -> dict | None:
        """Get session information."""
        session = self.session_store.get_session(session_id)
        if not session:
            return None

        return {
            "id": session["id"],
            "agent_type": session["agent_type"],
            "user_id": session["user_id"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "message_count": len(session["messages"]),
        }

    def end_conversation(self, session_id: str) -> dict[str, Any]:
        """End a conversation and get summary."""
        session = self.session_store.get_session(session_id)
        if not session:
            return {"error": "Session not found"}

        summary = {
            "session_id": session_id,
            "agent_type": session["agent_type"],
            "duration": session["updated_at"],
            "message_count": len(session["messages"]),
            "status": "ended",
        }

        # Optionally keep session for history
        # self.session_store.delete_session(session_id)

        return summary


# Global instance
conversation_manager = ConversationManager()
