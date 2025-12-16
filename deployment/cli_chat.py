#!/usr/bin/env python
"""CLI Chat Interface for IT Support Agents.

Usage:
    python cli_chat.py [--agent it_helpdesk|servicenow]

Interactive terminal chat with IT Support agents.
"""

import argparse
import sys
from typing import Literal

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.theme import Theme

# Load environment variables
load_dotenv()

# Rich console with custom theme
custom_theme = Theme({
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red bold",
    "user": "blue",
    "assistant": "green",
})

console = Console(theme=custom_theme)


def print_banner() -> None:
    """Print welcome banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║              IT Support Agent - CLI Interface                 ║
║                                                               ║
║  Available Agents:                                            ║
║  • it_helpdesk - General IT support                          ║
║  • servicenow  - ServiceNow ITSM operations                  ║
║                                                               ║
║  Commands: /help, /switch, /status, /history, /clear, /quit  ║
╚═══════════════════════════════════════════════════════════════╝
"""
    console.print(Panel(banner, style="cyan"))


def print_help() -> None:
    """Print help information."""
    help_text = """
## Available Commands

| Command | Description |
|---------|-------------|
| `/help` | Show this help message |
| `/switch <agent>` | Switch to a different agent |
| `/status` | Check IT system status |
| `/history` | View conversation history |
| `/clear` | Clear conversation |
| `/quit` or `/exit` | Exit the chat |

## Example Questions

**IT Helpdesk:**
- "I need to reset my password"
- "My VPN is not connecting"
- "How do I install software?"
- "Create a ticket for my broken laptop"

**ServiceNow:**
- "Search for my open tickets"
- "Show incident INC0010001"
- "Create a new incident for email issues"
- "What changes are scheduled this week?"
"""
    console.print(Markdown(help_text))


def select_agent() -> str:
    """Prompt user to select an agent."""
    console.print("\n[info]Available Agents:[/info]")
    console.print("  1. [cyan]it_helpdesk[/cyan] - General IT support, password resets, troubleshooting")
    console.print("  2. [cyan]servicenow[/cyan]  - Ticket management, CMDB, change requests")
    console.print()

    while True:
        choice = Prompt.ask(
            "[info]Select agent[/info]",
            choices=["1", "2", "it_helpdesk", "servicenow"],
            default="1",
        )

        if choice in ["1", "it_helpdesk"]:
            return "it_helpdesk"
        elif choice in ["2", "servicenow"]:
            return "servicenow"


def main(agent_type: str | None = None) -> None:
    """Main chat loop."""
    print_banner()

    # Select agent if not specified
    if not agent_type:
        agent_type = select_agent()

    # Import and initialize conversation manager
    try:
        from app.agents.conversation_manager import ConversationManager
        manager = ConversationManager()
    except ImportError as e:
        console.print(f"[error]Error importing agents: {e}[/error]")
        console.print("[info]Make sure you're running from the deployment directory.[/info]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[error]Error initializing agents: {e}[/error]")
        console.print("[info]Make sure OPENAI_API_KEY or ANTHROPIC_API_KEY is set.[/info]")
        sys.exit(1)

    # Start session
    console.print(f"\n[info]Starting session with [cyan]{agent_type}[/cyan] agent...[/info]")

    result = manager.start_conversation(agent_type=agent_type)

    if "error" in result:
        console.print(f"[error]Error: {result['error']}[/error]")
        sys.exit(1)

    session_id = result["session_id"]
    console.print(f"[success]Session started: {session_id[:8]}...[/success]\n")

    # Print welcome message
    console.print(Panel(
        Markdown(result["welcome_message"]),
        title=f"[assistant]{agent_type.upper()} Agent[/assistant]",
        border_style="green",
    ))

    # Chat loop
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n[user]You[/user]")

            if not user_input.strip():
                continue

            # Handle commands
            cmd = user_input.lower().strip()

            if cmd in ["/quit", "/exit", "/q"]:
                console.print("\n[info]Goodbye! Session ended.[/info]")
                break

            if cmd == "/help":
                print_help()
                continue

            if cmd.startswith("/switch"):
                parts = cmd.split()
                if len(parts) < 2:
                    console.print("[warning]Usage: /switch <agent> (it_helpdesk or servicenow)[/warning]")
                    continue

                new_agent = parts[1]
                if new_agent not in ["it_helpdesk", "servicenow"]:
                    console.print(f"[warning]Unknown agent: {new_agent}[/warning]")
                    continue

                # Start new session with different agent
                result = manager.start_conversation(agent_type=new_agent)
                if "error" in result:
                    console.print(f"[error]{result['error']}[/error]")
                    continue

                session_id = result["session_id"]
                agent_type = new_agent
                console.print(f"\n[success]Switched to {new_agent} agent[/success]")
                console.print(Panel(
                    Markdown(result["welcome_message"]),
                    title=f"[assistant]{agent_type.upper()} Agent[/assistant]",
                    border_style="green",
                ))
                continue

            # Send message to agent
            with console.status("[info]Agent is thinking...[/info]", spinner="dots"):
                response = manager.chat(session_id, user_input)

            if "error" in response:
                console.print(f"[error]Error: {response['error']}[/error]")
                continue

            # Display response
            console.print()
            console.print(Panel(
                Markdown(response["response"]),
                title=f"[assistant]{agent_type.upper()} Agent[/assistant]",
                border_style="green",
            ))

        except KeyboardInterrupt:
            console.print("\n\n[info]Session interrupted. Type /quit to exit.[/info]")
        except Exception as e:
            console.print(f"[error]Error: {e}[/error]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IT Support Agent CLI")
    parser.add_argument(
        "--agent",
        "-a",
        choices=["it_helpdesk", "servicenow"],
        help="Agent type to use",
    )
    args = parser.parse_args()

    main(args.agent)
