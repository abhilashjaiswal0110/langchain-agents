"""Conversation export utilities.

Provides export of conversation sessions to JSON, plain text, and PDF formats.
"""

import json
from typing import Any

from app.memory.base import Session


class ConversationExporter:
    """Export conversation sessions to various formats.

    Supports JSON (structured data), plain text (readable transcript),
    and PDF (printable document) output formats.
    """

    def to_json(self, session: Session) -> str:
        """Export session as a JSON string.

        Args:
            session: The session to export.

        Returns:
            Pretty-printed JSON string with session data.
        """
        data: dict[str, Any] = {
            "session_id": session.id,
            "agent_type": session.metadata.agent_type,
            "user_id": session.metadata.user_id,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                }
                for m in session.messages
            ],
        }
        return json.dumps(data, indent=2)

    def to_text(self, session: Session) -> str:
        """Export session as a plain-text transcript.

        Args:
            session: The session to export.

        Returns:
            Human-readable conversation transcript.
        """
        lines = [
            f"Conversation: {session.id}",
            f"Agent: {session.metadata.agent_type}",
            f"Started: {session.created_at.isoformat()}",
            "",
        ]
        for m in session.messages:
            lines.append(f"[{m.role.upper()}] {m.content}")
            lines.append("")
        return "\n".join(lines)

    def to_pdf(self, session: Session) -> bytes:
        """Export session as a PDF document.

        Args:
            session: The session to export.

        Returns:
            PDF file content as bytes.

        Raises:
            ImportError: If fpdf2 is not installed.
        """
        try:
            from fpdf import FPDF  # type: ignore[import-untyped]
        except ImportError as e:
            msg = "fpdf2 is required for PDF export. Install it with: pip install fpdf2"
            raise ImportError(msg) from e

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", style="B", size=14)
        pdf.cell(0, 10, f"Conversation: {session.id}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)
        pdf.cell(0, 8, f"Agent: {session.metadata.agent_type}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Started: {session.created_at.isoformat()}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        pdf.set_font("Helvetica", size=11)
        for m in session.messages:
            label = f"{m.role.upper()}:"
            pdf.set_font("Helvetica", style="B", size=11)
            pdf.cell(0, 8, label, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0, 7, m.content)
            pdf.ln(2)

        return bytes(pdf.output())
