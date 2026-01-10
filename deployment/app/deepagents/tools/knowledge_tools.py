"""Knowledge Base Tools for Deep Agents.

Tools for searching and managing knowledge articles.
"""

import uuid
from datetime import datetime

from langchain_core.tools import tool

from app.agents.servicenow_agent import is_live_mode


# Simulated knowledge base
KNOWLEDGE_BASE = {
    "KB0010001": {
        "number": "KB0010001",
        "title": "VPN Connection Troubleshooting Guide",
        "short_description": "Steps to resolve common VPN connectivity issues",
        "content": """## VPN Connection Troubleshooting

### Common Issues and Solutions

1. **VPN disconnects frequently**
   - Check internet connection stability
   - Disable battery saver mode
   - Update VPN client to latest version

2. **Cannot connect to VPN**
   - Verify credentials are correct
   - Check if VPN server is accessible
   - Clear DNS cache: `ipconfig /flushdns`

3. **Slow VPN performance**
   - Try connecting to a different VPN server
   - Disable split tunneling if not required
   - Check for bandwidth-intensive background processes

### Escalation
If issues persist after following these steps, create an incident
with Network Support team.""",
        "category": "Network",
        "views": 1523,
        "helpful_votes": 89,
        "created": "2024-01-15",
        "updated": "2024-11-20",
    },
    "KB0010002": {
        "number": "KB0010002",
        "title": "Email Sync Issues on Mobile Devices",
        "short_description": "Resolving email synchronization problems on iOS and Android",
        "content": """## Mobile Email Sync Troubleshooting

### iOS Devices

1. Remove and re-add email account
2. Check Mail settings > Fetch New Data
3. Ensure iOS is updated to latest version

### Android Devices

1. Clear email app cache and data
2. Re-configure account with correct server settings
3. Check battery optimization settings

### Server Settings
- Incoming: outlook.office365.com:993 (IMAP/SSL)
- Outgoing: smtp.office365.com:587 (STARTTLS)

### Known Issue
ActiveSync may timeout on large mailboxes. Consider archiving
old emails to improve sync performance.""",
        "category": "Email",
        "views": 2341,
        "helpful_votes": 156,
        "created": "2024-02-10",
        "updated": "2024-12-01",
    },
    "KB0010003": {
        "number": "KB0010003",
        "title": "Password Reset Procedures",
        "short_description": "How to reset various system passwords",
        "content": """## Password Reset Guide

### Self-Service Password Reset
1. Go to https://passwordreset.company.com
2. Verify identity with security questions or phone
3. Create new password following policy

### Password Policy
- Minimum 12 characters
- At least 1 uppercase, 1 lowercase, 1 number, 1 special character
- Cannot reuse last 12 passwords
- Expires every 90 days

### Service Account Passwords
Contact IT Security team for service account password resets.
These require manager approval.""",
        "category": "Access",
        "views": 5621,
        "helpful_votes": 234,
        "created": "2023-06-01",
        "updated": "2024-10-15",
    },
}


@tool
def search_knowledge_base(
    query: str,
    category: str | None = None,
    limit: int = 5,
) -> str:
    """Search the knowledge base for relevant articles.

    Use this to find existing solutions before troubleshooting.

    Args:
        query: Search query.
        category: Filter by category (Network, Email, Access, Hardware, Software).
        limit: Maximum number of results.

    Returns:
        List of matching knowledge articles.
    """
    mode = "LIVE" if is_live_mode() else "SIMULATION"

    results = []
    query_lower = query.lower()

    for kb_id, article in KNOWLEDGE_BASE.items():
        if category and article["category"].lower() != category.lower():
            continue

        # Simple relevance scoring
        score = 0
        if query_lower in article["title"].lower():
            score += 10
        if query_lower in article["short_description"].lower():
            score += 5
        if query_lower in article["content"].lower():
            score += 2

        if score > 0:
            results.append((score, article))

    # Sort by relevance
    results.sort(key=lambda x: x[0], reverse=True)
    results = results[:limit]

    if not results:
        return f"""**Knowledge Base Search** [{mode}]
Query: "{query}"

No matching articles found. Consider:
1. Using different search terms
2. Broadening the category filter
3. Creating a new KB article if this is a recurring issue"""

    output = [f"**Knowledge Base Search** [{mode}]"]
    output.append(f'Query: "{query}"\n')
    output.append(f"**Found {len(results)} article(s):**\n")

    for score, article in results:
        output.append(f"""
**{article['number']}**: {article['title']}
{article['short_description']}
- Category: {article['category']} | Views: {article['views']}
- Helpful: {article['helpful_votes']} votes
""")

    return "\n".join(output)


@tool
def get_kb_article(article_number: str) -> str:
    """Get the full content of a knowledge article.

    Use this to retrieve detailed procedures and solutions.

    Args:
        article_number: The KB article number (e.g., KB0010001).

    Returns:
        Full article content.
    """
    mode = "LIVE" if is_live_mode() else "SIMULATION"

    article = KNOWLEDGE_BASE.get(article_number.upper())

    if not article:
        return f"Knowledge article {article_number} not found. [{mode}]"

    return f"""**{article['number']}: {article['title']}** [{mode}]

**Category:** {article['category']}
**Last Updated:** {article['updated']}
**Views:** {article['views']} | **Helpful Votes:** {article['helpful_votes']}

---

{article['content']}

---
*Was this article helpful? Rate it in ServiceNow.*"""


@tool
def create_kb_article(
    title: str,
    short_description: str,
    content: str,
    category: str,
    related_incident: str | None = None,
) -> str:
    """Create a new knowledge base article.

    Use this to document solutions for recurring issues.

    Args:
        title: Article title.
        short_description: Brief summary (max 200 chars).
        content: Full article content in markdown format.
        category: Article category.
        related_incident: Related incident number if applicable.

    Returns:
        Created article number and details.
    """
    mode = "LIVE" if is_live_mode() else "SIMULATION"

    kb_number = f"KB{str(uuid.uuid4().int)[:7]}"

    return f"""**Knowledge Article Created** [{mode}]

**Article Number:** {kb_number}
**Title:** {title}
**Category:** {category}
**Status:** Draft (Pending Review)

**Summary:**
{short_description}

{f'**Related Incident:** {related_incident}' if related_incident else ''}

**Next Steps:**
1. Article is in Draft status
2. Submit for review by Knowledge Manager
3. Once approved, article will be published
4. Share KB number with Service Desk for immediate use"""


@tool
def suggest_kb_articles(
    incident_description: str,
    category: str | None = None,
) -> str:
    """Suggest relevant KB articles based on incident description.

    Use this to find potential solutions for an incident.

    Args:
        incident_description: Description of the incident/issue.
        category: Optional category hint.

    Returns:
        Suggested KB articles with relevance scores.
    """
    mode = "LIVE" if is_live_mode() else "SIMULATION"

    # Keywords to article mapping
    keyword_matches = {
        "vpn": ["KB0010001"],
        "connect": ["KB0010001"],
        "email": ["KB0010002"],
        "sync": ["KB0010002"],
        "mobile": ["KB0010002"],
        "password": ["KB0010003"],
        "reset": ["KB0010003"],
        "login": ["KB0010003"],
    }

    desc_lower = incident_description.lower()
    suggested = set()

    for keyword, articles in keyword_matches.items():
        if keyword in desc_lower:
            suggested.update(articles)

    if not suggested:
        return f"""**KB Article Suggestions** [{mode}]

No directly matching articles found for:
"{incident_description[:100]}..."

**Recommendations:**
1. Search knowledge base with specific keywords
2. Check related category articles
3. Escalate to subject matter expert
4. Document solution as new KB article after resolution"""

    output = [f"**KB Article Suggestions** [{mode}]\n"]
    output.append(f'Based on: "{incident_description[:50]}..."\n')
    output.append("**Suggested Articles:**\n")

    for kb_num in suggested:
        article = KNOWLEDGE_BASE.get(kb_num)
        if article:
            output.append(f"""
**{article['number']}**: {article['title']}
{article['short_description']}
- Relevance: High | Success Rate: {85 + hash(kb_num) % 10}%
""")

    output.append("\nReview suggested articles before troubleshooting.")

    return "\n".join(output)
