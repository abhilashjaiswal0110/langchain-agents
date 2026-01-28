"""Documentation Tools.

Tools for generating API documentation, README files, architecture diagrams,
and other technical documentation.
"""

import json
import re
import uuid
from datetime import datetime

from langchain_core.tools import tool
from langsmith import traceable


# Session storage
_doc_store: dict[str, dict] = {}


@tool
@traceable(name="generate_api_docs", tags=["documentation", "api"])
def generate_api_docs(
    endpoints: list[dict],
    format: str = "openapi",
    title: str = "API Documentation",
    version: str = "1.0.0",
) -> str:
    """Generate API documentation.

    Creates documentation in specified format:
    - openapi: OpenAPI/Swagger specification
    - markdown: Markdown documentation
    - html: HTML documentation

    Args:
        endpoints: List of endpoint specifications.
        format: Output format.
        title: API title.
        version: API version.

    Returns:
        JSON string with generated documentation.
    """
    if format == "openapi":
        doc = {
            "openapi": "3.0.3",
            "info": {
                "title": title,
                "version": version,
                "description": f"API documentation for {title}",
            },
            "servers": [
                {"url": "https://api.example.com/v1", "description": "Production"},
                {"url": "http://localhost:8000", "description": "Development"},
            ],
            "paths": {},
            "components": {
                "securitySchemes": {
                    "bearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT",
                    }
                }
            },
        }

        for endpoint in endpoints:
            path = endpoint.get("path", "/")
            method = endpoint.get("method", "get").lower()

            if path not in doc["paths"]:
                doc["paths"][path] = {}

            doc["paths"][path][method] = {
                "summary": endpoint.get("description", ""),
                "operationId": endpoint.get("operation_id", f"{method}_{path.replace('/', '_')}"),
                "tags": endpoint.get("tags", ["default"]),
                "parameters": endpoint.get("parameters", []),
                "responses": {
                    "200": {"description": "Successful response"},
                    "400": {"description": "Bad request"},
                    "401": {"description": "Unauthorized"},
                    "500": {"description": "Internal server error"},
                },
            }

            if endpoint.get("request_body"):
                doc["paths"][path][method]["requestBody"] = {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": endpoint["request_body"]
                        }
                    }
                }

        content = json.dumps(doc, indent=2)

    elif format == "markdown":
        lines = [
            f"# {title}",
            f"\nVersion: {version}",
            "\n## Endpoints\n",
        ]

        for endpoint in endpoints:
            method = endpoint.get("method", "GET").upper()
            path = endpoint.get("path", "/")
            desc = endpoint.get("description", "No description")

            lines.append(f"### {method} {path}")
            lines.append(f"\n{desc}\n")

            if endpoint.get("parameters"):
                lines.append("**Parameters:**\n")
                for param in endpoint["parameters"]:
                    lines.append(f"- `{param.get('name')}` ({param.get('type', 'string')}): {param.get('description', '')}")
                lines.append("")

            if endpoint.get("request_body"):
                lines.append("**Request Body:**\n```json")
                lines.append(json.dumps(endpoint["request_body"], indent=2))
                lines.append("```\n")

            lines.append("**Response:**\n```json")
            lines.append('{"status": "success", "data": {...}}')
            lines.append("```\n")

        content = "\n".join(lines)

    else:
        content = f"<h1>{title}</h1><p>API Documentation v{version}</p>"

    result = {
        "format": format,
        "title": title,
        "version": version,
        "endpoints_documented": len(endpoints),
        "content": content,
        "file_extension": ".yaml" if format == "openapi" else ".md" if format == "markdown" else ".html",
    }

    return json.dumps(result, indent=2)


@tool
@traceable(name="create_readme", tags=["documentation", "readme"])
def create_readme(
    project_name: str,
    description: str,
    features: list[str] | None = None,
    installation_steps: list[str] | None = None,
    usage_examples: list[str] | None = None,
    tech_stack: list[str] | None = None,
) -> str:
    """Create a README.md file for the project.

    Generates a comprehensive README with:
    - Project description
    - Features list
    - Installation guide
    - Usage examples
    - Tech stack
    - Contributing guidelines

    Args:
        project_name: Name of the project.
        description: Project description.
        features: Key features list.
        installation_steps: Installation instructions.
        usage_examples: Usage examples.
        tech_stack: Technologies used.

    Returns:
        JSON string with README content.
    """
    features = features or ["Feature 1", "Feature 2", "Feature 3"]
    installation_steps = installation_steps or [
        "Clone the repository",
        "Install dependencies: `pip install -r requirements.txt`",
        "Copy `.env.example` to `.env` and configure",
        "Run the application: `python main.py`",
    ]
    usage_examples = usage_examples or ["Example usage coming soon"]
    tech_stack = tech_stack or ["Python 3.11+", "FastAPI", "PostgreSQL"]

    readme = f'''# {project_name}

{description}

## Features

{chr(10).join(f"- {f}" for f in features)}

## Tech Stack

{chr(10).join(f"- {t}" for t in tech_stack)}

## Installation

{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(installation_steps))}

## Usage

```python
{chr(10).join(usage_examples)}
```

## Configuration

Copy `.env.example` to `.env` and set the following variables:

```bash
# Required
API_KEY=your-api-key

# Optional
DEBUG=false
LOG_LEVEL=info
```

## API Documentation

API documentation is available at `/docs` when running the server.

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linting
ruff check .
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue on GitHub.
'''

    result = {
        "project_name": project_name,
        "file_name": "README.md",
        "content": readme,
        "sections": [
            "Features",
            "Tech Stack",
            "Installation",
            "Usage",
            "Configuration",
            "API Documentation",
            "Development",
            "Contributing",
            "License",
            "Support",
        ],
    }

    return json.dumps(result, indent=2)


@tool
@traceable(name="document_architecture", tags=["documentation", "architecture"])
def document_architecture(
    components: list[dict],
    description: str | None = None,
    format: str = "markdown",
) -> str:
    """Generate architecture documentation.

    Creates documentation including:
    - System overview
    - Component descriptions
    - Data flow diagrams
    - Integration points

    Args:
        components: List of architecture components.
        description: Overall system description.
        format: Output format (markdown, mermaid).

    Returns:
        JSON string with architecture documentation.
    """
    description = description or "System architecture overview"

    if format == "markdown":
        doc = f'''# Architecture Documentation

## Overview

{description}

## Components

'''
        for comp in components:
            doc += f'''### {comp.get("name", "Component")}

**Type:** {comp.get("type", "service")}

**Description:** {comp.get("description", "No description")}

**Technologies:** {", ".join(comp.get("technologies", []))}

**Dependencies:** {", ".join(comp.get("dependencies", [])) or "None"}

---

'''

        # Add Mermaid diagram
        doc += '''## System Diagram

```mermaid
graph TB
'''
        for comp in components:
            name = comp.get("name", "Component").replace(" ", "_")
            doc += f"    {name}[{comp.get('name', 'Component')}]\n"

        for comp in components:
            name = comp.get("name", "").replace(" ", "_")
            for dep in comp.get("dependencies", []):
                dep_name = dep.replace(" ", "_")
                doc += f"    {name} --> {dep_name}\n"

        doc += "```\n"

    else:  # mermaid only
        doc = "graph TB\n"
        for comp in components:
            name = comp.get("name", "Component").replace(" ", "_")
            comp_type = comp.get("type", "component")
            if comp_type == "database":
                doc += f"    {name}[({comp['name']})]\n"
            elif comp_type == "queue":
                doc += f"    {name}[/{comp['name']}/]\n"
            else:
                doc += f"    {name}[{comp['name']}]\n"

        for comp in components:
            name = comp.get("name", "").replace(" ", "_")
            for dep in comp.get("dependencies", []):
                dep_name = dep.replace(" ", "_")
                doc += f"    {name} --> {dep_name}\n"

    result = {
        "format": format,
        "components_documented": len(components),
        "content": doc,
        "file_name": "ARCHITECTURE.md" if format == "markdown" else "architecture.mmd",
    }

    return json.dumps(result, indent=2)


@tool
@traceable(name="generate_changelog", tags=["documentation", "changelog"])
def generate_changelog(
    version: str,
    changes: dict[str, list[str]],
    date: str | None = None,
) -> str:
    """Generate changelog entry.

    Creates changelog following Keep a Changelog format:
    - Added
    - Changed
    - Deprecated
    - Removed
    - Fixed
    - Security

    Args:
        version: Version number.
        changes: Dictionary of change types to lists of changes.
        date: Release date (defaults to today).

    Returns:
        JSON string with changelog content.
    """
    date = date or datetime.now().strftime("%Y-%m-%d")

    changelog = f'''# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [{version}] - {date}

'''
    sections = ["Added", "Changed", "Deprecated", "Removed", "Fixed", "Security"]

    for section in sections:
        section_changes = changes.get(section.lower(), [])
        if section_changes:
            changelog += f"### {section}\n\n"
            for change in section_changes:
                changelog += f"- {change}\n"
            changelog += "\n"

    result = {
        "version": version,
        "date": date,
        "content": changelog,
        "file_name": "CHANGELOG.md",
        "changes_count": sum(len(v) for v in changes.values()),
    }

    return json.dumps(result, indent=2)


@tool
@traceable(name="add_inline_comments", tags=["documentation", "code"])
def add_inline_comments(
    code: str,
    language: str = "python",
    style: str = "concise",
) -> str:
    """Add inline comments to code.

    Adds explanatory comments to:
    - Complex logic
    - Function definitions
    - Important variables
    - Non-obvious code

    Args:
        code: Code to comment.
        language: Programming language.
        style: Comment style (concise, detailed).

    Returns:
        JSON string with commented code.
    """
    lines = code.split("\n")
    commented_lines = []
    comments_added = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        indent = line[:len(line) - len(line.lstrip())]

        # Add comments based on code patterns
        if stripped.startswith("def ") and '"""' not in lines[i+1] if i+1 < len(lines) else True:
            # Function definition without docstring
            func_name = stripped.split("(")[0].replace("def ", "")
            comment = f"{indent}# {func_name}: Handles {func_name.replace('_', ' ')}"
            commented_lines.append(comment)
            comments_added += 1

        elif stripped.startswith("class "):
            class_name = stripped.split("(")[0].split(":")[0].replace("class ", "")
            comment = f"{indent}# {class_name} class definition"
            commented_lines.append(comment)
            comments_added += 1

        elif stripped.startswith("if ") or stripped.startswith("elif "):
            if style == "detailed" and len(stripped) > 30:
                comment = f"{indent}# Condition: check following criteria"
                commented_lines.append(comment)
                comments_added += 1

        elif stripped.startswith("for ") or stripped.startswith("while "):
            comment = f"{indent}# Loop iteration"
            commented_lines.append(comment)
            comments_added += 1

        elif stripped.startswith("try:"):
            comment = f"{indent}# Error handling block"
            commented_lines.append(comment)
            comments_added += 1

        elif stripped.startswith("except"):
            comment = f"{indent}# Handle exception"
            commented_lines.append(comment)
            comments_added += 1

        elif "= lambda" in stripped:
            comment = f"{indent}# Anonymous function"
            commented_lines.append(comment)
            comments_added += 1

        # Add the original line
        commented_lines.append(line)

    commented_code = "\n".join(commented_lines)

    result = {
        "language": language,
        "style": style,
        "original_lines": len(lines),
        "comments_added": comments_added,
        "commented_code": commented_code,
    }

    return json.dumps(result, indent=2)


@tool
@traceable(name="create_user_guide", tags=["documentation", "user"])
def create_user_guide(
    product_name: str,
    features: list[dict],
    getting_started: str | None = None,
) -> str:
    """Create user guide documentation.

    Generates user-friendly documentation with:
    - Getting started guide
    - Feature explanations
    - Step-by-step tutorials
    - FAQ section

    Args:
        product_name: Name of the product.
        features: List of features with descriptions.
        getting_started: Getting started content.

    Returns:
        JSON string with user guide content.
    """
    getting_started = getting_started or f"Welcome to {product_name}! This guide will help you get started."

    guide = f'''# {product_name} User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Features](#features)
3. [Tutorials](#tutorials)
4. [FAQ](#faq)
5. [Troubleshooting](#troubleshooting)

## Getting Started

{getting_started}

### Prerequisites

- A modern web browser
- Account credentials

### Quick Start

1. Log in to your account
2. Navigate to the dashboard
3. Start using the features below

## Features

'''

    for feature in features:
        guide += f'''### {feature.get("name", "Feature")}

{feature.get("description", "Description coming soon.")}

**How to use:**

{feature.get("usage", "1. Navigate to the feature\\n2. Follow the prompts")}

---

'''

    guide += '''## Tutorials

### Tutorial 1: Basic Workflow

Step-by-step guide for the most common workflow.

### Tutorial 2: Advanced Features

Learn about advanced capabilities.

## FAQ

**Q: How do I reset my password?**
A: Click "Forgot Password" on the login page.

**Q: Where can I get help?**
A: Contact support or check our documentation.

## Troubleshooting

### Common Issues

1. **Login problems**: Clear browser cache and try again
2. **Slow performance**: Check your internet connection
3. **Missing features**: Ensure you have the correct permissions

### Contact Support

If you need additional help:
- Email: support@example.com
- Phone: 1-800-EXAMPLE
'''

    result = {
        "product_name": product_name,
        "features_documented": len(features),
        "content": guide,
        "file_name": "USER_GUIDE.md",
        "sections": [
            "Getting Started",
            "Features",
            "Tutorials",
            "FAQ",
            "Troubleshooting",
        ],
    }

    return json.dumps(result, indent=2)
