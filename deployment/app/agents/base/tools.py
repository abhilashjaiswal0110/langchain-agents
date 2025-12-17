"""Shared tool utilities for enterprise IT agents.

This module provides common utilities for creating and managing
tools across all agents.

Following Enterprise Development Standards:
- Input validation with Pydantic
- Consistent error handling
- Secure parameter processing
"""

import functools
import logging
from typing import Any, Callable, TypeVar

from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, ValidationError


logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def tool_error_handler(func: F) -> F:
    """Decorator to handle tool errors gracefully.

    Catches exceptions and returns formatted error messages
    that the LLM can understand and potentially recover from.

    Args:
        func: Tool function to wrap

    Returns:
        Wrapped function with error handling
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            error_msg = f"Input validation error: {e}"
            logger.warning(f"Tool {func.__name__} validation error: {e}")
            return error_msg
        except PermissionError as e:
            error_msg = f"Permission denied: {e}"
            logger.warning(f"Tool {func.__name__} permission error: {e}")
            return error_msg
        except FileNotFoundError as e:
            error_msg = f"File not found: {e}"
            logger.warning(f"Tool {func.__name__} file error: {e}")
            return error_msg
        except Exception as e:
            error_msg = f"Tool execution error: {type(e).__name__}: {e}"
            logger.error(f"Tool {func.__name__} error: {e}", exc_info=True)
            return error_msg

    return wrapper  # type: ignore


def validate_input(schema: type[BaseModel]) -> Callable[[F], F]:
    """Decorator to validate tool inputs against a Pydantic schema.

    Args:
        schema: Pydantic model class for validation

    Returns:
        Decorator function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Validate kwargs against schema
            try:
                validated = schema(**kwargs)
                return func(*args, **validated.model_dump())
            except ValidationError as e:
                raise ValueError(f"Invalid input: {e}")

        return wrapper  # type: ignore

    return decorator


def create_tool(
    name: str,
    description: str,
    func: Callable[..., str],
    args_schema: type[BaseModel] | None = None,
) -> StructuredTool:
    """Create a structured tool with consistent configuration.

    Args:
        name: Tool name (should be snake_case)
        description: Clear description of what the tool does
        func: The tool function
        args_schema: Optional Pydantic schema for arguments

    Returns:
        Configured StructuredTool instance
    """
    # Wrap with error handler
    wrapped_func = tool_error_handler(func)

    return StructuredTool.from_function(
        func=wrapped_func,
        name=name,
        description=description,
        args_schema=args_schema,
    )


# Common input schemas for reusable validation


class TextInput(BaseModel):
    """Schema for simple text input."""

    text: str


class QueryInput(BaseModel):
    """Schema for search/query input."""

    query: str
    max_results: int = 10


class FileInput(BaseModel):
    """Schema for file path input."""

    file_path: str


class CodeInput(BaseModel):
    """Schema for code analysis input."""

    code: str
    language: str = "python"


# Utility functions for common operations


def sanitize_output(text: str, max_length: int = 10000) -> str:
    """Sanitize tool output for safe display.

    Args:
        text: Raw output text
        max_length: Maximum length to return

    Returns:
        Sanitized and truncated text
    """
    if not text:
        return ""

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "\n... (truncated)"

    return text


def format_tool_result(
    success: bool,
    data: Any = None,
    error: str | None = None,
) -> str:
    """Format tool results consistently.

    Args:
        success: Whether the operation succeeded
        data: Result data (if successful)
        error: Error message (if failed)

    Returns:
        Formatted result string
    """
    if success:
        if isinstance(data, dict):
            lines = [f"- {k}: {v}" for k, v in data.items()]
            return "Success:\n" + "\n".join(lines)
        return f"Success: {data}"
    else:
        return f"Error: {error or 'Unknown error'}"


def chunk_text(text: str, chunk_size: int = 4000) -> list[str]:
    """Split text into chunks for processing.

    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk

    Returns:
        List of text chunks
    """
    chunks = []
    current_chunk = ""

    for paragraph in text.split("\n\n"):
        if len(current_chunk) + len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk += "\n\n" + paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
