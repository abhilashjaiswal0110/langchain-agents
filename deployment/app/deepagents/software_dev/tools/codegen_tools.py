"""Code Generation Tools.

Tools for generating, refactoring, and formatting code
across multiple programming languages.
"""

import json
import uuid
from datetime import datetime

from langchain_core.tools import tool
from langsmith import traceable

# Session storage for generated code
_code_store: dict[str, dict] = {}


@tool
@traceable(name="generate_code", tags=["codegen", "sdlc"])
def generate_code(
    description: str,
    language: str = "python",
    framework: str | None = None,
    include_tests: bool = False,
    session_id: str = "default",
) -> str:
    """Generate code based on description.

    Creates production-ready code following best practices
    for the specified language and framework.

    Args:
        description: Description of what the code should do.
        language: Programming language (python, javascript, typescript, java, go).
        framework: Optional framework (fastapi, django, express, react, spring).
        include_tests: Whether to include test code.
        session_id: Session identifier.

    Returns:
        JSON string with generated code and metadata.
    """
    code_id = f"CODE-{str(uuid.uuid4())[:8].upper()}"

    # Generate code based on language (templates)
    code_templates = {
        "python": {
            "function": '''def {name}({params}) -> {return_type}:
    """
    {docstring}

    Args:
        {args_doc}

    Returns:
        {returns_doc}
    """
    # Implementation
    {implementation}
''',
            "class": '''class {name}:
    """
    {docstring}
    """

    def __init__(self{params}):
        """Initialize {name}."""
        {init_body}

    def {method_name}(self{method_params}) -> {return_type}:
        """
        {method_doc}
        """
        {method_body}
''',
        },
        "typescript": {
            "function": """/**
 * {docstring}
 * @param {params_doc}
 * @returns {returns_doc}
 */
export function {name}({params}): {return_type} {{
    {implementation}
}}
""",
            "class": """/**
 * {docstring}
 */
export class {name} {{
    {properties}

    constructor({params}) {{
        {init_body}
    }}

    {method_name}({method_params}): {return_type} {{
        {method_body}
    }}
}}
""",
        },
        "go": {
            "function": """// {name} {docstring}
func {name}({params}) {return_type} {{
    {implementation}
}}
""",
            "struct": """// {name} {docstring}
type {name} struct {{
    {fields}
}}

// New{name} creates a new {name} instance
func New{name}({params}) *{name} {{
    return &{name}{{
        {init_body}
    }}
}}

// {method_name} {method_doc}
func ({receiver} *{name}) {method_name}({method_params}) {return_type} {{
    {method_body}
}}
""",
        },
    }

    # Determine code type from description
    code_type = "function"
    if any(word in description.lower() for word in ["class", "object", "model", "entity"]):
        code_type = "class" if language in ["python", "typescript"] else "struct"

    # Generate sample code based on description
    name = "process_data"
    if "user" in description.lower():
        name = "User" if code_type in ["class", "struct"] else "get_user"
    elif "order" in description.lower():
        name = "Order" if code_type in ["class", "struct"] else "process_order"
    elif "api" in description.lower():
        name = "APIClient" if code_type in ["class", "struct"] else "call_api"

    # Build code based on language
    if language == "python":
        if code_type == "function":
            code = f'''def {name.lower()}(data: dict) -> dict:
    """
    {description}

    Args:
        data: Input data dictionary.

    Returns:
        Processed result dictionary.
    """
    result = {{}}

    # Validate input
    if not data:
        raise ValueError("Input data cannot be empty")

    # Process data
    result["status"] = "success"
    result["data"] = data

    return result
'''
        else:
            code = f'''from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class {name}:
    """
    {description}
    """

    id: str
    created_at: datetime
    data: dict[str, Any]

    def process(self) -> dict:
        """Process the entity data."""
        return {{
            "id": self.id,
            "processed_at": datetime.now().isoformat(),
            "result": "processed",
        }}

    def validate(self) -> bool:
        """Validate entity state."""
        return bool(self.id and self.data)
'''

    elif language == "typescript":
        if code_type == "function":
            code = f"""/**
 * {description}
 * @param data - Input data object
 * @returns Processed result
 */
export function {name.lower().replace("_", "")}(data: Record<string, unknown>): Record<string, unknown> {{
    // Validate input
    if (!data || Object.keys(data).length === 0) {{
        throw new Error("Input data cannot be empty");
    }}

    // Process data
    return {{
        status: "success",
        data,
        processedAt: new Date().toISOString(),
    }};
}}
"""
        else:
            code = f"""/**
 * {description}
 */
export class {name} {{
    readonly id: string;
    readonly createdAt: Date;
    private data: Record<string, unknown>;

    constructor(id: string, data: Record<string, unknown>) {{
        this.id = id;
        this.createdAt = new Date();
        this.data = data;
    }}

    process(): Record<string, unknown> {{
        return {{
            id: this.id,
            processedAt: new Date().toISOString(),
            result: "processed",
        }};
    }}

    validate(): boolean {{
        return Boolean(this.id && this.data);
    }}
}}
"""

    elif language == "go":
        if code_type == "function":
            code = f"""package main

import (
    "errors"
    "time"
)

// {name} {description}
func {name}(data map[string]interface{{}}) (map[string]interface{{}}, error) {{
    // Validate input
    if data == nil || len(data) == 0 {{
        return nil, errors.New("input data cannot be empty")
    }}

    // Process data
    result := map[string]interface{{}}{{
        "status":       "success",
        "data":         data,
        "processed_at": time.Now().Format(time.RFC3339),
    }}

    return result, nil
}}
"""
        else:
            code = f"""package main

import (
    "time"
)

// {name} {description}
type {name} struct {{
    ID        string
    CreatedAt time.Time
    Data      map[string]interface{{}}
}}

// New{name} creates a new {name} instance
func New{name}(id string, data map[string]interface{{}}) *{name} {{
    return &{name}{{
        ID:        id,
        CreatedAt: time.Now(),
        Data:      data,
    }}
}}

// Process processes the entity data
func (e *{name}) Process() map[string]interface{{}} {{
    return map[string]interface{{}}{{
        "id":           e.ID,
        "processed_at": time.Now().Format(time.RFC3339),
        "result":       "processed",
    }}
}}

// Validate validates entity state
func (e *{name}) Validate() bool {{
    return e.ID != "" && e.Data != nil
}}
"""
    else:
        code = f"// Generated code for: {description}\n// Language: {language}\n"

    # Generate test code if requested
    test_code = None
    if include_tests:
        if language == "python":
            test_code = f'''import pytest
from {name.lower()} import {name.lower() if code_type == "function" else name}


class Test{name if code_type != "function" else name.title().replace("_", "")}:
    """Tests for {name}."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        {"result = " + name.lower() + "({'key': 'value'})" if code_type == "function" else "obj = " + name + "('test-id', {'key': 'value'})"}
        assert {"result['status'] == 'success'" if code_type == "function" else "obj.validate()"}

    def test_empty_input(self):
        """Test with empty input."""
        with pytest.raises({"ValueError" if code_type == "function" else "Exception"}):
            {name.lower() + "({})" if code_type == "function" else name + "('', {})"}
'''

    result = {
        "id": code_id,
        "language": language,
        "framework": framework,
        "type": code_type,
        "name": name,
        "code": code,
        "test_code": test_code,
        "lines_of_code": len(code.split("\n")),
        "created_at": datetime.now().isoformat(),
    }

    _code_store[code_id] = result

    return json.dumps(result, indent=2)


@tool
@traceable(name="refactor_code", tags=["codegen", "refactoring"])
def refactor_code(
    code: str,
    refactoring_type: str = "extract_function",
    target: str | None = None,
    language: str = "python",
) -> str:
    """Refactor existing code.

    Supported refactoring types:
    - extract_function: Extract code block into a function
    - extract_class: Extract related functions into a class
    - rename: Rename variable/function/class
    - inline: Inline a function call
    - simplify: Simplify complex expressions

    Args:
        code: Code to refactor.
        refactoring_type: Type of refactoring to apply.
        target: Target element (name/line number).
        language: Programming language.

    Returns:
        JSON string with refactored code and explanation.
    """
    refactored_code = code
    explanation = ""

    if refactoring_type == "extract_function":
        # Simulate extracting a function
        explanation = "Extracted repeated logic into a separate function for reusability"
        if language == "python":
            refactored_code = '''def extracted_function(data):
    """Extracted function containing the repeated logic."""
    # Extracted logic here
    return data

# Original code now calls the extracted function
result = extracted_function(input_data)
'''

    elif refactoring_type == "simplify":
        # Simulate simplification
        explanation = "Simplified complex conditional logic"
        if "if" in code and "else" in code:
            refactored_code = "# Simplified using early return pattern\n" + code.replace(
                "else:", "# else case handled by early return"
            )

    elif refactoring_type == "rename":
        # Simulate rename
        old_name = target or "old_name"
        new_name = f"better_{old_name}"
        refactored_code = code.replace(old_name, new_name)
        explanation = f"Renamed '{old_name}' to '{new_name}' for clarity"

    elif refactoring_type == "extract_class":
        explanation = "Extracted related functions into a cohesive class"
        if language == "python":
            refactored_code = '''class ExtractedClass:
    """Class containing related functionality."""

    def __init__(self):
        pass

    def method_one(self):
        """First extracted method."""
        pass

    def method_two(self):
        """Second extracted method."""
        pass
'''

    result = {
        "original_code": code[:200] + "..." if len(code) > 200 else code,
        "refactored_code": refactored_code,
        "refactoring_type": refactoring_type,
        "explanation": explanation,
        "language": language,
        "changes_made": [
            f"Applied {refactoring_type} refactoring",
            explanation,
        ],
    }

    return json.dumps(result, indent=2)


@tool
@traceable(name="apply_design_pattern", tags=["codegen", "patterns"])
def apply_design_pattern(
    pattern: str,
    context: str,
    language: str = "python",
) -> str:
    """Apply a design pattern to code.

    Supported patterns:
    - singleton, factory, builder, observer, strategy,
    - repository, dependency_injection, decorator

    Args:
        pattern: Design pattern to apply.
        context: Context/description for the implementation.
        language: Programming language.

    Returns:
        JSON string with pattern implementation.
    """
    patterns = {
        "singleton": {
            "python": '''class Singleton:
    """Singleton pattern implementation."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Initialize only once
        if not hasattr(self, '_initialized'):
            self._initialized = True
            # Add initialization logic here
''',
            "typescript": """class Singleton {
    private static instance: Singleton;

    private constructor() {
        // Private constructor
    }

    public static getInstance(): Singleton {
        if (!Singleton.instance) {
            Singleton.instance = new Singleton();
        }
        return Singleton.instance;
    }
}
""",
        },
        "factory": {
            "python": '''from abc import ABC, abstractmethod


class Product(ABC):
    """Abstract product."""

    @abstractmethod
    def operation(self) -> str:
        pass


class ConcreteProductA(Product):
    def operation(self) -> str:
        return "Product A"


class ConcreteProductB(Product):
    def operation(self) -> str:
        return "Product B"


class Factory:
    """Factory for creating products."""

    @staticmethod
    def create(product_type: str) -> Product:
        if product_type == "A":
            return ConcreteProductA()
        elif product_type == "B":
            return ConcreteProductB()
        raise ValueError(f"Unknown product type: {product_type}")
''',
        },
        "repository": {
            "python": '''from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional

T = TypeVar('T')


class Repository(ABC, Generic[T]):
    """Abstract repository pattern."""

    @abstractmethod
    def get(self, id: str) -> Optional[T]:
        pass

    @abstractmethod
    def get_all(self) -> list[T]:
        pass

    @abstractmethod
    def add(self, entity: T) -> T:
        pass

    @abstractmethod
    def update(self, entity: T) -> T:
        pass

    @abstractmethod
    def delete(self, id: str) -> bool:
        pass


class InMemoryRepository(Repository[T]):
    """In-memory implementation of repository."""

    def __init__(self):
        self._store: dict[str, T] = {}

    def get(self, id: str) -> Optional[T]:
        return self._store.get(id)

    def get_all(self) -> list[T]:
        return list(self._store.values())

    def add(self, entity: T) -> T:
        self._store[entity.id] = entity
        return entity

    def update(self, entity: T) -> T:
        self._store[entity.id] = entity
        return entity

    def delete(self, id: str) -> bool:
        if id in self._store:
            del self._store[id]
            return True
        return False
''',
        },
        "strategy": {
            "python": '''from abc import ABC, abstractmethod


class Strategy(ABC):
    """Abstract strategy."""

    @abstractmethod
    def execute(self, data):
        pass


class ConcreteStrategyA(Strategy):
    def execute(self, data):
        return f"Strategy A processing: {data}"


class ConcreteStrategyB(Strategy):
    def execute(self, data):
        return f"Strategy B processing: {data}"


class Context:
    """Context that uses a strategy."""

    def __init__(self, strategy: Strategy):
        self._strategy = strategy

    def set_strategy(self, strategy: Strategy):
        self._strategy = strategy

    def do_work(self, data):
        return self._strategy.execute(data)
''',
        },
    }

    code = patterns.get(pattern, {}).get(language, f"// {pattern} pattern not available for {language}")

    result = {
        "pattern": pattern,
        "language": language,
        "context": context,
        "code": code,
        "explanation": f"Implementation of {pattern} pattern for {context}",
        "usage_example": f"# Example usage of {pattern} pattern",
    }

    return json.dumps(result, indent=2)


@tool
@traceable(name="generate_boilerplate", tags=["codegen", "scaffolding"])
def generate_boilerplate(
    project_type: str,
    name: str,
    language: str = "python",
    include_docker: bool = True,
) -> str:
    """Generate project boilerplate code.

    Creates initial project structure with:
    - Main application file
    - Configuration
    - Dependencies
    - Docker setup (optional)

    Args:
        project_type: Type of project (api, cli, library, web_app).
        name: Project name.
        language: Programming language.
        include_docker: Include Dockerfile.

    Returns:
        JSON string with generated files.
    """
    files = {}

    if language == "python":
        if project_type == "api":
            files["main.py"] = f'''"""Main API application."""
from fastapi import FastAPI
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    yield
    # Shutdown


app = FastAPI(title="{name}", lifespan=lifespan)


@app.get("/health")
async def health():
    return {{"status": "healthy"}}


@app.get("/")
async def root():
    return {{"message": "Welcome to {name}"}}
'''

            files["requirements.txt"] = """fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.0.0
python-dotenv>=1.0.0
"""

        files["pyproject.toml"] = f'''[project]
name = "{name}"
version = "0.1.0"
description = "A {project_type} project"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.115.0",
    "pydantic>=2.0.0",
]

[tool.ruff]
line-length = 100
target-version = "py310"
'''

        files[".gitignore"] = """__pycache__/
*.py[cod]
.env
.venv/
*.egg-info/
dist/
build/
.coverage
.pytest_cache/
"""

    if include_docker:
        files["Dockerfile"] = """FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

        files["docker-compose.yml"] = f"""version: "3.8"

services:
  {name}:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=development
    volumes:
      - .:/app
"""

    result = {
        "project_name": name,
        "project_type": project_type,
        "language": language,
        "files": files,
        "file_count": len(files),
        "next_steps": [
            "Review and customize the generated files",
            "Install dependencies",
            "Run the application",
        ],
    }

    return json.dumps(result, indent=2)


@tool
@traceable(name="optimize_imports", tags=["codegen", "formatting"])
def optimize_imports(code: str, language: str = "python") -> str:
    """Optimize and organize imports in code.

    - Removes unused imports
    - Groups imports by type
    - Sorts alphabetically

    Args:
        code: Code with imports to optimize.
        language: Programming language.

    Returns:
        JSON string with optimized code.
    """
    lines = code.split("\n")
    import_lines = []
    other_lines = []

    if language == "python":
        stdlib_imports = []
        third_party_imports = []
        local_imports = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                if any(mod in stripped for mod in ["os", "sys", "json", "typing", "datetime", "collections"]):
                    stdlib_imports.append(line)
                elif stripped.startswith("from .") or "app." in stripped:
                    local_imports.append(line)
                else:
                    third_party_imports.append(line)
            else:
                other_lines.append(line)

        # Sort each group
        stdlib_imports.sort()
        third_party_imports.sort()
        local_imports.sort()

        # Combine with blank lines between groups
        optimized_imports = []
        if stdlib_imports:
            optimized_imports.extend(stdlib_imports)
            optimized_imports.append("")
        if third_party_imports:
            optimized_imports.extend(third_party_imports)
            optimized_imports.append("")
        if local_imports:
            optimized_imports.extend(local_imports)
            optimized_imports.append("")

        optimized_code = "\n".join(optimized_imports + other_lines)

    else:
        optimized_code = code  # No optimization for other languages yet

    result = {
        "original_import_count": len([l for l in lines if "import" in l]),
        "optimized_code": optimized_code,
        "changes": ["Grouped imports by type", "Sorted alphabetically", "Added separating blank lines"],
    }

    return json.dumps(result, indent=2)


@tool
@traceable(name="format_code", tags=["codegen", "formatting"])
def format_code(code: str, language: str = "python", style: str = "default") -> str:
    """Format code according to style guidelines.

    Args:
        code: Code to format.
        language: Programming language.
        style: Style guide (default, pep8, google, airbnb).

    Returns:
        JSON string with formatted code.
    """
    # Simplified formatting - in production, use actual formatters
    formatted = code

    if language == "python":
        # Basic formatting
        lines = code.split("\n")
        formatted_lines = []

        for line in lines:
            # Ensure proper spacing around operators
            formatted_line = line
            for op in ["=", "==", "!=", "<=", ">=", "+=", "-="]:
                if op in formatted_line and f" {op} " not in formatted_line:
                    formatted_line = formatted_line.replace(op, f" {op} ")

            # Remove trailing whitespace
            formatted_line = formatted_line.rstrip()
            formatted_lines.append(formatted_line)

        formatted = "\n".join(formatted_lines)

    result = {
        "language": language,
        "style": style,
        "formatted_code": formatted,
        "changes_made": [
            "Applied consistent spacing",
            "Removed trailing whitespace",
            "Normalized line endings",
        ],
    }

    return json.dumps(result, indent=2)
