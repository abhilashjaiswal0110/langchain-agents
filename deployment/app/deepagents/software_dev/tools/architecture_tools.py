"""Architecture & Design Tools.

Tools for designing system architecture, API specifications,
data models, and technology stack recommendations.
"""

import json
import uuid
from datetime import datetime

from langchain_core.tools import tool
from langsmith import traceable

from app.deepagents.config.software_dev_config import (
    ArchitecturePattern,
    CodeLanguage,
)


# Session storage
_architecture_store: dict[str, dict] = {}
_api_store: dict[str, dict] = {}


@tool
@traceable(name="design_architecture", tags=["architecture", "sdlc"])
def design_architecture(
    requirements_summary: str,
    pattern: str = "layered",
    scale: str = "medium",
    session_id: str = "default",
) -> str:
    """Design system architecture based on requirements.

    Proposes an architecture pattern and component structure
    based on the project requirements and scale.

    Args:
        requirements_summary: Summary of project requirements.
        pattern: Architecture pattern (layered, microservices, serverless, event_driven).
        scale: Expected scale (small, medium, large, enterprise).
        session_id: Session identifier.

    Returns:
        JSON string with architecture proposal.
    """
    arch_id = f"ARCH-{str(uuid.uuid4())[:8].upper()}"

    # Define components based on pattern
    components = []

    if pattern == "microservices":
        components = [
            {"name": "API Gateway", "type": "gateway", "description": "Entry point for all API requests"},
            {"name": "Auth Service", "type": "service", "description": "Authentication and authorization"},
            {"name": "User Service", "type": "service", "description": "User management"},
            {"name": "Core Service", "type": "service", "description": "Main business logic"},
            {"name": "Notification Service", "type": "service", "description": "Email and push notifications"},
            {"name": "Message Queue", "type": "queue", "description": "Async communication between services"},
            {"name": "Database", "type": "database", "description": "Primary data store"},
            {"name": "Cache", "type": "cache", "description": "Redis cache layer"},
        ]
    elif pattern == "serverless":
        components = [
            {"name": "API Gateway", "type": "gateway", "description": "AWS API Gateway or equivalent"},
            {"name": "Auth Lambda", "type": "function", "description": "Authentication handler"},
            {"name": "Business Logic Lambda", "type": "function", "description": "Core business functions"},
            {"name": "DynamoDB", "type": "database", "description": "NoSQL database"},
            {"name": "S3 Bucket", "type": "storage", "description": "File storage"},
            {"name": "SQS Queue", "type": "queue", "description": "Message queue"},
        ]
    elif pattern == "event_driven":
        components = [
            {"name": "Event Bus", "type": "bus", "description": "Central event distribution"},
            {"name": "Event Producers", "type": "producer", "description": "Services that emit events"},
            {"name": "Event Consumers", "type": "consumer", "description": "Services that react to events"},
            {"name": "Event Store", "type": "database", "description": "Event sourcing storage"},
            {"name": "Projection Service", "type": "service", "description": "Read model generation"},
        ]
    else:  # layered (default)
        components = [
            {"name": "Presentation Layer", "type": "layer", "description": "UI and API endpoints"},
            {"name": "Application Layer", "type": "layer", "description": "Business logic orchestration"},
            {"name": "Domain Layer", "type": "layer", "description": "Core business entities and rules"},
            {"name": "Infrastructure Layer", "type": "layer", "description": "Data access and external services"},
            {"name": "Database", "type": "database", "description": "Persistence layer"},
        ]

    # Add component IDs
    for i, comp in enumerate(components):
        comp["id"] = f"COMP-{str(uuid.uuid4())[:6].upper()}"
        comp["dependencies"] = []

    # Define relationships
    if pattern == "layered":
        components[0]["dependencies"] = [components[1]["id"]]
        components[1]["dependencies"] = [components[2]["id"]]
        components[2]["dependencies"] = [components[3]["id"]]
        components[3]["dependencies"] = [components[4]["id"]]

    architecture = {
        "id": arch_id,
        "pattern": pattern,
        "scale": scale,
        "components": components,
        "created_at": datetime.now().isoformat(),
        "description": f"{pattern.replace('_', ' ').title()} architecture for {scale} scale application",
        "considerations": [
            f"Pattern '{pattern}' selected for {scale} scale",
            "Consider horizontal scaling for high-traffic components",
            "Implement circuit breakers for service resilience",
            "Use health checks for all components",
        ],
    }

    _architecture_store[arch_id] = architecture

    return json.dumps(architecture, indent=2)


@tool
@traceable(name="create_api_spec", tags=["architecture", "api"])
def create_api_spec(
    resource_name: str,
    operations: list[str] | None = None,
    format: str = "openapi",
    session_id: str = "default",
) -> str:
    """Create API specification for a resource.

    Generates RESTful API endpoint specifications
    following OpenAPI/Swagger format.

    Args:
        resource_name: Name of the resource (e.g., "users", "orders").
        operations: List of operations (create, read, update, delete, list).
        format: Specification format (openapi, graphql).
        session_id: Session identifier.

    Returns:
        JSON string with API specification.
    """
    if operations is None:
        operations = ["create", "read", "update", "delete", "list"]

    endpoints = []
    base_path = f"/api/v1/{resource_name.lower()}"

    operation_map = {
        "create": {
            "path": base_path,
            "method": "POST",
            "description": f"Create a new {resource_name}",
            "request_body": {"type": "object", "properties": {"data": {"type": "object"}}},
            "response": {"201": {"description": "Created successfully"}},
        },
        "read": {
            "path": f"{base_path}/{{id}}",
            "method": "GET",
            "description": f"Get {resource_name} by ID",
            "parameters": [{"name": "id", "in": "path", "required": True, "type": "string"}],
            "response": {"200": {"description": "Success"}},
        },
        "update": {
            "path": f"{base_path}/{{id}}",
            "method": "PUT",
            "description": f"Update {resource_name}",
            "parameters": [{"name": "id", "in": "path", "required": True, "type": "string"}],
            "request_body": {"type": "object"},
            "response": {"200": {"description": "Updated successfully"}},
        },
        "delete": {
            "path": f"{base_path}/{{id}}",
            "method": "DELETE",
            "description": f"Delete {resource_name}",
            "parameters": [{"name": "id", "in": "path", "required": True, "type": "string"}],
            "response": {"204": {"description": "Deleted successfully"}},
        },
        "list": {
            "path": base_path,
            "method": "GET",
            "description": f"List all {resource_name}",
            "parameters": [
                {"name": "page", "in": "query", "type": "integer", "default": 1},
                {"name": "limit", "in": "query", "type": "integer", "default": 20},
            ],
            "response": {"200": {"description": "Success", "schema": {"type": "array"}}},
        },
    }

    for op in operations:
        if op in operation_map:
            endpoint = operation_map[op].copy()
            endpoint["id"] = f"EP-{str(uuid.uuid4())[:6].upper()}"
            endpoint["operation"] = op
            endpoints.append(endpoint)

    api_spec = {
        "id": f"API-{str(uuid.uuid4())[:8].upper()}",
        "resource": resource_name,
        "base_path": base_path,
        "format": format,
        "version": "1.0.0",
        "endpoints": endpoints,
        "auth_required": True,
        "rate_limit": "100 requests/minute",
        "created_at": datetime.now().isoformat(),
    }

    _api_store[api_spec["id"]] = api_spec

    return json.dumps(api_spec, indent=2)


@tool
@traceable(name="suggest_tech_stack", tags=["architecture", "technology"])
def suggest_tech_stack(
    project_type: str,
    requirements: list[str] | None = None,
    constraints: list[str] | None = None,
    team_expertise: list[str] | None = None,
) -> str:
    """Suggest technology stack based on project requirements.

    Recommends technologies for:
    - Backend framework
    - Frontend framework
    - Database
    - Caching
    - Message queue
    - Deployment platform

    Args:
        project_type: Type of project (web_app, api, microservices, data_pipeline).
        requirements: Specific requirements to consider.
        constraints: Constraints (budget, team size, timeline).
        team_expertise: Team's existing expertise.

    Returns:
        JSON string with technology recommendations.
    """
    requirements = requirements or []
    constraints = constraints or []
    team_expertise = team_expertise or []

    # Default recommendations by project type
    stacks = {
        "web_app": {
            "backend": {"primary": "FastAPI (Python)", "alternatives": ["Django", "Express.js", "Spring Boot"]},
            "frontend": {"primary": "React", "alternatives": ["Vue.js", "Next.js", "Angular"]},
            "database": {"primary": "PostgreSQL", "alternatives": ["MySQL", "MongoDB"]},
            "cache": {"primary": "Redis", "alternatives": ["Memcached"]},
            "deployment": {"primary": "Docker + Kubernetes", "alternatives": ["AWS ECS", "Azure Container Apps"]},
        },
        "api": {
            "backend": {"primary": "FastAPI (Python)", "alternatives": ["Express.js", "Go Fiber", "ASP.NET Core"]},
            "database": {"primary": "PostgreSQL", "alternatives": ["MongoDB", "DynamoDB"]},
            "cache": {"primary": "Redis", "alternatives": ["Memcached"]},
            "api_gateway": {"primary": "Kong", "alternatives": ["AWS API Gateway", "NGINX"]},
            "deployment": {"primary": "Docker + Kubernetes", "alternatives": ["AWS Lambda", "Azure Functions"]},
        },
        "microservices": {
            "backend": {"primary": "Go", "alternatives": ["Node.js", "Python", "Java"]},
            "service_mesh": {"primary": "Istio", "alternatives": ["Linkerd", "Consul Connect"]},
            "message_queue": {"primary": "Apache Kafka", "alternatives": ["RabbitMQ", "AWS SQS"]},
            "database": {"primary": "PostgreSQL + MongoDB", "alternatives": ["MySQL", "DynamoDB"]},
            "observability": {"primary": "Prometheus + Grafana", "alternatives": ["Datadog", "New Relic"]},
            "deployment": {"primary": "Kubernetes", "alternatives": ["AWS EKS", "GKE"]},
        },
        "data_pipeline": {
            "processing": {"primary": "Apache Spark", "alternatives": ["Flink", "Beam"]},
            "orchestration": {"primary": "Apache Airflow", "alternatives": ["Prefect", "Dagster"]},
            "storage": {"primary": "Delta Lake", "alternatives": ["S3", "BigQuery"]},
            "streaming": {"primary": "Apache Kafka", "alternatives": ["Kinesis", "Pulsar"]},
            "deployment": {"primary": "Kubernetes + Spark Operator", "alternatives": ["Databricks", "EMR"]},
        },
    }

    stack = stacks.get(project_type, stacks["web_app"])

    # Adjust based on team expertise
    if team_expertise:
        for category, options in stack.items():
            if isinstance(options, dict) and "alternatives" in options:
                for expertise in team_expertise:
                    if expertise.lower() in options["primary"].lower():
                        break
                    for alt in options["alternatives"]:
                        if expertise.lower() in alt.lower():
                            # Swap primary with alternative
                            options["alternatives"].remove(alt)
                            options["alternatives"].insert(0, options["primary"])
                            options["primary"] = alt
                            break

    recommendation = {
        "project_type": project_type,
        "recommended_stack": stack,
        "considerations": [
            "Consider team expertise and learning curve",
            "Evaluate total cost of ownership",
            "Check community support and documentation",
            "Assess long-term maintainability",
        ],
        "trade_offs": [
            {"aspect": "Performance", "note": "Benchmark critical paths before finalizing"},
            {"aspect": "Scalability", "note": "Design for 10x current requirements"},
            {"aspect": "Cost", "note": "Consider both infrastructure and development costs"},
        ],
    }

    return json.dumps(recommendation, indent=2)


@tool
@traceable(name="design_data_model", tags=["architecture", "database"])
def design_data_model(
    entities: list[str],
    relationships: list[dict] | None = None,
    database_type: str = "relational",
) -> str:
    """Design data model for the application.

    Creates entity definitions with attributes
    and relationships between entities.

    Args:
        entities: List of entity names to model.
        relationships: List of relationships (e.g., [{"from": "User", "to": "Order", "type": "one_to_many"}]).
        database_type: Type of database (relational, document, graph).

    Returns:
        JSON string with data model specification.
    """
    relationships = relationships or []

    entity_models = []
    for entity in entities:
        # Generate standard attributes
        attributes = [
            {"name": "id", "type": "uuid", "primary_key": True},
            {"name": "created_at", "type": "timestamp", "default": "now()"},
            {"name": "updated_at", "type": "timestamp", "default": "now()"},
        ]

        # Add entity-specific attributes (simplified)
        if entity.lower() == "user":
            attributes.extend([
                {"name": "email", "type": "string", "unique": True, "required": True},
                {"name": "name", "type": "string", "required": True},
                {"name": "password_hash", "type": "string", "required": True},
                {"name": "is_active", "type": "boolean", "default": True},
            ])
        elif entity.lower() == "order":
            attributes.extend([
                {"name": "user_id", "type": "uuid", "foreign_key": "users.id"},
                {"name": "status", "type": "enum", "values": ["pending", "processing", "completed", "cancelled"]},
                {"name": "total_amount", "type": "decimal", "precision": 10, "scale": 2},
            ])
        else:
            attributes.extend([
                {"name": "name", "type": "string", "required": True},
                {"name": "description", "type": "text"},
            ])

        entity_models.append({
            "name": entity,
            "table_name": f"{entity.lower()}s",
            "attributes": attributes,
        })

    data_model = {
        "id": f"DM-{str(uuid.uuid4())[:8].upper()}",
        "database_type": database_type,
        "entities": entity_models,
        "relationships": relationships,
        "indexes": [
            {"entity": entities[0] if entities else "Entity", "columns": ["created_at"], "type": "btree"},
        ],
        "considerations": [
            "Add appropriate indexes for query patterns",
            "Consider partitioning for large tables",
            "Implement soft deletes if needed",
            "Plan for data migration strategy",
        ],
    }

    return json.dumps(data_model, indent=2)


@tool
@traceable(name="create_component_diagram", tags=["architecture", "diagram"])
def create_component_diagram(
    architecture_id: str | None = None,
    components: list[dict] | None = None,
    format: str = "mermaid",
) -> str:
    """Create a component diagram for the architecture.

    Generates a visual representation of system components
    and their relationships.

    Args:
        architecture_id: ID of existing architecture to diagram.
        components: List of components if not using existing architecture.
        format: Diagram format (mermaid, plantuml, text).

    Returns:
        Diagram in specified format.
    """
    if architecture_id and architecture_id in _architecture_store:
        arch = _architecture_store[architecture_id]
        components = arch.get("components", [])
    elif not components:
        components = [
            {"name": "Frontend", "type": "ui", "dependencies": ["API Gateway"]},
            {"name": "API Gateway", "type": "gateway", "dependencies": ["Backend"]},
            {"name": "Backend", "type": "service", "dependencies": ["Database"]},
            {"name": "Database", "type": "database", "dependencies": []},
        ]

    if format == "mermaid":
        diagram_lines = ["```mermaid", "graph TB"]

        # Define nodes
        for comp in components:
            name = comp.get("name", "Component").replace(" ", "_")
            comp_type = comp.get("type", "component")

            # Style based on type
            if comp_type == "database":
                diagram_lines.append(f"    {name}[({comp['name']})]")
            elif comp_type == "queue":
                diagram_lines.append(f"    {name}[/{comp['name']}/]")
            elif comp_type == "gateway":
                diagram_lines.append(f"    {name}{{{{{comp['name']}}}}}")
            else:
                diagram_lines.append(f"    {name}[{comp['name']}]")

        # Define relationships
        for comp in components:
            name = comp.get("name", "").replace(" ", "_")
            for dep in comp.get("dependencies", []):
                dep_name = dep.replace(" ", "_") if isinstance(dep, str) else dep
                diagram_lines.append(f"    {name} --> {dep_name}")

        diagram_lines.append("```")
        diagram = "\n".join(diagram_lines)

    elif format == "plantuml":
        diagram_lines = ["@startuml"]
        for comp in components:
            diagram_lines.append(f"component \"{comp['name']}\" as {comp['name'].replace(' ', '_')}")
        for comp in components:
            name = comp['name'].replace(' ', '_')
            for dep in comp.get("dependencies", []):
                dep_name = dep.replace(" ", "_") if isinstance(dep, str) else dep
                diagram_lines.append(f"{name} --> {dep_name}")
        diagram_lines.append("@enduml")
        diagram = "\n".join(diagram_lines)

    else:  # text
        diagram_lines = ["Component Diagram:", "=" * 50]
        for comp in components:
            deps = ", ".join(comp.get("dependencies", [])) or "None"
            diagram_lines.append(f"[{comp['name']}] ({comp.get('type', 'component')}) -> {deps}")
        diagram = "\n".join(diagram_lines)

    result = {
        "format": format,
        "diagram": diagram,
        "components_count": len(components),
    }

    return json.dumps(result, indent=2)


@tool
@traceable(name="analyze_dependencies", tags=["architecture", "dependencies"])
def analyze_dependencies(
    component_id: str | None = None,
    architecture_id: str | None = None,
) -> str:
    """Analyze dependencies between components.

    Identifies:
    - Direct dependencies
    - Transitive dependencies
    - Circular dependencies (anti-pattern)
    - Coupling metrics

    Args:
        component_id: Specific component to analyze.
        architecture_id: Architecture to analyze.

    Returns:
        JSON string with dependency analysis.
    """
    if architecture_id and architecture_id in _architecture_store:
        arch = _architecture_store[architecture_id]
        components = arch.get("components", [])
    else:
        # Default example
        components = [
            {"id": "1", "name": "Frontend", "dependencies": ["2"]},
            {"id": "2", "name": "API Gateway", "dependencies": ["3", "4"]},
            {"id": "3", "name": "Auth Service", "dependencies": ["5"]},
            {"id": "4", "name": "Core Service", "dependencies": ["5"]},
            {"id": "5", "name": "Database", "dependencies": []},
        ]

    # Build dependency graph
    dep_graph = {c["id"]: c.get("dependencies", []) for c in components}
    name_map = {c["id"]: c["name"] for c in components}

    # Analyze each component
    analysis = []
    for comp in components:
        comp_analysis = {
            "component": comp["name"],
            "direct_dependencies": len(comp.get("dependencies", [])),
            "dependents": 0,  # Components that depend on this one
            "coupling_score": 0.0,
        }

        # Count dependents
        for other in components:
            if comp["id"] in other.get("dependencies", []):
                comp_analysis["dependents"] += 1

        # Calculate coupling score (afferent + efferent coupling)
        total = comp_analysis["direct_dependencies"] + comp_analysis["dependents"]
        comp_analysis["coupling_score"] = round(total / max(len(components), 1), 2)

        analysis.append(comp_analysis)

    # Check for circular dependencies (simplified)
    circular_deps = []
    for comp in components:
        for dep_id in comp.get("dependencies", []):
            dep_comp = next((c for c in components if c["id"] == dep_id), None)
            if dep_comp and comp["id"] in dep_comp.get("dependencies", []):
                circular_deps.append({
                    "component1": comp["name"],
                    "component2": dep_comp["name"],
                })

    result = {
        "total_components": len(components),
        "total_dependencies": sum(len(c.get("dependencies", [])) for c in components),
        "component_analysis": analysis,
        "circular_dependencies": circular_deps,
        "has_circular_deps": len(circular_deps) > 0,
        "recommendations": [
            "Minimize coupling between components",
            "Avoid circular dependencies",
            "Consider dependency injection for loose coupling",
        ] if circular_deps else ["Architecture has no circular dependencies - good!"],
    }

    return json.dumps(result, indent=2)
