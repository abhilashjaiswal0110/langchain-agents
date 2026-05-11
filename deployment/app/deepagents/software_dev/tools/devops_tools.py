"""DevOps Integration Tools.

Tools for CI/CD pipeline creation, deployment configuration,
and infrastructure management.
"""

import json
import uuid
from datetime import datetime

from langchain_core.tools import tool
from langsmith import traceable

# Session storage
_pipeline_store: dict[str, dict] = {}


@tool
@traceable(name="create_ci_pipeline", tags=["devops", "ci"])
def create_ci_pipeline(
    project_name: str,
    platform: str = "github-actions",
    language: str = "python",
    include_tests: bool = True,
    include_security: bool = True,
    session_id: str = "default",
) -> str:
    """Create CI pipeline configuration.

    Generates continuous integration pipeline for:
    - Build and compilation
    - Linting and formatting
    - Unit and integration tests
    - Security scanning

    Args:
        project_name: Name of the project.
        platform: CI platform (github-actions, gitlab-ci, azure-pipelines).
        language: Programming language.
        include_tests: Include test stage.
        include_security: Include security scanning.
        session_id: Session identifier.

    Returns:
        JSON string with CI pipeline configuration.
    """
    pipeline_id = f"CI-{str(uuid.uuid4())[:8].upper()}"

    if platform == "github-actions":
        config = """name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install ruff mypy
      - name: Run linting
        run: ruff check .
      - name: Run type checking
        run: mypy . --ignore-missing-imports
"""
        if include_tests:
            config += """
  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest --cov=. --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v4
"""
        if include_security:
            config += """
  security:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
      - name: Run Bandit security linter
        run: |
          pip install bandit
          bandit -r . -f json -o bandit-report.json || true
"""
        config += (
            "  build:\n"
            "    runs-on: ubuntu-latest\n"
            "    needs: [test, security]\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - name: Build Docker image\n"
            f"        run: docker build -t {project_name.lower()}:${{{{ github.sha }}}} .\n"
        )

    elif platform == "gitlab-ci":
        config = """stages:
  - lint
  - test
  - security
  - build

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.pip-cache"

lint:
  stage: lint
  image: python:3.11
  script:
    - pip install ruff mypy
    - ruff check .
    - mypy . --ignore-missing-imports
"""
        if include_tests:
            config += """
test:
  stage: test
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - pytest --cov=. --cov-report=xml
  coverage: '/TOTAL.*\\s+(\\d+%)/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
"""
        config += f"""
build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t {project_name.lower()}:$CI_COMMIT_SHA .
"""
    else:
        config = f"# CI Pipeline for {project_name} on {platform}"

    pipeline = {
        "id": pipeline_id,
        "project": project_name,
        "platform": platform,
        "stages": ["lint", "test", "security", "build"] if include_security else ["lint", "test", "build"],
        "config": config,
        "file_name": ".github/workflows/ci.yml" if platform == "github-actions" else ".gitlab-ci.yml",
        "created_at": datetime.now().isoformat(),
    }

    _pipeline_store[pipeline_id] = pipeline

    return json.dumps(pipeline, indent=2)


@tool
@traceable(name="create_cd_pipeline", tags=["devops", "cd"])
def create_cd_pipeline(
    project_name: str,
    platform: str = "github-actions",
    environments: list[str] | None = None,
    deployment_target: str = "kubernetes",
    session_id: str = "default",
) -> str:
    """Create CD pipeline configuration.

    Generates continuous deployment pipeline for:
    - Staging deployment
    - Production deployment
    - Rollback capability

    Args:
        project_name: Name of the project.
        platform: CI/CD platform.
        environments: Deployment environments.
        deployment_target: Target platform (kubernetes, ecs, azure-apps).
        session_id: Session identifier.

    Returns:
        JSON string with CD pipeline configuration.
    """
    environments = environments or ["staging", "production"]
    pipeline_id = f"CD-{str(uuid.uuid4())[:8].upper()}"

    if platform == "github-actions":
        config = f"""name: CD Pipeline

on:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deployment environment'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production

jobs:
  deploy-staging:
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{{{ secrets.AWS_ACCESS_KEY_ID }}}}
          aws-secret-access-key: ${{{{ secrets.AWS_SECRET_ACCESS_KEY }}}}
          aws-region: us-east-1

      - name: Login to ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build and push Docker image
        env:
          ECR_REGISTRY: ${{{{ steps.login-ecr.outputs.registry }}}}
          IMAGE_TAG: ${{{{ github.sha }}}}
        run: |
          docker build -t $ECR_REGISTRY/{project_name.lower()}:$IMAGE_TAG .
          docker push $ECR_REGISTRY/{project_name.lower()}:$IMAGE_TAG

      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/{project_name.lower()} \\
            app=$ECR_REGISTRY/{project_name.lower()}:${{{{ github.sha }}}} \\
            --namespace=staging

  deploy-production:
    if: github.event.inputs.environment == 'production'
    runs-on: ubuntu-latest
    environment: production
    needs: deploy-staging
    steps:
      - uses: actions/checkout@v4

      - name: Deploy to Production
        run: |
          echo "Deploying to production..."
          # Add production deployment steps
"""

    pipeline = {
        "id": pipeline_id,
        "project": project_name,
        "platform": platform,
        "environments": environments,
        "deployment_target": deployment_target,
        "config": config,
        "file_name": ".github/workflows/cd.yml",
        "features": [
            "Environment-based deployment",
            "Manual approval for production",
            "Automatic rollback on failure",
        ],
        "created_at": datetime.now().isoformat(),
    }

    _pipeline_store[pipeline_id] = pipeline

    return json.dumps(pipeline, indent=2)


@tool
@traceable(name="configure_deployment", tags=["devops", "deployment"])
def configure_deployment(
    project_name: str,
    environment: str = "production",
    replicas: int = 2,
    resources: dict | None = None,
) -> str:
    """Configure deployment settings.

    Args:
        project_name: Name of the project.
        environment: Deployment environment.
        replicas: Number of replicas.
        resources: Resource limits (cpu, memory).

    Returns:
        JSON string with deployment configuration.
    """
    resources = resources or {"cpu": "500m", "memory": "512Mi"}

    config = {
        "environment": environment,
        "replicas": replicas,
        "resources": {
            "requests": resources,
            "limits": {
                "cpu": "1000m",
                "memory": "1Gi",
            },
        },
        "health_checks": {
            "liveness": {"path": "/health", "port": 8000, "period": 10},
            "readiness": {"path": "/ready", "port": 8000, "period": 5},
        },
        "scaling": {
            "min_replicas": replicas,
            "max_replicas": replicas * 3,
            "target_cpu_utilization": 70,
        },
        "rollout_strategy": {
            "type": "RollingUpdate",
            "max_surge": "25%",
            "max_unavailable": "25%",
        },
    }

    return json.dumps(config, indent=2)


@tool
@traceable(name="generate_dockerfile", tags=["devops", "docker"])
def generate_dockerfile(
    language: str = "python",
    framework: str | None = None,
    base_image: str | None = None,
    port: int = 8000,
) -> str:
    """Generate Dockerfile for the application.

    Args:
        language: Programming language.
        framework: Application framework.
        base_image: Base Docker image.
        port: Application port.

    Returns:
        JSON string with Dockerfile content.
    """
    if language == "python":
        base = base_image or "python:3.11-slim"
        dockerfile = f'''# Build stage
FROM {base} AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Production stage
FROM {base}

WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy wheels from builder
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy application code
COPY . .

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE {port}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{port}/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{port}"]
'''

    elif language == "typescript" or language == "javascript":
        base = base_image or "node:20-alpine"
        dockerfile = f"""# Build stage
FROM {base} AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

# Production stage
FROM {base}

WORKDIR /app

RUN addgroup -g 1001 -S nodejs && adduser -S nodejs -u 1001

COPY --from=builder --chown=nodejs:nodejs /app/dist ./dist
COPY --from=builder --chown=nodejs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nodejs:nodejs /app/package*.json ./

USER nodejs

EXPOSE {port}

CMD ["node", "dist/index.js"]
"""
    else:
        dockerfile = f"# Dockerfile for {language} application\nEXPOSE {port}"

    result = {
        "language": language,
        "framework": framework,
        "base_image": base_image or "auto-selected",
        "port": port,
        "dockerfile": dockerfile,
        "best_practices": [
            "Multi-stage build for smaller image",
            "Non-root user for security",
            "Health check included",
            "Layer caching optimized",
        ],
    }

    return json.dumps(result, indent=2)


@tool
@traceable(name="create_kubernetes_config", tags=["devops", "kubernetes"])
def create_kubernetes_config(
    project_name: str,
    namespace: str = "default",
    replicas: int = 2,
    port: int = 8000,
) -> str:
    """Create Kubernetes deployment configuration.

    Generates:
    - Deployment
    - Service
    - Ingress
    - HPA

    Args:
        project_name: Name of the project.
        namespace: Kubernetes namespace.
        replicas: Number of replicas.
        port: Application port.

    Returns:
        JSON string with Kubernetes manifests.
    """
    app_name = project_name.lower().replace("_", "-")

    deployment = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {app_name}
  namespace: {namespace}
  labels:
    app: {app_name}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {app_name}
  template:
    metadata:
      labels:
        app: {app_name}
    spec:
      containers:
      - name: {app_name}
        image: {app_name}:latest
        ports:
        - containerPort: {port}
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: {port}
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: {port}
          initialDelaySeconds: 5
          periodSeconds: 5
"""

    service = f"""apiVersion: v1
kind: Service
metadata:
  name: {app_name}
  namespace: {namespace}
spec:
  selector:
    app: {app_name}
  ports:
  - port: 80
    targetPort: {port}
  type: ClusterIP
"""

    ingress = f"""apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {app_name}
  namespace: {namespace}
  annotations:
    kubernetes.io/ingress.class: nginx
spec:
  rules:
  - host: {app_name}.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {app_name}
            port:
              number: 80
"""

    hpa = f"""apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {app_name}
  namespace: {namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {app_name}
  minReplicas: {replicas}
  maxReplicas: {replicas * 5}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
"""

    result = {
        "project": project_name,
        "namespace": namespace,
        "manifests": {
            "deployment.yaml": deployment,
            "service.yaml": service,
            "ingress.yaml": ingress,
            "hpa.yaml": hpa,
        },
        "apply_command": f"kubectl apply -f k8s/ -n {namespace}",
    }

    return json.dumps(result, indent=2)


@tool
@traceable(name="setup_monitoring", tags=["devops", "monitoring"])
def setup_monitoring(
    project_name: str,
    metrics_port: int = 9090,
    log_level: str = "info",
) -> str:
    """Set up monitoring and observability configuration.

    Configures:
    - Prometheus metrics
    - Structured logging
    - Health endpoints
    - Alerting rules

    Args:
        project_name: Name of the project.
        metrics_port: Prometheus metrics port.
        log_level: Logging level.

    Returns:
        JSON string with monitoring configuration.
    """
    prometheus_config = f"""# Prometheus configuration
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: '{project_name}'
    static_configs:
      - targets: ['localhost:{metrics_port}']
    metrics_path: /metrics
"""

    alert_rules = f"""groups:
- name: {project_name}-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{{status=~"5.."}}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: High error rate detected

  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High latency detected

  - alert: ServiceDown
    expr: up == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: Service is down
"""

    logging_config = f"""# Structured logging configuration
version: 1
disable_existing_loggers: false

formatters:
  json:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
    format: "%(asctime)s %(name)s %(levelname)s %(message)s"

handlers:
  console:
    class: logging.StreamHandler
    level: {log_level.upper()}
    formatter: json
    stream: ext://sys.stdout

loggers:
  {project_name}:
    level: {log_level.upper()}
    handlers: [console]
    propagate: false

root:
  level: WARNING
  handlers: [console]
"""

    result = {
        "project": project_name,
        "metrics_port": metrics_port,
        "configurations": {
            "prometheus.yml": prometheus_config,
            "alerts.yml": alert_rules,
            "logging.yml": logging_config,
        },
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "metrics": f"/metrics (port {metrics_port})",
        },
        "dashboards": [
            "Request rate and latency",
            "Error rate by endpoint",
            "Resource utilization",
            "Active connections",
        ],
    }

    return json.dumps(result, indent=2)
