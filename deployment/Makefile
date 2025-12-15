.PHONY: help install dev run docker-build docker-run docker-stop test lint format clean

# Default target
help:
	@echo "LangChain Platform - Available Commands"
	@echo "========================================"
	@echo ""
	@echo "Setup:"
	@echo "  install      Install production dependencies"
	@echo "  dev          Install development dependencies"
	@echo ""
	@echo "Run:"
	@echo "  run          Run the server locally (requires .env)"
	@echo "  run-reload   Run with auto-reload for development"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run with Docker Compose"
	@echo "  docker-stop  Stop Docker containers"
	@echo "  docker-logs  View container logs"
	@echo ""
	@echo "Quality:"
	@echo "  test         Run tests"
	@echo "  lint         Run linter"
	@echo "  format       Format code"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean        Remove build artifacts"

# Setup
install:
	uv pip install -e .

dev:
	uv pip install -e ".[dev]"

# Run
run:
	@if [ ! -f .env ]; then echo "Error: .env file not found. Copy .env.example to .env"; exit 1; fi
	uv run uvicorn app.server:app --host 0.0.0.0 --port 8000

run-reload:
	@if [ ! -f .env ]; then echo "Error: .env file not found. Copy .env.example to .env"; exit 1; fi
	uv run uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload

# Docker
docker-build:
	docker build -t langchain-platform:latest .

docker-run:
	@if [ ! -f .env ]; then echo "Error: .env file not found. Copy .env.example to .env"; exit 1; fi
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Quality
test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .
	uv run mypy app/

format:
	uv run ruff format .
	uv run ruff check --fix .

# Cleanup
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf *.egg-info build dist
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
