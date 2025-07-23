.PHONY: help install proto lint test clean docker-build docker-up docker-down format type coverage dev-setup verify quick

# Default target
help: ## Show available commands
	@echo "Ω-PHR Framework - Research-grade AI Security Testing"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Setup commands
install: ## Install all dependencies
	@echo "Installing dependencies..."
	python -m pip install --upgrade pip wheel
	pip install -e .[dev]
	@for service in services/*/; do \
		if [ -f "$$service/requirements.txt" ]; then \
			echo "Installing $$service dependencies..."; \
			pip install -r "$$service/requirements.txt"; \
		fi; \
	done
	pre-commit install

install-dev: ## Install development dependencies only
	pip install pytest pytest-asyncio pytest-cov black ruff mypy pre-commit grpcio-tools
	pre-commit install

proto: ## Generate Protocol Buffer stubs
	@echo "Generating protobuf stubs..."
	@for service in services/*/; do \
		if [ -d "$$service/proto" ]; then \
			echo "Generating protos for $$service..."; \
			cd "$$service" && \
			python -m grpc_tools.protoc \
				--proto_path=proto \
				--python_out=. \
				--grpc_python_out=. \
				proto/*.proto && \
			cd - > /dev/null; \
		fi; \
	done

proto-clean: ## Clean generated protobuf files
	find . -name "*_pb2.py" -delete
	find . -name "*_pb2_grpc.py" -delete

# Development commands
lint: ## Run linting with ruff
	@echo "Running ruff linter..."
	ruff check omega_phr/ services/ tests/ libs/

lint-fix: ## Run linting with auto-fix
	ruff check omega_phr/ services/ tests/ libs/ --fix

format: ## Format code with black and ruff
	@echo "Formatting code..."
	ruff format omega_phr/ services/ tests/ libs/
	ruff check omega_phr/ services/ tests/ libs/ --fix --select I

type: ## Run type checking with mypy
	@echo "Running mypy type checker..."
	mypy omega_phr/ --ignore-missing-imports
	@for service in services/*/; do \
		if [ -d "$$service/app" ]; then \
			echo "Type checking $$service..."; \
			mypy "$$service" --ignore-missing-imports || true; \
		fi; \
	done

check: format lint type ## Run all code quality checks

test: ## Run unit tests
	@echo "Running unit tests..."
	PYTHONPATH=. pytest tests/unit/ -v --tb=short

test-quick: ## Run quick tests (exit on first failure)
	PYTHONPATH=. pytest tests/unit/ -x --tb=short

test-all: ## Run all tests
	PYTHONPATH=. pytest tests/ -v

coverage: ## Run tests with coverage report
	PYTHONPATH=. pytest tests/ --cov=omega_phr --cov=services --cov-report=html --cov-report=term

test-unit: ## Run unit tests only
	PYTHONPATH=. pytest tests/unit -v

test-integration: ## Run integration tests
	PYTHONPATH=. pytest tests/integration -v

test-e2e: ## Run end-to-end tests
	PYTHONPATH=. pytest tests/e2e -v

test-fast: ## Run fast unit tests
	PYTHONPATH=. pytest tests/unit -m "not slow" -x

clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/ .coverage htmlcov/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

clean-all: clean proto-clean ## Clean everything including protobuf files

# Docker commands
docker-build: ## Build all Docker images
	@echo "Building Docker images..."
	docker-compose build

docker-up: ## Start all services with docker-compose
	@echo "Starting services..."
	docker-compose up -d

docker-down: ## Stop all services
	@echo "Stopping services..."
	docker-compose down -v

docker-logs: ## View docker-compose logs
	docker-compose logs -f

docker-clean: ## Clean docker images and containers
	docker-compose down -v --rmi all
	docker system prune -f

# Ray commands
ray-start:
	ray start --head --num-cpus=4 --memory=4000000000

ray-stop:
	ray stop

# Kubernetes commands
k8s-deploy:
	kubectl apply -f k8s/

k8s-clean:
	kubectl delete -f k8s/

# Development environment
dev-setup: install proto ## Complete development setup
	@echo "Development environment ready!"

dev-run: ## Start all services in development mode
	@echo "Starting all services in development mode..."
	SCYLLA_OFFLINE=1 USE_SQLITE=1 ./scripts/dev-run.sh || echo "Note: dev-run.sh script not found, use 'make start-services' instead"

start-services: ## Start all services locally
	@echo "Starting services locally..."
	cd services/timeline_lattice && PYTHONPATH=../../ python main.py &
	cd services/hive_orchestrator && PYTHONPATH=../../ python main.py &
	@echo "Services started in background"

stop-services: ## Stop all local services
	pkill -f "python main.py" || echo "No services running"

# Research readiness checks
check-prod:
	ruff check .
	mypy .
	pytest tests/unit tests/integration
	@echo "Research readiness checks passed!"

# Security scans
security-scan:
	pip-audit
	bandit -r services/ libs/

# Monitoring
metrics:
	@echo "Starting Prometheus on http://localhost:9090"
	docker run -d -p 9090:9090 prom/prometheus

# Benchmarking
benchmark:
	python scripts/benchmark.py

# Documentation
docs-build:
	mkdocs build

docs-serve:
	mkdocs serve

# Release commands
release-patch:
	bump2version patch

release-minor:
	bump2version minor

release-major:
	bump2version major
