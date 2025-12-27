# IntentForge - NLP-driven Code Generation Framework
# Production-ready Makefile with full lifecycle management

.PHONY: all install dev test lint format security validate docs build publish clean docker help

# Configuration
PYTHON := python3
PIP := pip3
PROJECT := intentforge
VERSION := $(shell cat VERSION 2>/dev/null || echo "0.1.0")
DOCKER_IMAGE := $(PROJECT):$(VERSION)

# Colors
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m

# Default target
all: install lint test validate build

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INSTALLATION
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

install: ## Install production dependencies
	@echo "$(GREEN)Installing production dependencies...$(NC)"
	$(PIP) install -e .

dev: ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	$(PIP) install -e ".[dev,test,docs,server]"
	pre-commit install

deps-update: ## Update all dependencies
	$(PIP) install --upgrade pip
	$(PIP) install pip-tools
	pip-compile requirements.in -o requirements.txt
	pip-compile requirements-dev.in -o requirements-dev.txt

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QUALITY ASSURANCE
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

lint: ## Run all linters
	@echo "$(GREEN)Running linters...$(NC)"
	ruff check $(PROJECT) tests
	mypy $(PROJECT) --ignore-missing-imports

format: ## Format code
	@echo "$(GREEN)Formatting code...$(NC)"
	ruff format $(PROJECT) tests
	ruff check --fix $(PROJECT) tests

security: ## Run security checks
	@echo "$(YELLOW)Running security analysis...$(NC)"
	bandit -r $(PROJECT) -ll
	safety check
	semgrep --config=auto $(PROJECT)

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTING & VALIDATION
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

test: ## Run tests with coverage
	@echo "$(GREEN)Running tests...$(NC)"
	pytest tests/ -v --cov=$(PROJECT) --cov-report=html --cov-report=term-missing

test-fast: ## Run tests without coverage
	pytest tests/ -v -x --tb=short

test-integration: ## Run integration tests
	pytest tests/integration/ -v --timeout=60

validate: ## Validate generated code samples
	@echo "$(GREEN)Validating code generation...$(NC)"
	$(PYTHON) -m $(PROJECT).cli validate-samples
	$(PYTHON) -m $(PROJECT).cli validate-schemas

benchmark: ## Run performance benchmarks
	@echo "$(GREEN)Running benchmarks...$(NC)"
	pytest tests/benchmarks/ -v --benchmark-only --benchmark-json=benchmark.json

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BUILD & PACKAGE
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

build: clean ## Build distribution packages
	@echo "$(GREEN)Building packages...$(NC)"
	$(PYTHON) -m build

publish-test: build ## Publish to TestPyPI
	twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	twine upload dist/*

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DOCKER
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE) .
	docker tag $(DOCKER_IMAGE) $(PROJECT):latest

docker-run: ## Run Docker container
	docker run -it --rm \
		-p 8000:8000 \
		-p 1883:1883 \
		-v $(PWD)/.env:/app/.env:ro \
		$(DOCKER_IMAGE)

docker-compose-up: ## Start all services
	docker-compose up -d

docker-compose-down: ## Stop all services
	docker-compose down -v

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CODE GENERATION & PATTERNS
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

generate-crud: ## Generate CRUD pattern for a table (usage: make generate-crud TABLE=users)
	@echo "$(GREEN)Generating CRUD pattern for $(TABLE)...$(NC)"
	$(PYTHON) -c "from $(PROJECT).patterns import FullstackPatterns, PatternConfig, PatternType; \
		config = PatternConfig(PatternType.CRUD_API, '$(TABLE)'); \
		result = FullstackPatterns.form_to_database(config); \
		import os; os.makedirs('generated/$(TABLE)', exist_ok=True); \
		[open(f'generated/$(TABLE)/{k}.py', 'w').write(v) for k, v in result.items()]"
	@echo "$(GREEN)Generated files in generated/$(TABLE)/$(NC)"

generate-form: ## Generate form-to-database pattern
	@echo "$(GREEN)Generating form pattern...$(NC)"
	$(PYTHON) -m $(PROJECT).cli generate-form --output generated/

validate-schema: ## Validate all JSON schemas
	@echo "$(GREEN)Validating schemas...$(NC)"
	$(PYTHON) -c "from $(PROJECT).schema_registry import get_registry; \
		r = get_registry(); \
		print('Schema registry loaded successfully'); \
		print(f'Available schemas: {list(r._schemas.keys())}')"

validate-code: ## Validate generated code sample
	@echo "$(GREEN)Validating code samples...$(NC)"
	$(PYTHON) -c "from $(PROJECT).validator import CodeValidator; \
		v = CodeValidator(); \
		result = v.validate_sync('print(\"hello\")', 'python'); \
		print(f'Valid: {result.is_valid}')"

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENVIRONMENT & CONFIGURATION
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

env-init: ## Generate .env.example template
	@echo "$(GREEN)Generating .env.example...$(NC)"
	$(PYTHON) -c "from $(PROJECT).env_handler import EnvHandler; \
		EnvHandler().generate_template('.env.example')"
	@echo "$(GREEN)Generated .env.example - copy to .env and configure$(NC)"

env-check: ## Validate environment configuration
	@echo "$(GREEN)Checking environment...$(NC)"
	@test -f .env || (echo "$(RED)ERROR: .env file not found. Run 'make env-init' first$(NC)" && exit 1)
	$(PYTHON) -c "from $(PROJECT).env_handler import get_env; \
		env = get_env(); \
		print('Environment loaded successfully'); \
		print(f'Database URL: {env.get_database_url()}')"

env-show: ## Show current environment (masked)
	@echo "$(YELLOW)Current environment configuration:$(NC)"
	$(PYTHON) -c "from $(PROJECT).env_handler import get_env; \
		env = get_env(); \
		for k, v in sorted(env.to_dict().items()): \
			masked = v[:3] + '***' if 'KEY' in k or 'PASSWORD' in k or 'SECRET' in k else v; \
			print(f'  {k}={masked}')"

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATABASE
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

db-init: ## Initialize database
	@echo "$(GREEN)Initializing database...$(NC)"
	alembic upgrade head

db-migrate: ## Create new migration (usage: make db-migrate MSG="add users table")
	alembic revision --autogenerate -m "$(MSG)"

db-upgrade: ## Apply migrations
	alembic upgrade head

db-downgrade: ## Rollback last migration
	alembic downgrade -1

db-reset: ## Reset database (DANGER!)
	@echo "$(RED)WARNING: This will delete all data!$(NC)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	alembic downgrade base
	alembic upgrade head

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEVELOPMENT TOOLS
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

run-server: ## Run development server
	$(PYTHON) -m $(PROJECT).server --reload --host 0.0.0.0 --port 8000

run-broker: ## Run MQTT broker
	mosquitto -c mosquitto.conf

run-worker: ## Run background worker
	$(PYTHON) -m $(PROJECT).worker

run-all: ## Run all services (server + broker + worker)
	@echo "$(GREEN)Starting all services...$(NC)"
	docker-compose up -d mqtt redis
	$(PYTHON) -m $(PROJECT).worker &
	$(PYTHON) -m $(PROJECT).server --reload --host 0.0.0.0 --port 8000

shell: ## Open interactive shell
	$(PYTHON) -m $(PROJECT).shell

repl: ## Open Python REPL with project loaded
	$(PYTHON) -c "from $(PROJECT) import *; \
		from $(PROJECT).core import *; \
		from $(PROJECT).patterns import *; \
		print('IntentForge loaded. Available: IntentForge, Intent, FullstackPatterns')" -i

docs: ## Generate documentation
	mkdocs build
	
docs-serve: ## Serve documentation locally
	mkdocs serve

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QUICK START
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

quickstart: ## Complete setup for new developers
	@echo "$(GREEN)Setting up IntentForge...$(NC)"
	$(MAKE) dev
	$(MAKE) env-init
	@cp .env.example .env 2>/dev/null || true
	@echo "$(YELLOW)Please edit .env with your configuration$(NC)"
	$(MAKE) validate-schema
	@echo "$(GREEN)Setup complete! Run 'make run-server' to start.$(NC)"

demo: ## Run demo with sample form-to-database
	@echo "$(GREEN)Running demo...$(NC)"
	$(PYTHON) -c "\
from $(PROJECT).patterns import FullstackPatterns, PatternConfig, PatternType; \
config = PatternConfig( \
    PatternType.FORM_TO_DATABASE, \
    'contacts', \
    fields=[ \
        {'name': 'name', 'type': 'text', 'required': True}, \
        {'name': 'email', 'type': 'email', 'required': True}, \
        {'name': 'message', 'type': 'textarea'} \
    ] \
); \
result = FullstackPatterns.form_to_database(config); \
print('=== Generated API Endpoint ==='); \
print(result['backend_api'][:2000]); \
print('\\n=== Generated Frontend Form ==='); \
print(result['frontend_html'][:1000])"

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLEANUP
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

clean: ## Clean build artifacts
	@echo "$(RED)Cleaning...$(NC)"
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-all: clean ## Clean everything including venv
	rm -rf .venv/

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELP
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

help: ## Show this help
	@echo "IntentForge - NLP-driven Code Generation Framework"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
