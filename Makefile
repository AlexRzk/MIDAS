.PHONY: setup build up down logs clean test inspect lint validate rebuild preflight

# Default target
all: preflight setup build up

# Preflight check - ensure directories exist
preflight:
	@echo "Running preflight checks..."
	@mkdir -p data/raw data/clean data/features
	@mkdir -p logs/collector logs/processor logs/features
	@mkdir -p prometheus grafana/dashboards
	@echo "Preflight complete!"

# Create directories and environment
setup: preflight
	@echo "Setting up MIDAS..."
	@test -f .env || cp .env.example .env
	@echo "Setup complete!"

# Build all Docker images
build:
	@echo "Building Docker images..."
	docker compose build

# Build specific service
build-collector:
	docker compose build collector

build-processor:
	docker compose build processor

build-features:
	docker compose build features

# Rebuild only changed services
rebuild:
	@echo "Rebuilding changed services..."
	docker compose build --no-cache

# Start all services
up: preflight
	@echo "Starting MIDAS pipeline..."
	docker compose up -d
	@echo "Pipeline started! Use 'make logs' to view output."

# Start with monitoring stack
up-monitoring: preflight
	@echo "Starting MIDAS pipeline with monitoring..."
	docker compose --profile monitoring up -d
	@echo "Pipeline and monitoring started!"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana:    http://localhost:3000"

# Stop all services
down:
	@echo "Stopping MIDAS pipeline..."
	docker compose down

# Stop including monitoring
down-all:
	@echo "Stopping all services including monitoring..."
	docker compose --profile monitoring down

# View logs
logs:
	docker compose logs -f

logs-collector:
	docker compose logs -f collector

logs-processor:
	docker compose logs -f processor

logs-features:
	docker compose logs -f features

# Restart services
restart:
	docker compose restart

restart-collector:
	docker compose restart collector

restart-processor:
	docker compose restart processor

restart-features:
	docker compose restart features

# Clean data directories
clean-data:
	@echo "Cleaning data directories..."
	@rm -rf data/raw/* data/clean/* data/features/*
	@rm -f data/clean/.processed_files data/features/.processed_files
	@echo "Data cleaned!"

# Clean everything including Docker
clean: clean-data
	@echo "Cleaning Docker resources..."
	docker compose --profile monitoring down -v --rmi local
	@echo "Clean complete!"

# Inspect collected data
inspect:
	@python scripts/inspect_data.py

# Run tests
test:
	@echo "Running processor tests..."
	cd processor && python -m pytest tests/ -v
	@echo "Running features tests..."
	cd features && python -m pytest tests/ -v

test-processor:
	cd processor && python -m pytest tests/ -v

test-features:
	cd features && python -m pytest tests/ -v

# Lint code
lint:
	@echo "Linting Python code..."
	cd processor && python -m ruff check .
	cd features && python -m ruff check .
	@echo "Linting Rust code..."
	cd collector && cargo clippy -- -D warnings

lint-fix:
	@echo "Fixing Python lint issues..."
	cd processor && python -m ruff check . --fix
	cd features && python -m ruff check . --fix
	@echo "Fixing Rust lint issues..."
	cd collector && cargo clippy --fix --allow-dirty

# Format code
fmt:
	@echo "Formatting Python code..."
	cd processor && python -m ruff format .
	cd features && python -m ruff format .
	@echo "Formatting Rust code..."
	cd collector && cargo fmt

# Validate data quality
validate:
	@echo "Validating data quality..."
	@python scripts/validate_data.py

# Schema validation
validate-schema:
	@echo "Validating Parquet schemas..."
	@python scripts/validate_schema.py

# Status of services
status:
	docker compose ps
	@echo ""
	@echo "Health status:"
	@docker inspect --format='{{.Name}}: {{.State.Health.Status}}' midas-collector 2>/dev/null || echo "collector: not running"
	@docker inspect --format='{{.Name}}: {{.State.Health.Status}}' midas-processor 2>/dev/null || echo "processor: not running"
	@docker inspect --format='{{.Name}}: {{.State.Health.Status}}' midas-features 2>/dev/null || echo "features: not running"

# Shell into containers
shell-collector:
	docker compose exec collector /bin/bash

shell-processor:
	docker compose exec processor /bin/bash

shell-features:
	docker compose exec features /bin/bash

# Development helpers
dev-processor:
	@echo "Starting processor in dev mode..."
	cd processor && python -m processor.main

dev-features:
	@echo "Starting features in dev mode..."
	cd features && python -m features.main

# Data stats
stats:
	@echo "=== Raw Data ==="
	@ls -lh data/raw/*.jsonl.zst 2>/dev/null || echo "No raw data files"
	@echo ""
	@echo "=== Clean Data ==="
	@ls -lh data/clean/*.parquet 2>/dev/null || echo "No clean data files"
	@echo ""
	@echo "=== Feature Data ==="
	@ls -lh data/features/*.parquet 2>/dev/null || echo "No feature data files"

# Help
help:
	@echo "MIDAS - Market Intelligence Data Acquisition System"
	@echo ""
	@echo "Core targets:"
	@echo "  setup          - Create directories and .env file"
	@echo "  build          - Build Docker images"
	@echo "  up             - Start all services"
	@echo "  up-monitoring  - Start with Prometheus & Grafana"
	@echo "  down           - Stop all services"
	@echo "  restart        - Restart all services"
	@echo "  status         - Show service status and health"
	@echo ""
	@echo "Logs:"
	@echo "  logs           - View all logs"
	@echo "  logs-collector - View collector logs"
	@echo "  logs-processor - View processor logs"
	@echo "  logs-features  - View features logs"
	@echo ""
	@echo "Development:"
	@echo "  test           - Run all tests"
	@echo "  lint           - Lint all code"
	@echo "  lint-fix       - Fix lint issues"
	@echo "  fmt            - Format all code"
	@echo "  rebuild        - Rebuild changed services"
	@echo ""
	@echo "Data:"
	@echo "  inspect        - Inspect collected data"
	@echo "  validate       - Validate data quality"
	@echo "  stats          - Show data file stats"
	@echo "  clean-data     - Clean data directories"
	@echo "  clean          - Clean everything"
