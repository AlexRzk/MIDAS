.PHONY: setup build up down logs clean test rebuild preflight validate-features

# Default target
all: preflight setup build up

# Preflight check - ensure directories exist
preflight:
	@echo "Running preflight checks..."
	@mkdir -p data/raw data/clean data/features models reports logs/collector logs/processor logs/features
	@echo "Preflight complete!"

# Create directories and environment
setup: preflight
	@test -f .env || cp .env.example .env
	@echo "Setup complete!"

# Build all Docker images
build:
	docker compose build

build-collector:
	docker compose build collector

build-processor:
	docker compose build processor

build-features:
	docker compose build features

build-tools:
	docker compose build tools

build-training:
	docker compose -f docker-compose.training.yml build training

# Rebuild without cache
rebuild:
	docker compose build --no-cache

# Start all services
up: preflight
	docker compose up -d
	@echo "Pipeline started! Use 'make logs' to view output."

# Stop all services
down:
	docker compose down

down-all:
	docker compose down
	docker compose -f docker-compose.training.yml down

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
	docker compose down -v --rmi local
	docker compose -f docker-compose.training.yml down -v --rmi local
	@echo "Clean complete!"

# Run tests
test:
	docker compose run --rm tools bash -lc "pip install -r features/requirements.txt && python -m pytest tests/ -v"

# Validate feature Parquet files
validate-features:
	@echo "Validating feature Parquet files..."
	@docker compose build tools >/dev/null 2>&1 || true
	@docker compose run --rm tools bash -lc "pip install -r features/requirements.txt && python scripts/validate_features.py --dir /app/data/features --output /app/reports/validation_report.json"

# Training
train:
	docker compose -f docker-compose.training.yml up training

train-interactive:
	docker compose -f docker-compose.training.yml run --rm training bash

# Backtesting
backtest:
	docker compose -f docker-compose.training.yml run --rm backtest

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

shell-tools:
	docker compose run --rm tools bash

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
	@echo "Setup:"
	@echo "  setup          - Create directories and .env file"
	@echo "  build          - Build Docker images"
	@echo "  up             - Start data collection pipeline"
	@echo "  down           - Stop all services"
	@echo "  status         - Show service status"
	@echo ""
	@echo "Development:"
	@echo "  test           - Run tests"
	@echo "  validate-features - Validate feature Parquet files"
	@echo "  stats          - Show data file stats"
	@echo "  clean-data     - Clean data directories"
	@echo "  clean          - Clean everything"
	@echo ""
	@echo "Training:"
	@echo "  train          - Train TFT model"
	@echo "  train-interactive - Interactive training shell"
	@echo "  backtest       - Run backtest"
	@echo ""
	@echo "Logs:"
	@echo "  logs           - View all logs"
	@echo "  logs-<service> - View specific service logs"

