.PHONY: setup build up down logs clean test inspect

# Default target
all: setup build up

# Create directories and environment
setup:
	@echo "Setting up MIDAS..."
	@mkdir -p data/raw data/clean data/features
	@mkdir -p logs/collector logs/processor logs/features
	@test -f .env || cp .env.example .env
	@echo "Setup complete!"

# Build all Docker images
build:
	@echo "Building Docker images..."
	docker compose build

# Start all services
up:
	@echo "Starting MIDAS pipeline..."
	docker compose up -d
	@echo "Pipeline started! Use 'make logs' to view output."

# Stop all services
down:
	@echo "Stopping MIDAS pipeline..."
	docker compose down

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
	docker compose down -v --rmi local
	@echo "Clean complete!"

# Inspect collected data
inspect:
	@python scripts/inspect_data.py

# Status of services
status:
	docker compose ps

# Shell into containers
shell-collector:
	docker compose exec collector /bin/bash

shell-processor:
	docker compose exec processor /bin/bash

shell-features:
	docker compose exec features /bin/bash

# Help
help:
	@echo "MIDAS - Market Intelligence Data Acquisition System"
	@echo ""
	@echo "Available targets:"
	@echo "  setup          - Create directories and .env file"
	@echo "  build          - Build Docker images"
	@echo "  up             - Start all services"
	@echo "  down           - Stop all services"
	@echo "  restart        - Restart all services"
	@echo "  logs           - View all logs"
	@echo "  logs-collector - View collector logs"
	@echo "  logs-processor - View processor logs"
	@echo "  logs-features  - View features logs"
	@echo "  status         - Show service status"
	@echo "  inspect        - Inspect collected data"
	@echo "  clean-data     - Clean data directories"
	@echo "  clean          - Clean everything"
	@echo "  help           - Show this help"
