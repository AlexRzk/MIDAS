#!/bin/bash
# MIDAS Pipeline Setup Script
# Run this script to initialize the project

set -e

echo "==================================="
echo "MIDAS - Market Intelligence Data Acquisition System"
echo "==================================="

# Create data directories
echo "Creating data directories..."
mkdir -p data/raw data/clean data/features
mkdir -p logs/collector logs/processor logs/features

# Copy environment file if not exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env to configure your settings"
fi

# Set permissions
echo "Setting permissions..."
chmod -R 755 data logs

echo ""
echo "Setup complete!"
echo ""
echo "To start the pipeline:"
echo "  docker compose up -d"
echo ""
echo "To view logs:"
echo "  docker compose logs -f collector"
echo "  docker compose logs -f processor"
echo "  docker compose logs -f features"
echo ""
echo "To stop the pipeline:"
echo "  docker compose down"
echo ""
