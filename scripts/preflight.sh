#!/usr/bin/env bash
set -euo pipefail

echo "MIDAS Preflight Check"
echo "Checking data directories and Docker setup..."

ROOT_DIR=$(cd "$(dirname "$0")/.." >/dev/null 2>&1 && pwd)
echo "Project root: $ROOT_DIR"

MISSING=0
for d in data/raw data/clean data/features logs/collector logs/processor logs/features; do
    if [ ! -d "$ROOT_DIR/$d" ]; then
        echo "⚠️  Missing: $ROOT_DIR/$d"
        MISSING=1
    else
        echo "✅ Found: $ROOT_DIR/$d"
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo "\nOne or more directories are missing. Run the setup script to create them:" \
         "\n  chmod +x scripts/setup.sh && ./scripts/setup.sh"
    exit 2
fi

echo "Checking Docker availability..."
if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker CLI can't connect to the Docker daemon. Ensure Docker is installed and running."
    exit 3
fi

echo "Checking docker-compose file for ${PWD} device mapping..."
if grep -q "device: \${PWD}/data" docker-compose.yml; then
    echo "Note: docker-compose.yml uses ${PWD}/data/* bind mounts. Make sure to run compose from the project root and not via symlinked path."
fi

echo "Preflight checks passed. You can run: docker compose up -d"
exit 0
