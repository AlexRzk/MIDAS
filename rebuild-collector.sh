#!/bin/bash
# Rebuild and restart collector service

echo "Fixing data directory permissions..."
chmod -R 777 ./data/raw ./data/clean ./data/features 2>/dev/null || true

echo "Stopping collector..."
docker compose down collector

echo "Rebuilding collector..."
docker compose build --no-cache collector

echo "Starting collector..."
docker compose up -d collector

echo "Waiting 5 seconds..."
sleep 5

echo "Checking logs..."
docker logs midas-collector --tail 30
