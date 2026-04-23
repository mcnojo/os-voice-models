#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

echo "Starting voice assistant..."

mkdir -p logs temp models templates static

echo "Building Docker container..."
docker compose build --no-cache

echo "Starting services..."
docker compose up -d

sleep 3

if docker compose ps | grep -q "Up"; then
    echo "Voice assistant is running at http://localhost:5001"
    echo "Logs:  docker compose logs -f"
    echo "Stop:  ./stop.sh"
else
    echo "Failed to start. Check logs: docker compose logs"
    exit 1
fi