#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

echo "Stopping voice assistant..."
docker compose down

if [ $? -eq 0 ]; then
    echo "Stopped. Restart with ./start.sh"
else
    echo "Failed to stop. Check: docker ps"
    exit 1
fi