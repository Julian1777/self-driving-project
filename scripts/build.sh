#!/bin/bash

echo "Building Docker containers"
echo ""

# Always run from the docker directory for correct context
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/docker"

docker compose build

if [ $? -eq 0 ]; then
    echo ""
    echo "All containers built successfully!"
    echo ""
    echo "Run './start_services.sh' to start the services from the project root."
else
    echo ""
    echo "Build failed"
    exit 1
fi
