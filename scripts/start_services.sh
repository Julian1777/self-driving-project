#!/bin/bash

echo "Starting Docker services"
echo ""

cd "$(dirname "$0")/docker"

docker compose up -d

echo ""
echo "Waiting for services to initialize"
sleep 10

echo ""
echo "Checking service health"
echo ""

services=(
    "lane_detection:4777"
    "object_detection:5777"
    "traffic_light_detection:6777"
    "sign_detection:7777"
    "sign_classification:8777"
)

all_healthy=true

for service_port in "${services[@]}"; do
    service_name="${service_port%:*}"
    port="${service_port##*:}"
    
    if curl -f http://localhost:$port/health >/dev/null 2>&1; then
        echo "$service_name (localhost:$port)"
    else
        echo "$service_name (localhost:$port)"
        all_healthy=false
    fi
done

echo ""

if [ "$all_healthy" = true ]; then
    echo "All services healthy!"
    echo ""
    echo "Next step: Run './start_simulation.sh' to start BeamNG + Foxglove"
else
    echo "Some services failed to start"
    echo ""
    echo "Check logs: docker compose logs -f"
    exit 1
fi
