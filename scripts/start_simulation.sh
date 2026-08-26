#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/ufldv2:$PYTHONPATH"

cd "$PROJECT_ROOT"

python scripts/start_foxglove.py &
FOXGLOVE_PID=$!

sleep 3

python simulation/beamng.py

kill "$FOXGLOVE_PID" 2>/dev/null
wait "$FOXGLOVE_PID" 2>/dev/null