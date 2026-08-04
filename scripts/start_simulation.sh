#!/bin/bash

# Get the project root directory (parent of scripts directory)
PROJECT_ROOT="$(cd "$(dirname "$0")" && cd .. && pwd)"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/ufldv2:$PYTHONPATH"

cd "$PROJECT_ROOT"

python -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')

from simulation.foxglove_integration.bridge_instance import bridge
import time

try:
    bridge.start_server()
    bridge.initialize_channels()
    print('Foxglove ready - ws://localhost:8765')
    time.sleep(2)
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
" &

FOXGLOVE_PID=$!

sleep 3

python simulation/beamng.py

kill $FOXGLOVE_PID 2>/dev/null
wait $FOXGLOVE_PID 2>/dev/null
