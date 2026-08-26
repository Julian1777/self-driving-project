import sys
import os
import time

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, project_root)

from simulation.foxglove_integration.bridge_instance import bridge

try:
    bridge.start_server()
    bridge.initialize_channels()
    print("Foxglove ready - ws://localhost:8765")
    time.sleep(2)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)