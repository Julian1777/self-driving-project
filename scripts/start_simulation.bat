@echo off
setlocal enabledelayedexpansion

REM Get the project root directory (parent of scripts directory)
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"

REM Set PYTHONPATH
set "PYTHONPATH=%PROJECT_ROOT%;%PROJECT_ROOT%\ufldv2;%PYTHONPATH%"

REM Change to project root
cd /d "%PROJECT_ROOT%"

REM Start Foxglove bridge server in background
start "Foxglove Bridge" python -c "^
import sys; ^
sys.path.insert(0, '%PROJECT_ROOT%'); ^
from simulation.foxglove_integration.bridge_instance import bridge; ^
import time; ^
try: ^
    bridge.start_server(); ^
    bridge.initialize_channels(); ^
    print('Foxglove ready - ws://localhost:8765'); ^
    time.sleep(2); ^
except Exception as e: ^
    print(f'Error: {e}', file=sys.stderr); ^
    sys.exit(1); ^
"

REM Wait for Foxglove to start
timeout /t 3 /nobreak

REM Run the main simulation
python simulation/beamng.py

REM Kill the Foxglove bridge (using taskkill to find process by window title)
taskkill /FI "WINDOWTITLE eq Foxglove Bridge" /T /F 2>nul

endlocal