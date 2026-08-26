@echo off
setlocal

for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"

set "PYTHONPATH=%PROJECT_ROOT%;%PROJECT_ROOT%\ufldv2;%PYTHONPATH%"

cd /d "%PROJECT_ROOT%"

start "Foxglove Bridge" python scripts\start_foxglove.py

timeout /t 3 /nobreak

python simulation\beamng.py

taskkill /FI "WINDOWTITLE eq Foxglove Bridge" /T /F 2>nul

endlocal