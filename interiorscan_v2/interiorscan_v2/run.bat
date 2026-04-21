@echo off
REM Interior Mapping Scanner v2.0 - Windows Launch Script

echo ==========================================
echo Interior Mapping Scanner v2.0
echo Advanced Edition with Anomaly Detection
echo ==========================================
echo.

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3 and try again
    pause
    exit /b 1
)

echo Step 1: Running advanced system scan...
echo.
echo Scanning for:
echo   * Processes (with capabilities)
echo   * Memory regions (VMA analysis)
echo   * Network connections (enhanced)
echo   * File descriptors
echo   * Namespaces
echo   * Anomalies
echo.

REM Run the scanner
cd backend
python scanner_v2.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Scanner failed. You may need administrator privileges.
    echo Try running this script as Administrator.
    pause
    exit /b 1
)

echo.
echo Scan complete!
echo.
echo Step 2: Starting visualization server...
echo.

REM Start the web server
cd ..\frontend
echo ==========================================
echo   Server running at:
echo      http://localhost:8000
echo ==========================================
echo.
echo Features:
echo   * 3D interactive graph visualization
echo   * Real-time anomaly detection
echo   * Advanced security analysis
echo   * Auto-refresh monitoring
echo   * Full-text search
echo   * Metrics dashboard
echo.
echo Press Ctrl+C to stop the server
echo.

python -m http.server 8000
