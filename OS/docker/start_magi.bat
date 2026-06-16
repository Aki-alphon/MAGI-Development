@echo off
:: =============================================================================
:: MAGI OS — Windows Docker Quick-Start
:: Double-click this file OR run from PowerShell to launch the test environment
:: Requires: Docker Desktop for Windows (with Linux containers)
:: =============================================================================

title MAGI OS Docker Launcher

echo.
echo  ███╗   ███╗ █████╗  ██████╗ ██╗
echo  ████╗ ████║██╔══██╗██╔════╝ ██║
echo  ██╔████╔██║███████║██║  ███╗██║
echo  ██║╚██╔╝██║██╔══██║██║   ██║██║
echo  ██║ ╚═╝ ██║██║  ██║╚██████╔╝██║
echo  ╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝
echo  MAGI OS — Docker Test Environment
echo ===============================================

:: Check Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Desktop is not running!
    echo         Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo [OK] Docker is running.
echo.

cd /d "%~dp0.."

echo Choose an option:
echo.
echo  [1] Start ALL services (sensor_hub + magi1 + magi2 + magi3)
echo  [2] Start + open live monitor dashboard
echo  [3] View logs (all services)
echo  [4] Stop all services
echo  [5] Rebuild Docker image
echo  [6] Open bash shell in container
echo  [Q] Quit
echo.

set /p choice="Enter choice: "

if /i "%choice%"=="1" goto START
if /i "%choice%"=="2" goto START_MONITOR
if /i "%choice%"=="3" goto LOGS
if /i "%choice%"=="4" goto STOP
if /i "%choice%"=="5" goto BUILD
if /i "%choice%"=="6" goto SHELL
if /i "%choice%"=="Q" exit /b 0

:START
echo.
echo [MAGI] Starting all services...
docker compose -f docker/docker-compose.yml up -d sensor_hub camera magi1 magi2 magi3
echo.
echo [MAGI] All services started!
echo.
echo  View logs:    docker compose -f docker/docker-compose.yml logs -f
echo  Stop:         docker compose -f docker/docker-compose.yml down
echo  Monitor:      docker compose -f docker/docker-compose.yml run --rm monitor
echo.
pause
goto END

:START_MONITOR
echo.
echo [MAGI] Starting all services + monitor...
docker compose -f docker/docker-compose.yml up -d sensor_hub camera magi1 magi2 magi3
timeout /t 8 /nobreak >nul
docker compose -f docker/docker-compose.yml --profile monitor up monitor
goto END

:LOGS
echo.
echo [MAGI] Streaming logs (Ctrl+C to stop)...
docker compose -f docker/docker-compose.yml logs -f --tail=50
goto END

:STOP
echo.
echo [MAGI] Stopping all MAGI services...
docker compose -f docker/docker-compose.yml down
echo [MAGI] All services stopped.
pause
goto END

:BUILD
echo.
echo [MAGI] Rebuilding Docker image (this may take a few minutes)...
docker compose -f docker/docker-compose.yml build --no-cache
echo [MAGI] Build complete.
pause
goto END

:SHELL
echo.
echo [MAGI] Opening bash shell in MAGI container...
docker compose -f docker/docker-compose.yml run --rm magi1 bash
goto END

:END
echo.
