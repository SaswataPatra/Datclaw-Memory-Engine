@echo off
REM Datclaw Memory Engine - Windows Setup Script
REM Handles Python detection, virtual environment, and dependencies

setlocal enabledelayedexpansion

echo ================================================================
echo        Datclaw Memory Engine - Setup Script (Windows)
echo ================================================================
echo.

REM ============================================================================
REM Step 1: Check Prerequisites
REM ============================================================================

echo [1/7] Checking prerequisites...

REM Check Docker
where docker >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker not found
    echo.
    echo Please install Docker Desktop for Windows:
    echo   https://www.docker.com/products/docker-desktop
    exit /b 1
)
echo [OK] Docker found

REM Check Docker Compose
docker compose version >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker Compose not found
    exit /b 1
)
echo [OK] Docker Compose found

REM Check Python
set PYTHON_CMD=
for %%p in (python3.12 python3.11 python3.10 python3.9 python3 python) do (
    where %%p >nul 2>nul
    if !ERRORLEVEL! EQU 0 (
        set PYTHON_CMD=%%p
        goto :python_found
    )
)

:python_found
if "%PYTHON_CMD%"=="" (
    echo [ERROR] Python 3.9+ not found
    echo.
    echo Please install Python from https://www.python.org/downloads/
    exit /b 1
)

for /f "tokens=2" %%v in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VERSION=%%v
echo [OK] Python found: %PYTHON_CMD% (%PYTHON_VERSION%)

echo.

REM ============================================================================
REM Step 2: Check Environment Configuration
REM ============================================================================

echo [2/7] Checking environment configuration...

if not exist ".env" (
    echo [WARN] .env file not found, creating from template...
    copy .env.example .env >nul
    echo [OK] Created .env file
    echo.
    echo [IMPORTANT] Edit .env and set:
    echo   - OPENAI_API_KEY=sk-your-key-here
    echo   - ARANGODB_PASSWORD=your-secure-password
    echo.
    pause
) else (
    echo [OK] .env file exists
)

echo.

REM ============================================================================
REM Step 3: Start Docker Services
REM ============================================================================

echo [3/7] Starting Docker services (Redis, ArangoDB, Qdrant)...

docker-compose up -d
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to start Docker services
    exit /b 1
)

echo [OK] Docker services started
echo   Waiting for services to be healthy (30s)...
timeout /t 30 /nobreak >nul

echo.

REM ============================================================================
REM Step 4: Setup Python Backend
REM ============================================================================

echo [4/7] Setting up Python backend...

cd llm-orchestration

REM Create virtual environment
if not exist ".venv" (
    echo   Creating virtual environment...
    %PYTHON_CMD% -m venv .venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment exists
)

REM Activate virtual environment
echo   Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo   Upgrading pip...
python -m pip install --upgrade pip -q

REM Ask about ML installation
echo.
echo Installation Options:
echo   1^) Core only (~200MB, fast^) - Recommended
echo   2^) Core + ML (~2-3GB^) - For local ML features
echo.
set /p INSTALL_TYPE="Choose installation type (1 or 2, default=1): "
if "%INSTALL_TYPE%"=="" set INSTALL_TYPE=1

if "%INSTALL_TYPE%"=="2" (
    echo.
    echo Installing core + ML dependencies (5-10 minutes^)...
    pip install -r requirements.txt
    pip install -r requirements-ml.txt
    
    echo   Downloading spaCy model...
    python -m spacy download en_core_web_sm -q
    
    echo [OK] Core + ML dependencies installed
) else (
    echo.
    echo Installing core dependencies...
    pip install -r requirements.txt
    echo [OK] Core dependencies installed
)

echo.

REM ============================================================================
REM Step 5: Verify Installation
REM ============================================================================

echo [5/7] Verifying installation...

python -c "import fastapi; print('  [OK] FastAPI:', fastapi.__version__)" || exit /b 1
python -c "import openai; print('  [OK] OpenAI:', openai.__version__)" || exit /b 1
python -c "import redis; print('  [OK] Redis:', redis.__version__)" || exit /b 1

echo [OK] Installation verified
echo.

cd ..

REM ============================================================================
REM Step 6: Setup Frontend (Optional)
REM ============================================================================

where node >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [6/7] Frontend setup available
    set /p INSTALL_FRONTEND="Install frontend? (y/N): "
    
    if /i "!INSTALL_FRONTEND!"=="y" (
        cd frontend
        echo   Installing npm dependencies...
        call npm install
        echo [OK] Frontend dependencies installed
        cd ..
    ) else (
        echo [SKIP] Frontend setup skipped
    )
) else (
    echo [6/7] Frontend setup skipped (Node.js not found^)
)

echo.

REM ============================================================================
REM Step 7: Final Instructions
REM ============================================================================

echo [7/7] Setup complete!
echo.
echo ================================================================
echo                    Setup Complete! 🎉
echo ================================================================
echo.
echo Next Steps:
echo.
echo 1. Start the backend:
echo    cd llm-orchestration
echo    start_service.sh
echo.
echo 2. Test the API:
echo    curl http://localhost:8000/health
echo.
echo 3. Try the CLI chat:
echo    cd llm-orchestration
echo    chat.sh
echo.
echo Useful Commands:
echo   Check Docker: docker-compose ps
echo   View logs: docker-compose logs -f
echo   Stop services: docker-compose down
echo.
echo Happy memory engineering! 🧠✨
echo.

pause
