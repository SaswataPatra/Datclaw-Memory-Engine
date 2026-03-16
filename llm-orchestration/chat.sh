#!/bin/bash
# Datclaw Chatbot CLI Launcher

cd "$(dirname "$0")"

# Load environment variables from parent .env
if [ -f "../.env" ]; then
    export $(grep -v '^#' ../.env | xargs)
fi

# Detect Python executable
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python not found. Please install Python 3.9+"
    exit 1
fi

# Activate venv if it exists
if [ -d ".venv" ]; then
    echo "🔧 Activating virtual environment..."
    source .venv/bin/activate
    # After activation, use 'python' directly (from venv)
    PYTHON_CMD="python"
elif [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
    PYTHON_CMD="python"
fi

# Check if dependencies are installed
$PYTHON_CMD -c "import rich, httpx" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing dependencies (this may take a minute)..."
    $PYTHON_CMD -m pip install --quiet rich httpx || {
        echo "⚠️  Warning: Failed to install dependencies. Trying anyway..."
    }
fi

# Launch CLI
echo "🚀 Launching Datclaw Chatbot CLI..."
echo "   Make sure the backend is running (./start_service.sh)"
echo ""
$PYTHON_CMD cli/chat_cli.py "$@"

