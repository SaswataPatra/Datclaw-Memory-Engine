#!/bin/bash
# Start Datclaw LLM Orchestration Service Locally

cd "$(dirname "$0")"

# Load environment variables from parent .env
if [ -f "../.env" ]; then
    export $(grep -v '^#' ../.env | xargs)
fi

echo "🔧 Activating virtual environment..."
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Run: ./setup_dependencies.sh"
    echo "   Or: cd .. && ./setup.sh"
    exit 1
fi

source .venv/bin/activate

echo "🔍 Checking prerequisites..."

# Check Docker services
if ! docker ps | grep -q "datclaw-redis\|datclaw-arangodb\|datclaw-qdrant"; then
    echo "⚠️  Warning: Docker services not running!"
    echo "   Start them with: cd .. && docker-compose up -d"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set!"
    echo "   Add it to ../.env file"
    echo "   Or use Ollama (free): Edit config/base.yaml and set provider to 'ollama'"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check ArangoDB password
if [ -z "$ARANGODB_PASSWORD" ]; then
    echo "⚠️  Warning: ARANGODB_PASSWORD not set!"
    echo "   Add it to ../.env file"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "🚀 Starting Datclaw LLM Orchestration Service on port 8000..."
echo "   API: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo "   Press Ctrl+C to stop"
echo ""

uvicorn api.main:app --reload --port 8000
