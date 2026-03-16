#!/bin/bash
# Start DAPPY LLM Orchestration Service Locally

cd "$(dirname "$0")"

echo "🔧 Activating virtual environment..."
source .venv/bin/activate

echo "🔑 Checking for API key..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set!"
    echo "   Set it with: export OPENAI_API_KEY='sk-...'"
    echo "   Or use Ollama (free): Edit config/base.yaml and set provider to 'ollama'"
    echo ""
fi

echo "🚀 Starting DAPPY LLM Orchestration Service on port 8000..."
echo "   Press Ctrl+C to stop"
echo ""

uvicorn api.main:app --reload --port 8000
