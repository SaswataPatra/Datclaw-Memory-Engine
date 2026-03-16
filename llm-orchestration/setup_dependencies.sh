#!/bin/bash
# Setup script for Datclaw Memory Engine dependencies

set -e  # Exit on error

echo "🔧 Setting up Datclaw Memory Engine dependencies..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please create it first:"
    echo "   python -m venv .venv"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Download spacy language model
echo "🌐 Downloading spacy English language model..."
python -m spacy download en_core_web_sm

# Verify installation
echo ""
echo "✅ Verifying installation..."
python -c "import spacy; print(f'Spacy version: {spacy.__version__}')"
python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"
python -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')"

echo ""
echo "✨ Setup complete! You can now start the service with:"
echo "   ./start_service.sh"
