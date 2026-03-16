#!/bin/bash
set -e

echo "Setting up Datclaw Memory Engine dependencies..."

if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing core dependencies..."
pip install -r requirements.txt

# Check if ML install is requested
if [ "$1" = "--ml" ] || [ "$1" = "--full" ]; then
    echo ""
    echo "Installing ML dependencies (this may take a while)..."
    pip install -r requirements-ml.txt
    echo "Downloading spacy English language model..."
    python -m spacy download en_core_web_sm
fi

echo ""
echo "Verifying installation..."
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
python -c "import pydantic; print(f'Pydantic: {pydantic.__version__}')"
python -c "import openai; print(f'OpenAI: {openai.__version__}')"

if [ "$1" = "--ml" ] || [ "$1" = "--full" ]; then
    python -c "import spacy; print(f'spaCy: {spacy.__version__}')" 2>/dev/null || echo "spaCy: not installed"
    python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch: not installed"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Usage:"
echo "  ./setup_dependencies.sh        # Core only (~200MB, fast)"
echo "  ./setup_dependencies.sh --ml    # Core + ML (~2-3GB, includes PyTorch/spaCy)"
echo ""
echo "Start the service with: ./start_service.sh"
