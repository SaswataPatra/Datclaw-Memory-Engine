#!/bin/bash
# Datclaw Memory Engine - Universal Setup Script
# Works on macOS, Linux, and Windows (via Git Bash/WSL)
# Handles Python detection, virtual environment, dependencies, and database initialization

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect OS
OS="unknown"
case "$(uname -s)" in
    Darwin*)    OS="macOS";;
    Linux*)     OS="Linux";;
    MINGW*|MSYS*|CYGWIN*)  OS="Windows";;
esac

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        Datclaw Memory Engine - Setup Script               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Detected OS: $OS${NC}"
echo ""

# ============================================================================
# Step 1: Check Prerequisites
# ============================================================================

echo -e "${BLUE}[1/7] Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not found${NC}"
    echo ""
    echo "Please install Docker first:"
    echo "  macOS/Windows: https://www.docker.com/products/docker-desktop"
    echo "  Linux: https://docs.docker.com/engine/install/"
    exit 1
fi
echo -e "${GREEN}✓ Docker found: $(docker --version | cut -d' ' -f3 | cut -d',' -f1)${NC}"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}✗ Docker Compose not found${NC}"
    echo "Please install Docker Compose"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose found${NC}"

# Check Python (try multiple versions)
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3.9 python3 python; do
    if command -v $cmd &> /dev/null; then
        VERSION=$($cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        MAJOR=$(echo $VERSION | cut -d. -f1)
        MINOR=$(echo $VERSION | cut -d. -f2)
        
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 9 ]; then
            PYTHON_CMD=$cmd
            echo -e "${GREEN}✓ Python found: $cmd ($VERSION)${NC}"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}✗ Python 3.9+ not found${NC}"
    echo ""
    echo "Please install Python 3.9 or higher:"
    echo "  macOS: brew install python@3.12"
    echo "  Linux: sudo apt install python3.12 python3.12-venv"
    echo "  Windows: https://www.python.org/downloads/"
    exit 1
fi

# Check if venv module is available
$PYTHON_CMD -m venv --help &> /dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Python venv module not available${NC}"
    echo ""
    echo "Please install the venv module:"
    echo "  Ubuntu/Debian: sudo apt install python3-venv"
    echo "  Fedora/RHEL: sudo dnf install python3-virtualenv"
    exit 1
fi
echo -e "${GREEN}✓ Python venv module available${NC}"

# Check Node.js (optional, for frontend)
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version | grep -oE '[0-9]+' | head -1)
    if [ "$NODE_VERSION" -ge 18 ]; then
        echo -e "${GREEN}✓ Node.js found: $(node --version) (for frontend)${NC}"
        HAS_NODE=true
    else
        echo -e "${YELLOW}⚠ Node.js version too old (need 18+), frontend will be skipped${NC}"
        HAS_NODE=false
    fi
else
    echo -e "${YELLOW}⚠ Node.js not found, frontend will be skipped${NC}"
    HAS_NODE=false
fi

echo ""

# ============================================================================
# Step 2: Check Environment Configuration
# ============================================================================

echo -e "${BLUE}[2/7] Checking environment configuration...${NC}"

if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠ .env file not found. Let's create it interactively.${NC}"
    echo ""
    
    # Collect OpenAI API Key
    echo -e "${BLUE}Enter your OpenAI API Key:${NC}"
    echo -e "${YELLOW}(Get one at: https://platform.openai.com/api-keys)${NC}"
    read -p "OPENAI_API_KEY: " OPENAI_KEY
    
    # Collect ArangoDB Password
    echo ""
    echo -e "${BLUE}Set a password for ArangoDB:${NC}"
    echo -e "${YELLOW}(Choose a secure password, min 8 characters)${NC}"
    read -sp "ARANGODB_PASSWORD: " ARANGO_PASS
    echo ""
    
    # Optional: HuggingFace API Key
    echo ""
    echo -e "${BLUE}HuggingFace API Key (optional, press Enter to skip):${NC}"
    read -p "HUGGINGFACE_API_KEY: " HF_KEY
    
    # Create .env file
    cat > .env << EOF
# OpenAI API Configuration
OPENAI_API_KEY=${OPENAI_KEY}

# Redis Configuration
REDIS_URL=redis://localhost:6379

# ArangoDB Configuration
ARANGODB_URL=http://localhost:8529
ARANGODB_USERNAME=root
ARANGODB_PASSWORD=${ARANGO_PASS}
ARANGODB_DATABASE=dappy

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=memories

# HuggingFace API Key (optional, for ML classifiers)
HUGGINGFACE_API_KEY=${HF_KEY:-hf_your_key_here}

# CORS Origins (comma-separated, leave empty for permissive dev mode)
CORS_ORIGINS=

# Logging Level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO
EOF
    
    echo ""
    echo -e "${GREEN}✓ Created .env file with your credentials${NC}"
else
    echo -e "${GREEN}✓ .env file exists${NC}"
    
    # Validate critical env vars
    NEEDS_UPDATE=false
    if grep -q "sk-your-key-here" .env 2>/dev/null; then
        echo -e "${YELLOW}  ⚠ OPENAI_API_KEY has placeholder value${NC}"
        NEEDS_UPDATE=true
    fi
    if grep -q "<your-arango-password>\|your-secure-password\|your_secure_password" .env 2>/dev/null; then
        echo -e "${YELLOW}  ⚠ ARANGODB_PASSWORD has placeholder value${NC}"
        NEEDS_UPDATE=true
    fi
    
    if [ "$NEEDS_UPDATE" = true ]; then
        echo ""
        echo -e "${YELLOW}Would you like to update credentials interactively? (y/N)${NC}"
        read -p "> " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # Collect new credentials
            echo ""
            echo -e "${BLUE}Enter your OpenAI API Key:${NC}"
            read -p "OPENAI_API_KEY: " OPENAI_KEY
            
            echo ""
            echo -e "${BLUE}Set a password for ArangoDB:${NC}"
            read -sp "ARANGODB_PASSWORD: " ARANGO_PASS
            echo ""
            
            # Update .env
            sed -i.bak "s|OPENAI_API_KEY=.*|OPENAI_API_KEY=${OPENAI_KEY}|" .env
            sed -i.bak "s|ARANGODB_PASSWORD=.*|ARANGODB_PASSWORD=${ARANGO_PASS}|" .env
            rm -f .env.bak
            
            echo -e "${GREEN}✓ Updated .env file${NC}"
        else
            echo -e "${YELLOW}Please edit .env manually before continuing.${NC}"
            read -p "Press Enter after updating .env, or Ctrl+C to exit..."
        fi
    fi
fi

echo ""

# ============================================================================
# Step 3: Start Docker Services
# ============================================================================

echo -e "${BLUE}[3/7] Starting Docker services (Redis, ArangoDB, Qdrant)...${NC}"

# Check if containers are already running
if docker ps | grep -q "datclaw-redis\|datclaw-arangodb\|datclaw-qdrant"; then
    echo -e "${YELLOW}⚠ Datclaw containers already running${NC}"
    read -p "Restart them? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose down
        docker-compose up -d
    fi
else
    echo "  Starting containers..."
    if ! docker-compose up -d; then
        echo -e "${RED}✗ Failed to start Docker services${NC}"
        echo ""
        echo "Troubleshooting:"
        echo "  1. Check if Docker daemon is running: docker ps"
        echo "  2. Check .env has ARANGODB_PASSWORD set"
        echo "  3. Check logs: docker-compose logs"
        echo "  4. Try: docker-compose down && docker-compose up -d"
        exit 1
    fi
fi

echo -e "${GREEN}✓ Docker services started${NC}"

# Wait for services to be healthy
echo "  Waiting for services to be healthy (30s)..."
sleep 5

for i in {1..25}; do
    if docker ps --filter "health=healthy" | grep -q "datclaw-redis" && \
       docker ps --filter "health=healthy" | grep -q "datclaw-arangodb" && \
       docker ps --filter "health=healthy" | grep -q "datclaw-qdrant"; then
        echo -e "${GREEN}✓ All services healthy${NC}"
        break
    fi
    
    if [ $i -eq 25 ]; then
        echo -e "${RED}✗ Services not healthy after 30s${NC}"
        echo "Check logs: docker-compose logs"
        exit 1
    fi
    
    sleep 1
done

echo ""

# ============================================================================
# Step 4: Setup Python Backend
# ============================================================================

echo -e "${BLUE}[4/7] Setting up Python backend...${NC}"

cd llm-orchestration

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "  Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi

# Activate virtual environment
echo "  Activating virtual environment..."
if [ "$OS" = "Windows" ]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

# Verify pip is available
if ! command -v pip &> /dev/null; then
    echo -e "${RED}✗ pip not found in virtual environment${NC}"
    echo "  Attempting to bootstrap pip..."
    $PYTHON_CMD -m ensurepip --upgrade
fi

# Upgrade pip
echo "  Upgrading pip..."
pip install --upgrade pip -q

# Ask about ML installation
echo ""
echo -e "${YELLOW}Installation Options:${NC}"
echo "  1) Core only (~200MB, fast) - Recommended for most users"
echo "  2) Core + ML (~2-3GB) - For local ML classifiers and training"
echo ""
read -p "Choose installation type (1 or 2, default=1): " INSTALL_TYPE
INSTALL_TYPE=${INSTALL_TYPE:-1}

if [ "$INSTALL_TYPE" = "2" ]; then
    echo ""
    echo -e "${BLUE}Installing core + ML dependencies (this will take 5-10 minutes)...${NC}"
    pip install -r requirements.txt
    pip install -r requirements-ml.txt
    
    echo "  Downloading spaCy English model..."
    python -m spacy download en_core_web_sm -q
    
    echo -e "${GREEN}✓ Core + ML dependencies installed${NC}"
else
    echo ""
    echo -e "${BLUE}Installing core dependencies...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Core dependencies installed${NC}"
fi

echo ""

# ============================================================================
# Step 5: Verify Installation
# ============================================================================

echo -e "${BLUE}[5/7] Verifying installation...${NC}"

python -c "import fastapi; print('  ✓ FastAPI:', fastapi.__version__)" || exit 1
python -c "import openai; print('  ✓ OpenAI:', openai.__version__)" || exit 1
python -c "import redis; print('  ✓ Redis:', redis.__version__)" || exit 1
python -c "import arango; print('  ✓ ArangoDB client: installed')" || exit 1

if [ "$INSTALL_TYPE" = "2" ]; then
    python -c "import spacy; print('  ✓ spaCy:', spacy.__version__)" 2>/dev/null || echo "  ⚠ spaCy not available"
    python -c "import torch; print('  ✓ PyTorch:', torch.__version__)" 2>/dev/null || echo "  ⚠ PyTorch not available"
fi

echo -e "${GREEN}✓ Installation verified${NC}"
echo ""

# ============================================================================
# Step 6: Setup Frontend (Optional)
# ============================================================================

cd ..  # Back to root

if [ "$HAS_NODE" = true ]; then
    echo -e "${BLUE}[6/7] Setting up frontend...${NC}"
    read -p "Install frontend? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd frontend
        echo "  Installing npm dependencies..."
        npm install
        echo -e "${GREEN}✓ Frontend dependencies installed${NC}"
        cd ..
    else
        echo -e "${YELLOW}⚠ Skipping frontend setup${NC}"
    fi
else
    echo -e "${BLUE}[6/7] Frontend setup skipped (Node.js not available)${NC}"
fi

echo ""

# ============================================================================
# Step 7: Final Instructions
# ============================================================================

echo -e "${BLUE}[7/7] Setup complete!${NC}"
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Setup Complete! 🎉                      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo ""
echo "1. Start the backend:"
echo -e "   ${BLUE}cd llm-orchestration${NC}"
echo -e "   ${BLUE}./start_service.sh${NC}"
echo ""
echo "2. Test the API:"
echo -e "   ${BLUE}curl http://localhost:8000/health${NC}"
echo ""
echo "3. Try the CLI chat:"
echo -e "   ${BLUE}cd llm-orchestration${NC}"
echo -e "   ${BLUE}./chat.sh${NC}"
echo ""

if [ "$HAS_NODE" = true ] && [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "4. Start the frontend (optional):"
    echo -e "   ${BLUE}cd frontend${NC}"
    echo -e "   ${BLUE}npm run dev${NC}"
    echo ""
fi

echo -e "${YELLOW}Useful Commands:${NC}"
echo "  Check Docker services: docker-compose ps"
echo "  View logs: docker-compose logs -f"
echo "  Stop services: docker-compose down"
echo "  Reset databases: cd llm-orchestration && python scripts/reset_all_databases.py"
echo ""
echo -e "${GREEN}Happy memory engineering! 🧠✨${NC}"
