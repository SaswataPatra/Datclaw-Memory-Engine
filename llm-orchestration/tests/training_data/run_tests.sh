#!/bin/bash

# Training Data Collection Test Runner
# 
# Usage:
#   ./run_tests.sh              # Run all tests
#   ./run_tests.sh unit         # Run only unit tests
#   ./run_tests.sh integration  # Run only integration tests
#   ./run_tests.sh e2e          # Run only E2E tests
#   ./run_tests.sh coverage     # Run with coverage report

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Training Data Collection Test Runner${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed${NC}"
    echo "Install with: pip install pytest pytest-asyncio pytest-cov"
    exit 1
fi

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Parse command line argument
TEST_TYPE="${1:-all}"

case "$TEST_TYPE" in
    unit)
        echo -e "${YELLOW}Running unit tests...${NC}"
        pytest tests/training_data/test_training_data_collector.py \
               tests/training_data/test_confidence_routing.py \
               -v
        ;;
    
    integration)
        echo -e "${YELLOW}Running integration tests...${NC}"
        pytest tests/training_data/test_semantic_validator_integration.py \
               tests/training_data/test_api_endpoints.py \
               -v
        ;;
    
    e2e)
        echo -e "${YELLOW}Running E2E tests...${NC}"
        pytest tests/training_data/test_e2e_integration.py -v
        ;;
    
    coverage)
        echo -e "${YELLOW}Running all tests with coverage...${NC}"
        pytest tests/training_data/ \
               --cov=services.training_data_collector \
               --cov=services.classification.semantic_validator \
               --cov=services.classification.classifier_manager \
               --cov-report=html \
               --cov-report=term \
               -v
        
        echo ""
        echo -e "${GREEN}Coverage report generated: htmlcov/index.html${NC}"
        ;;
    
    quick)
        echo -e "${YELLOW}Running quick smoke tests...${NC}"
        pytest tests/training_data/test_training_data_collector.py::TestDatabaseInitialization \
               tests/training_data/test_confidence_routing.py::TestRoutingDecisionLogic \
               -v
        ;;
    
    all|*)
        echo -e "${YELLOW}Running all tests...${NC}"
        pytest tests/training_data/ -v
        ;;
esac

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ All tests passed!${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}❌ Some tests failed${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

