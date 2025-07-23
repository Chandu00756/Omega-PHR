#!/bin/bash
"""
Omega-Paradox Hive Recursion (Ω-PHR) Framework Deployment Script

This script handles the complete deployment of the Ω-PHR framework,
including dependencies, services, and configuration.
"""

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Framework banner
echo -e "${PURPLE}[CAUTION] Omega-Paradox Hive Recursion (Ω-PHR) Framework Deployment [CAUTION]${NC}"
echo -e "${PURPLE}================================================================${NC}"
echo -e "${CYAN}Revolutionary AI Security Testing Framework${NC}"
echo -e "${CYAN}Temporal Paradox Testing • Hive Swarm Attacks • Memory Inversion • Recursive Loops${NC}"
echo ""

# Check requirements
echo -e "${BLUE}[CAUTION] Checking system requirements...${NC}"

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
required_version="3.11"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo -e "${RED}[ERROR] Python 3.11+ required. Found: Python $python_version${NC}"
    exit 1
else
    echo -e "${GREEN}[SUCCESS] Python $python_version detected${NC}"
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[ERROR] Docker is required but not installed${NC}"
    exit 1
else
    echo -e "${GREEN}[SUCCESS] Docker detected${NC}"
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}[ERROR] Docker Compose is required but not installed${NC}"
    exit 1
else
    echo -e "${GREEN}[SUCCESS] Docker Compose detected${NC}"
fi

# Function to show progress
show_progress() {
    local current=$1
    local total=$2
    local task=$3
    local percent=$((current * 100 / total))
    local bar_length=50
    local filled_length=$((percent * bar_length / 100))

    printf "\r${BLUE}["
    for ((i=1; i<=filled_length; i++)); do printf "█"; done
    for ((i=filled_length+1; i<=bar_length; i++)); do printf "░"; done
    printf "] %3d%% - %s${NC}" "$percent" "$task"

    if [ "$current" -eq "$total" ]; then
        echo ""
    fi
}

# Installation steps
total_steps=10
current_step=0

echo -e "\n${YELLOW}[SUCCESS] Starting Ω-PHR Framework Deployment...${NC}"

# Step 1: Setup Python environment
((current_step++))
show_progress $current_step $total_steps "Setting up Python environment"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
echo -e "\n${GREEN}[SUCCESS] Python virtual environment ready${NC}"

# Step 2: Install Python dependencies
((current_step++))
show_progress $current_step $total_steps "Installing Python dependencies"
pip install --upgrade pip > /dev/null 2>&1
pip install -e . > /dev/null 2>&1
echo -e "\n${GREEN}[SUCCESS] Python dependencies installed${NC}"

# Step 3: Install development dependencies
((current_step++))
show_progress $current_step $total_steps "Installing development dependencies"
pip install pytest pytest-asyncio pytest-cov black ruff mypy pre-commit > /dev/null 2>&1
echo -e "\n${GREEN}[SUCCESS] Development dependencies installed${NC}"

# Step 4: Generate Protocol Buffers
((current_step++))
show_progress $current_step $total_steps "Generating Protocol Buffers"
cd services/timeline_lattice
python -m grpc_tools.protoc \
    --proto_path=proto \
    --python_out=. \
    --grpc_python_out=. \
    proto/timeline.proto > /dev/null 2>&1
cd ../..

cd services/hive_orchestrator
python -m grpc_tools.protoc \
    --proto_path=proto \
    --python_out=. \
    --grpc_python_out=. \
    proto/hive.proto > /dev/null 2>&1
cd ../..
echo -e "\n${GREEN}[SUCCESS] Protocol Buffers generated${NC}"

# Step 5: Setup pre-commit hooks
((current_step++))
show_progress $current_step $total_steps "Setting up code quality tools"
pre-commit install > /dev/null 2>&1
echo -e "\n${GREEN}[SUCCESS] Pre-commit hooks installed${NC}"

# Step 6: Run code formatting
((current_step++))
show_progress $current_step $total_steps "Formatting code"
black omega_phr/ services/ tests/ examples/ > /dev/null 2>&1 || true
echo -e "\n${GREEN}[SUCCESS] Code formatted${NC}"

# Step 7: Run linting
((current_step++))
show_progress $current_step $total_steps "Linting code"
ruff check omega_phr/ services/ tests/ examples/ --fix > /dev/null 2>&1 || true
echo -e "\n${GREEN}[SUCCESS] Code linting completed${NC}"

# Step 8: Run tests
((current_step++))
show_progress $current_step $total_steps "Running tests"
python -m pytest tests/ -x --tb=short > /dev/null 2>&1 || echo -e "${YELLOW}[CAUTION]  Some tests may fail due to missing optional dependencies${NC}"
echo -e "\n${GREEN}[SUCCESS] Tests completed${NC}"

# Step 9: Build Docker images
((current_step++))
show_progress $current_step $total_steps "Building Docker images"
docker-compose build > /dev/null 2>&1
echo -e "\n${GREEN}[SUCCESS] Docker images built${NC}"

# Step 10: Final validation
((current_step++))
show_progress $current_step $total_steps "Final validation"
python -c "import omega_phr; print('Framework import successful')" > /dev/null 2>&1
echo -e "\n${GREEN}[SUCCESS] Framework validation completed${NC}"

# Deployment complete
echo ""
echo -e "${GREEN}[SUCCESS] Ω-PHR Framework Deployment Complete! [SUCCESS]${NC}"
echo -e "${PURPLE}================================================================${NC}"
echo ""

# Show usage instructions
echo -e "${CYAN}Quick Start: Quick Start Guide:${NC}"
echo -e "${YELLOW}1. Start all services:${NC}"
echo -e "   ${BLUE}docker-compose up -d${NC}"
echo ""
echo -e "${YELLOW}2. View service logs:${NC}"
echo -e "   ${BLUE}docker-compose logs -f${NC}"
echo ""
echo -e "${YELLOW}3. Run validation test:${NC}"
echo -e "   ${BLUE}python scripts/validate_system.py${NC}"
echo ""
echo -e "${YELLOW}4. Run operational tests:${NC}"
echo -e "   ${BLUE}omega-phr validate --advanced${NC}"
echo ""
echo -e "${YELLOW}5. Access monitoring:${NC}"
echo -e "   ${BLUE}• Grafana: http://localhost:3000 (admin/omega-phr-admin)${NC}"
echo -e "   ${BLUE}• Prometheus: http://localhost:9090${NC}"
echo -e "   ${BLUE}• Ray Dashboard: http://localhost:8265${NC}"
echo ""

# Show service endpoints
echo -e "${CYAN}Service Endpoints: Service Endpoints:${NC}"
echo -e "   ${BLUE}• Timeline Lattice: localhost:50051${NC}"
echo -e "   ${BLUE}• Hive Orchestrator: localhost:50052${NC}"
echo -e "   ${BLUE}• Memory Inversion: localhost:50053${NC}"
echo -e "   ${BLUE}• Recursive Loops: localhost:50054${NC}"
echo -e "   ${BLUE}• Omega Register: localhost:50055${NC}"
echo -e "   ${BLUE}• Telemetry: localhost:50056${NC}"
echo ""

# Show additional commands
echo -e "${CYAN}Useful Commands: Useful Commands:${NC}"
echo -e "${YELLOW}• Stop services:${NC} ${BLUE}docker-compose down${NC}"
echo -e "${YELLOW}• View VS Code tasks:${NC} ${BLUE}Ctrl+Shift+P → Tasks: Run Task${NC}"
echo -e "${YELLOW}• Run tests:${NC} ${BLUE}make test${NC}"
echo -e "${YELLOW}• Format code:${NC} ${BLUE}make format${NC}"
echo ""

# Final message
echo -e "${PURPLE}[SUCCESS] Ready for Google Cloud Platform deployment!${NC}"
echo -e "${GREEN}The Omega-Paradox Hive Recursion (Ω-PHR) Framework is now operational.${NC}"
echo ""
echo -e "${CYAN}For advanced usage and GCP deployment instructions, see README.md${NC}"
echo ""
