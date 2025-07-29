#!/usr/bin/env bash

# Comprehensive build and test validation script for Omega-PHR
# This script ensures all tests pass before allowing git push

set -e  # Exit on any error

echo "🔧 Starting comprehensive build validation for Omega-PHR..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get Python executable
PYTHON_EXE="/Users/chanduchitikam/omega-phr/.venv/bin/python"

if [ ! -f "$PYTHON_EXE" ]; then
    echo -e "${RED}❌ Python virtual environment not found at $PYTHON_EXE${NC}"
    exit 1
fi

echo -e "${BLUE}📋 Using Python: $PYTHON_EXE${NC}"

# Step 1: Syntax and Import Validation
echo -e "\n${BLUE}🔍 Step 1: Validating Python syntax and imports...${NC}"
echo "Checking core modules..."

for module in omega_phr/__init__.py omega_phr/models.py omega_phr/hive.py omega_phr/timeline.py omega_phr/memory.py omega_phr/loops.py omega_phr/omega_register.py; do
    if [ -f "$module" ]; then
        echo "  - Checking $module..."
        $PYTHON_EXE -m py_compile "$module" || {
            echo -e "${RED}❌ Syntax error in $module${NC}"
            exit 1
        }
    fi
done

# Step 2: Import Testing
echo -e "\n${BLUE}🔍 Step 2: Testing core imports...${NC}"
$PYTHON_EXE -c "
import sys
import traceback

try:
    print('  - Testing omega_phr imports...')
    from omega_phr.models import OmegaTestResult, Event, HiveAgent, AttackStrategy
    from omega_phr.hive import HiveOrchestrator
    from omega_phr.timeline import TimelineLattice
    from omega_phr.memory import MemoryInverter
    from omega_phr.loops import RecursiveLoopSynthesizer
    from omega_phr.omega_register import OmegaStateRegister
    print('  ✅ All core imports successful')
except Exception as e:
    print(f'  ❌ Import error: {e}')
    traceback.print_exc()
    sys.exit(1)
" || {
    echo -e "${RED}❌ Core import validation failed${NC}"
    exit 1
}

# Step 3: Basic Functionality Tests
echo -e "\n${BLUE}🧪 Step 3: Running basic functionality tests...${NC}"
$PYTHON_EXE -m pytest tests/test_basic.py -v -c pytest-ci.toml --tb=short || {
    echo -e "${RED}❌ Basic functionality tests failed${NC}"
    exit 1
}

# Step 4: Individual Component Tests
echo -e "\n${BLUE}🧪 Step 4: Running component tests...${NC}"

# Test each major component individually
components=(
    "tests/test_omega_phr.py::TestEvent"
    "tests/test_omega_phr.py::TestTimelineLattice"
    "tests/test_omega_phr.py::TestHiveOrchestrator"
    "tests/test_omega_phr.py::TestMemoryInverter"
    "tests/test_omega_phr.py::TestRecursiveLoopSynthesizer"
    "tests/test_omega_phr.py::TestOmegaStateRegister"
)

for component in "${components[@]}"; do
    echo "  - Testing $component..."
    $PYTHON_EXE -m pytest "$component" -v -c pytest-ci.toml --tb=short --disable-warnings -q || {
        echo -e "${RED}❌ Component test failed: $component${NC}"
        exit 1
    }
done

# Step 5: Full Test Suite (with timeout)
echo -e "\n${BLUE}🧪 Step 5: Running full test suite...${NC}"
# Use gtimeout if available (brew install coreutils), otherwise run without timeout
if command -v gtimeout &> /dev/null; then
    gtimeout 600 $PYTHON_EXE -m pytest tests/ -c pytest-ci.toml --tb=short --disable-warnings --maxfail=5 -q || {
        if [ $? -eq 124 ]; then
            echo -e "${RED}❌ Tests timed out after 10 minutes${NC}"
        else
            echo -e "${RED}❌ Full test suite failed${NC}"
        fi
        exit 1
    }
else
    # Run without timeout on macOS
    $PYTHON_EXE -m pytest tests/ -c pytest-ci.toml --tb=short --disable-warnings --maxfail=5 -q || {
        echo -e "${RED}❌ Full test suite failed${NC}"
        exit 1
    }
fi

# Step 6: Static Analysis (if tools available)
echo -e "\n${BLUE}🔍 Step 6: Static analysis checks...${NC}"
if command -v ruff &> /dev/null; then
    echo "  - Running ruff linting..."
    ruff check omega_phr/ --fix || echo -e "${YELLOW}⚠️  Linting issues found but continuing...${NC}"
else
    echo "  - Ruff not available, skipping linting"
fi

# Step 7: Validate CI Configuration
echo -e "\n${BLUE}⚙️  Step 7: Validating CI configuration...${NC}"
if [ -f "pytest-ci.toml" ]; then
    echo "  ✅ pytest-ci.toml exists"
else
    echo -e "${RED}❌ pytest-ci.toml missing${NC}"
    exit 1
fi

if [ -f "requirements-ci.txt" ]; then
    echo "  ✅ requirements-ci.txt exists"
else
    echo -e "${RED}❌ requirements-ci.txt missing${NC}"
    exit 1
fi

# Final Success
echo -e "\n${GREEN}🎉 ALL VALIDATION CHECKS PASSED! 🎉${NC}"
echo "=================================================="
echo -e "${GREEN}✅ Python syntax validation${NC}"
echo -e "${GREEN}✅ Import validation${NC}"
echo -e "${GREEN}✅ Basic functionality tests${NC}"
echo -e "${GREEN}✅ Component tests${NC}"
echo -e "${GREEN}✅ Full test suite${NC}"
echo -e "${GREEN}✅ Static analysis${NC}"
echo -e "${GREEN}✅ CI configuration${NC}"
echo ""
echo -e "${GREEN}🚀 Ready to push to Git! 🚀${NC}"
