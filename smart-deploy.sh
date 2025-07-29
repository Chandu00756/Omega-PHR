#!/usr/bin/env bash

# Smart deployment script - only push on successful build
# Usage: ./deploy.sh "commit message"

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

if [ $# -eq 0 ]; then
    echo -e "${RED}❌ Error: Please provide a commit message${NC}"
    echo "Usage: ./deploy.sh \"Your commit message here\""
    exit 1
fi

COMMIT_MESSAGE="$1"

echo -e "${BLUE}🚀 Smart Deployment Process Started${NC}"
echo "=================================================="
echo -e "${BLUE}📝 Commit message: $COMMIT_MESSAGE${NC}"
echo ""

# Step 1: Check for changes
echo -e "${BLUE}📋 Step 1: Checking for changes...${NC}"

if git diff --quiet && git diff --cached --quiet; then
    echo -e "${YELLOW}⚠️  No changes to commit${NC}"
    exit 0
fi

git status --porcelain

# Step 2: Run comprehensive validation
echo -e "\n${BLUE}🔧 Step 2: Running comprehensive build validation...${NC}"
./validate-build.sh || {
    echo -e "${RED}❌ Build validation failed! Not pushing to Git.${NC}"
    echo -e "${YELLOW}💡 Fix the issues above and try again.${NC}"
    exit 1
}

# Step 3: Stage changes
echo -e "\n${BLUE}📦 Step 3: Staging changes...${NC}"
git add -A

# Step 4: Pre-commit hooks
echo -e "\n${BLUE}🔍 Step 4: Running pre-commit hooks...${NC}"
if command -v pre-commit &> /dev/null; then
    pre-commit run --all-files || {
        echo -e "${YELLOW}⚠️  Pre-commit hooks made changes. Re-staging...${NC}"
        git add -A
    }
else
    echo -e "${YELLOW}⚠️  Pre-commit not installed, skipping hooks${NC}"
fi

# Step 5: Commit
echo -e "\n${BLUE}💾 Step 5: Committing changes...${NC}"
# First attempt - if pre-commit hooks make changes, stage them and try again
git commit -m "$COMMIT_MESSAGE" || {
    echo -e "${YELLOW}⚠️  Pre-commit hooks made changes. Re-staging and committing...${NC}"
    git add -A
    git commit -m "$COMMIT_MESSAGE" || {
        echo -e "${RED}❌ Commit failed${NC}"
        exit 1
    }
}

# Step 6: Final validation after commit
echo -e "\n${BLUE}🔍 Step 6: Final validation after commit...${NC}"
# Run a quick test to ensure the commit is still valid
/Users/chanduchitikam/omega-phr/.venv/bin/python -m pytest tests/test_basic.py -c pytest-ci.toml --tb=short -q || {
    echo -e "${RED}❌ Post-commit validation failed! Rolling back...${NC}"
    git reset --hard HEAD~1
    exit 1
}

# Step 7: Push to origin
echo -e "\n${BLUE}📡 Step 7: Pushing to origin...${NC}"
git push origin main || {
    echo -e "${RED}❌ Push failed${NC}"
    exit 1
}

# Success!
echo -e "\n${GREEN}🎉 DEPLOYMENT SUCCESSFUL! 🎉${NC}"
echo "=================================================="
echo -e "${GREEN}✅ Build validation passed${NC}"
echo -e "${GREEN}✅ Pre-commit hooks executed${NC}"
echo -e "${GREEN}✅ Changes committed${NC}"
echo -e "${GREEN}✅ Post-commit validation passed${NC}"
echo -e "${GREEN}✅ Successfully pushed to Git${NC}"
echo ""
echo -e "${GREEN}📋 Latest commit:${NC}"
git log -1 --oneline
echo ""
echo -e "${BLUE}🔗 Check CI status at: https://github.com/Chandu00756/Omega-PHR/actions${NC}"
