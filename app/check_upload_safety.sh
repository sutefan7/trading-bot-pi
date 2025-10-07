#!/bin/bash
# ============================================
# TRADING BOT PI - UPLOAD SAFETY CHECK
# ============================================
# 
# Dit script verifieert dat er geen gevoelige
# informatie in je repository zit voordat je
# naar GitHub upload.
#
# Usage: bash check_upload_safety.sh
# ============================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
ERRORS=0
WARNINGS=0
PASSED=0

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Trading Bot Pi - Upload Safety Check ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# ============================================
# CHECK 1: Secrets files
# ============================================
echo -e "${BLUE}[1/8] Checking for secrets files...${NC}"
SECRETS_FILES=$(find . -type f \( -name "secrets.yaml" -o -name "secrets.enc" -o -name "encryption.key" -o -name ".env" -o -name "*.env" \) 2>/dev/null | grep -v venv | grep -v ".git")

if [ -z "$SECRETS_FILES" ]; then
    echo -e "${GREEN}✅ PASS: No secrets files found${NC}"
    ((PASSED++))
else
    echo -e "${RED}❌ FAIL: Found secrets files:${NC}"
    echo "$SECRETS_FILES"
    echo -e "${RED}   → Action: Remove these files before upload!${NC}"
    ((ERRORS++))
fi
echo ""

# ============================================
# CHECK 2: Hardcoded API keys in code
# ============================================
echo -e "${BLUE}[2/8] Checking for hardcoded API keys...${NC}"
API_KEYS=$(grep -r -i "api_key.*=.*['\"][A-Za-z0-9]\{10,\}" --include="*.py" --include="*.yaml" src/ config*.yaml 2>/dev/null | grep -v "api_key.*=.*get\|api_key.*=.*environ\|api_key.*=.*''\|api_key.*=.*\"\"\|# api_key" || true)

if [ -z "$API_KEYS" ]; then
    echo -e "${GREEN}✅ PASS: No hardcoded API keys found${NC}"
    ((PASSED++))
else
    echo -e "${RED}❌ FAIL: Found potential hardcoded API keys:${NC}"
    echo "$API_KEYS"
    echo -e "${RED}   → Action: Remove hardcoded keys and use environment variables!${NC}"
    ((ERRORS++))
fi
echo ""

# ============================================
# CHECK 3: AWS/Cloud credentials
# ============================================
echo -e "${BLUE}[3/8] Checking for cloud credentials...${NC}"
CLOUD_CREDS=$(grep -r "AKIA\|sk_live\|AIza" --include="*.py" --include="*.yaml" src/ config*.yaml 2>/dev/null || true)

if [ -z "$CLOUD_CREDS" ]; then
    echo -e "${GREEN}✅ PASS: No cloud credentials found${NC}"
    ((PASSED++))
else
    echo -e "${RED}❌ FAIL: Found potential cloud credentials:${NC}"
    echo "$CLOUD_CREDS"
    echo -e "${RED}   → Action: Remove these immediately!${NC}"
    ((ERRORS++))
fi
echo ""

# ============================================
# CHECK 4: Virtual environment
# ============================================
echo -e "${BLUE}[4/8] Checking for virtual environment...${NC}"
if [ -d "venv" ] || [ -d ".venv" ] || [ -d "ENV" ]; then
    # Check if in .gitignore
    if grep -q "venv" .gitignore 2>/dev/null && grep -q ".venv" .gitignore 2>/dev/null; then
        echo -e "${GREEN}✅ PASS: Virtual env exists but is in .gitignore${NC}"
        ((PASSED++))
    else
        echo -e "${YELLOW}⚠️  WARNING: Virtual env exists and may not be ignored${NC}"
        echo -e "${YELLOW}   → Action: Verify .gitignore includes venv/${NC}"
        ((WARNINGS++))
    fi
else
    echo -e "${GREEN}✅ PASS: No virtual environment directory found${NC}"
    ((PASSED++))
fi
echo ""

# ============================================
# CHECK 5: Large files (>10MB)
# ============================================
echo -e "${BLUE}[5/8] Checking for large files (>10MB)...${NC}"
LARGE_FILES=$(find . -type f -size +10M 2>/dev/null | grep -v venv | grep -v ".git" || true)

if [ -z "$LARGE_FILES" ]; then
    echo -e "${GREEN}✅ PASS: No large files found${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠️  WARNING: Found large files (>10MB):${NC}"
    echo "$LARGE_FILES"
    echo -e "${YELLOW}   → Action: Use Git LFS or add to .gitignore${NC}"
    ((WARNINGS++))
fi
echo ""

# ============================================
# CHECK 6: .gitignore exists
# ============================================
echo -e "${BLUE}[6/8] Checking .gitignore configuration...${NC}"
if [ -f ".gitignore" ]; then
    # Check for essential entries
    MISSING_ENTRIES=""
    for entry in "*.env" "secrets.yaml" "venv/" "__pycache__/" "*.log"; do
        if ! grep -q "$entry" .gitignore 2>/dev/null; then
            MISSING_ENTRIES="$MISSING_ENTRIES\n  - $entry"
        fi
    done
    
    if [ -z "$MISSING_ENTRIES" ]; then
        echo -e "${GREEN}✅ PASS: .gitignore is properly configured${NC}"
        ((PASSED++))
    else
        echo -e "${YELLOW}⚠️  WARNING: .gitignore missing entries:${NC}"
        echo -e "${MISSING_ENTRIES}"
        echo -e "${YELLOW}   → Action: Add these to .gitignore${NC}"
        ((WARNINGS++))
    fi
else
    echo -e "${RED}❌ FAIL: .gitignore not found${NC}"
    echo -e "${RED}   → Action: Create .gitignore before upload!${NC}"
    ((ERRORS++))
fi
echo ""

# ============================================
# CHECK 7: Config files
# ============================================
echo -e "${BLUE}[7/8] Checking config files for credentials...${NC}"
CONFIG_CREDS=0

for config in config.yaml config.production.yaml; do
    if [ -f "$config" ]; then
        # Check for non-empty credentials
        CREDS=$(grep -E "bot_token|api_key|api_secret|chat_id" "$config" | grep -v "''\|\"\"" | grep -v "^#" | grep -v "Will be loaded from" || true)
        if [ ! -z "$CREDS" ]; then
            echo -e "${RED}❌ Found credentials in $config:${NC}"
            echo "$CREDS"
            CONFIG_CREDS=1
        fi
    fi
done

if [ $CONFIG_CREDS -eq 0 ]; then
    echo -e "${GREEN}✅ PASS: No credentials in config files${NC}"
    ((PASSED++))
else
    echo -e "${RED}   → Action: Remove credentials or don't commit these files!${NC}"
    ((ERRORS++))
fi
echo ""

# ============================================
# CHECK 8: Git repository status
# ============================================
echo -e "${BLUE}[8/8] Checking Git repository status...${NC}"
if [ -d ".git" ]; then
    # Check if there are staged files
    STAGED=$(git diff --cached --name-only 2>/dev/null || true)
    if [ ! -z "$STAGED" ]; then
        echo -e "${YELLOW}⚠️  INFO: Found staged files:${NC}"
        echo "$STAGED" | head -10
        if [ $(echo "$STAGED" | wc -l) -gt 10 ]; then
            echo "   ... and $(( $(echo "$STAGED" | wc -l) - 10 )) more"
        fi
        echo -e "${YELLOW}   → Review with: git diff --staged${NC}"
        ((WARNINGS++))
    else
        echo -e "${GREEN}✅ PASS: Git initialized, no staged files${NC}"
        ((PASSED++))
    fi
else
    echo -e "${YELLOW}⚠️  INFO: Git not initialized yet${NC}"
    echo -e "${YELLOW}   → Run: git init${NC}"
    ((WARNINGS++))
fi
echo ""

# ============================================
# SUMMARY
# ============================================
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║            SAFETY CHECK SUMMARY         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "Total checks:    8"
echo -e "${GREEN}Passed:          $PASSED${NC}"
echo -e "${YELLOW}Warnings:        $WARNINGS${NC}"
echo -e "${RED}Errors:          $ERRORS${NC}"
echo ""

# ============================================
# FINAL VERDICT
# ============================================
if [ $ERRORS -eq 0 ]; then
    if [ $WARNINGS -eq 0 ]; then
        echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  ✅ SAFE TO UPLOAD TO PRIVATE GITHUB   ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${GREEN}Next steps:${NC}"
        echo -e "  1. git init"
        echo -e "  2. git add .gitignore"
        echo -e "  3. git commit -m 'Initial commit'"
        echo -e "  4. Create PRIVATE repo on GitHub"
        echo -e "  5. git remote add origin YOUR_REPO_URL"
        echo -e "  6. git push -u origin main"
        echo ""
        exit 0
    else
        echo -e "${YELLOW}╔════════════════════════════════════════╗${NC}"
        echo -e "${YELLOW}║  ⚠️  SAFE BUT WITH WARNINGS           ║${NC}"
        echo -e "${YELLOW}╚════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${YELLOW}Review warnings above before upload.${NC}"
        echo -e "${YELLOW}If OK, proceed with upload to PRIVATE repo.${NC}"
        echo ""
        exit 0
    fi
else
    echo -e "${RED}╔════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  ❌ NOT SAFE TO UPLOAD - FIX ERRORS!   ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${RED}Fix the errors above before uploading!${NC}"
    echo -e "${RED}DO NOT proceed with upload.${NC}"
    echo ""
    exit 1
fi




