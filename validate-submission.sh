#!/usr/bin/env bash
# =============================================================================
# validate-submission.sh — Pre-Submission Validation Script
# =============================================================================
# Run this script before submitting to the Meta OpenEnv Hackathon.
# It checks all three requirements from the Pre-Submission Checklist:
#   1. HF Space is live and POST /reset returns 200
#   2. docker build succeeds locally
#   3. openenv validate passes
#
# Usage:
#   chmod +x validate-submission.sh
#   ./validate-submission.sh
#
# Override the Space URL:
#   SPACE_URL=https://huggingface.co/spaces/unknown1321/autonomous-enterprise-payment-orchestrator \
#     ./validate-submission.sh
# =============================================================================

set -e   # abort on first error

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'   # No Colour

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; EXIT_CODE=1; }
info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

EXIT_CODE=0

echo "============================================================"
echo "  AOEP Pre-Submission Validation"
echo "============================================================"
echo ""

# ── Config ───────────────────────────────────────────────────────────────────
SPACE_URL="${SPACE_URL:-https://unknown1321-autonomous-enterprise-payment-orchestrator.hf.space}"
IMAGE_NAME="aoep-validate"

# ── Check 1: HF Space health ─────────────────────────────────────────────────
echo "── Check 1: HF Space liveness ──────────────────────────────"
info "Pinging GET ${SPACE_URL} ..."

HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "${SPACE_URL}" || echo "000")
if [ "$HTTP_STATUS" = "200" ]; then
    pass "GET / returned HTTP 200"
else
    fail "GET / returned HTTP ${HTTP_STATUS} (Space may be sleeping or URL is wrong)"
fi

info "Checking POST ${SPACE_URL}/reset for task=easy ..."
RESET_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 \
    -X POST "${SPACE_URL}/reset" \
    -H "Content-Type: application/json" \
    -d '{"task": "easy"}' || echo "000")

if [ "$RESET_STATUS" = "200" ]; then
    pass "POST /reset returned HTTP 200"
else
    fail "POST /reset returned HTTP ${RESET_STATUS}"
fi

echo ""

# ── Check 2: Docker build ─────────────────────────────────────────────────────
echo "── Check 2: Docker build ───────────────────────────────────"
if ! command -v docker &> /dev/null; then
    info "Docker not found — skipping build check"
else
    info "Building Docker image '${IMAGE_NAME}' ..."
    if docker build --platform linux/amd64 -t "${IMAGE_NAME}" . --quiet; then
        pass "docker build succeeded"
        # Cleanup
        docker rmi "${IMAGE_NAME}" --force > /dev/null 2>&1 || true
    else
        fail "docker build FAILED — review Dockerfile and requirements.txt"
    fi
fi

echo ""

# ── Check 3: openenv validate ─────────────────────────────────────────────────
echo "── Check 3: openenv validate ───────────────────────────────"
if ! command -v openenv &> /dev/null; then
    info "openenv CLI not found — installing openenv ..."
    pip3 install openenv --quiet
fi

# tomllib is stdlib in Python 3.11+; patch openenv to use tomli backport on 3.10
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "11")
if [ "$PY_MINOR" -lt 11 ]; then
    pip3 install tomli --quiet
    python3 -c "
import site, pathlib
for d in site.getsitepackages():
    f = pathlib.Path(d) / 'openenv' / 'cli' / '_validation.py'
    if f.exists():
        txt = f.read_text()
        if 'import tomli as tomllib' not in txt:
            txt = txt.replace('import tomllib', 'try:\n    import tomllib\nexcept ImportError:\n    import tomli as tomllib')
            f.write_text(txt)
        break
" 2>/dev/null
fi

if openenv validate .; then
    pass "openenv validate PASSED"
else
    fail "openenv validate FAILED — review openenv.yaml and unified_gateway.py"
fi

echo ""

# ── Check 4: pytest suite ──────────────────────────────────────────────────────
echo "── Check 4: pytest suite ───────────────────────────────────"
if ! command -v pytest &> /dev/null; then
    info "pytest not found — installing ..."
    pip3 install pytest --quiet
fi

if pytest tests/ -q --tb=short 2>&1; then
    pass "All pytest tests passed"
else
    fail "pytest tests FAILED — review tests/ directory"
fi

echo ""

# ── Summary ──────────────────────────────────────────────────────────────────
echo "============================================================"
if [ "$EXIT_CODE" -eq 0 ]; then
    echo -e "${GREEN}  ✅ ALL CHECKS PASSED — Safe to submit!${NC}"
else
    echo -e "${RED}  ❌ SOME CHECKS FAILED — Fix issues before submitting.${NC}"
fi
echo "============================================================"

exit $EXIT_CODE