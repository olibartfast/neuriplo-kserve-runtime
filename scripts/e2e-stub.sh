#!/usr/bin/env bash
set -euo pipefail

# E2E Stub Smoke Test
# Verifies the KServe V2 HTTP surface with the stub backend via curl:
# health, metadata, inference, metrics, and admin model lifecycle.

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

PASS=0
FAIL=0
RUNTIME_PID=""

pass() { echo -e "${GREEN}[PASS]${NC} $1"; PASS=$((PASS + 1)); }
fail() { echo -e "${RED}[FAIL]${NC} $1"; FAIL=$((FAIL + 1)); }

cleanup() {
    if [ -n "$RUNTIME_PID" ]; then
        kill "$RUNTIME_PID" 2>/dev/null || true
        wait "$RUNTIME_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${BUILD_DIR:-${REPO_DIR}/build/debug}"
PORT="${PORT:-19080}"
BASE="http://127.0.0.1:${PORT}"

# Check 1: Runtime binary exists
echo "=== Check 1: Runtime binary ==="
if [ -x "$BUILD_DIR/neuriplo-kserve-runtime" ]; then
    pass "Binary found at $BUILD_DIR/neuriplo-kserve-runtime"
else
    fail "Binary not found at $BUILD_DIR/neuriplo-kserve-runtime (build the debug preset first)"
    exit 1
fi

# Check 2: Runtime starts with stub backend
echo "=== Check 2: Runtime startup ==="
"$BUILD_DIR/neuriplo-kserve-runtime" \
    --model-name demo \
    --backend stub \
    --port "$PORT" \
    &>/tmp/e2e-stub-runtime.log &
RUNTIME_PID=$!

for _ in $(seq 1 30); do
    if curl -s "$BASE/v2/health/live" > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

if curl -s "$BASE/v2/health/live" | grep -q '"live":true'; then
    pass "Runtime starts and /v2/health/live returns true"
else
    fail "Runtime failed to start"
    cat /tmp/e2e-stub-runtime.log
    exit 1
fi

# Check 3: Readiness
echo "=== Check 3: Readiness ==="
if curl -s "$BASE/v2/health/ready" | grep -q '"ready":true'; then
    pass "/v2/health/ready returns true"
else
    fail "/v2/health/ready returned false"
    curl -s "$BASE/v2/health/ready"
fi

# Check 4: Server metadata
echo "=== Check 4: Server metadata ==="
if curl -s "$BASE/v2" | grep -q '"name"'; then
    pass "/v2 returns server metadata"
else
    fail "/v2 missing server metadata"
fi

# Check 5: Model metadata
echo "=== Check 5: Model metadata ==="
META=$(curl -s "$BASE/v2/models/demo")
if echo "$META" | grep -q '"platform":"neuriplo_stub"' && echo "$META" | grep -q '"input"'; then
    pass "Metadata shows stub platform and input tensor"
else
    fail "Metadata missing expected fields: $META"
fi

# Check 6: Inference
echo "=== Check 6: HTTP inference ==="
INFER_BODY='{"id":"stub-smoke","inputs":[{"name":"input","shape":[1,3,224,224],"datatype":"FP32","data":[]}]}'
RESP=$(curl -s -X POST -H 'Content-Type: application/json' -d "$INFER_BODY" "$BASE/v2/models/demo/infer")
if echo "$RESP" | grep -q '"id":"stub-smoke"' && echo "$RESP" | grep -q '"name":"output"'; then
    pass "Inference echoes id and returns stub output tensor"
else
    fail "Inference response unexpected: $RESP"
fi

# Check 7: Unknown model returns 404
echo "=== Check 7: Unknown model error ==="
STATUS=$(curl -s -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' \
    -d "$INFER_BODY" "$BASE/v2/models/missing/infer")
if [ "$STATUS" = "404" ]; then
    pass "Unknown model infer returns 404"
else
    fail "Unknown model infer returned $STATUS"
fi

# Check 8: Metrics
echo "=== Check 8: Metrics ==="
METRICS=$(curl -s "$BASE/metrics")
if echo "$METRICS" | grep -q 'neuriplo_http_infer_requests_total' && \
   echo "$METRICS" | grep -q 'model="demo"'; then
    pass "Metrics expose infer counters with model label"
else
    fail "Metrics missing expected series"
fi

# Check 9: Admin list models
echo "=== Check 9: Admin list ==="
if curl -s "$BASE/v2/admin/models" | grep -q '"name":"demo"'; then
    pass "Admin list shows demo model"
else
    fail "Admin list missing demo model"
fi

# Check 10: Admin load + unload second model
echo "=== Check 10: Admin load/unload ==="
LOAD_RESP=$(curl -s -X POST -H 'Content-Type: application/json' \
    -d '{"model_name":"second","backend":"stub"}' "$BASE/v2/admin/models/load")
READY_RESP=$(curl -s "$BASE/v2/models/second/ready")
UNLOAD_STATUS=$(curl -s -o /dev/null -w '%{http_code}' -X DELETE "$BASE/v2/admin/models/second")
if echo "$LOAD_RESP" | grep -q '"loaded":true' && \
   echo "$READY_RESP" | grep -q '"ready":true' && \
   [ "$UNLOAD_STATUS" = "200" ]; then
    pass "Admin load, readiness, and unload of a second model succeed"
else
    fail "Admin lifecycle failed: load=$LOAD_RESP ready=$READY_RESP unload=$UNLOAD_STATUS"
fi

# Check 11: First model unaffected after unload
echo "=== Check 11: Isolation ==="
if curl -s "$BASE/v2/models/demo/ready" | grep -q '"ready":true'; then
    pass "demo model still ready after second model unload"
else
    fail "demo model readiness lost"
fi

echo ""
echo "========================================="
echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}"
echo "========================================="

[ "$FAIL" -eq 0 ]
