#!/usr/bin/env bash
set -euo pipefail

# E2E YOLO Smoke Test
# Verifies: runtime build → real neuriplo → ONNX model load → metadata → inference

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
BUILD_DIR="${REPO_DIR}/build/real-onnx"
PORT=19090
MODEL="${REPO_DIR}/../neuriplo-infer/models/e2e/yolo26s.onnx"

# Check 1: Model exists
echo "=== Check 1: Model file ==="
if [ -f "$MODEL" ]; then
    pass "Model found at $MODEL"
else
    fail "Model not found at $MODEL"
    exit 1
fi

# Check 2: Build with real neuriplo
echo "=== Check 2: Build ==="
if cmake --preset real-onnx -S "$REPO_DIR" -B "$BUILD_DIR" > /dev/null 2>&1 && \
   cmake --build --preset real-onnx > /dev/null 2>&1; then
    pass "Build succeeds with real neuriplo"
else
    fail "Build failed"
    exit 1
fi

# Check 3: Runtime starts
echo "=== Check 3: Runtime startup ==="
"$BUILD_DIR/neuriplo-kserve-runtime" \
    --model-name yolo \
    --model-path "$MODEL" \
    --backend onnx_runtime \
    --port "$PORT" \
    --instances 1 \
    &>/tmp/e2e-runtime.log &
RUNTIME_PID=$!

for i in $(seq 1 30); do
    if curl -s "http://127.0.0.1:${PORT}/v2/health/live" > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

if curl -s "http://127.0.0.1:${PORT}/v2/health/live" | grep -q '"live":true'; then
    pass "Runtime starts and /v2/health/live returns true"
else
    fail "Runtime failed to start"
    cat /tmp/e2e-runtime.log
    exit 1
fi

# Check 4: Readiness
echo "=== Check 4: Readiness ==="
if curl -s "http://127.0.0.1:${PORT}/v2/health/ready" | grep -q '"ready":true'; then
    pass "/v2/health/ready returns true (model loaded)"
else
    fail "/v2/health/ready returned false"
    curl -s "http://127.0.0.1:${PORT}/v2/health/ready"
fi

# Check 5: Model metadata from neuriplo
echo "=== Check 5: Model metadata ==="
META=$(curl -s "http://127.0.0.1:${PORT}/v2/models/yolo")
if echo "$META" | grep -q '"images"' && echo "$META" | grep -q '"output0"' && \
   echo "$META" | grep -q '"platform":"neuriplo_onnx_runtime"'; then
    pass "Metadata shows neuriplo input 'images' and output 'output0'"
else
    fail "Metadata missing expected fields"
    echo "$META" | python3 -m json.tool 2>/dev/null || echo "$META"
fi

# Check 6: Inference reaches backend
echo "=== Check 6: Inference ==="
RESP=$(python3 -c "
import urllib.request, json
body = json.dumps({
    'id': 'e2e-smoke',
    'inputs': [{
        'name': 'images',
        'shape': [1, 3, 640, 640],
        'datatype': 'FP32',
        'data': [0.5] * 1228800
    }],
    'outputs': [{'name': 'output0'}]
}).encode()
req = urllib.request.Request('http://127.0.0.1:${PORT}/v2/models/yolo/infer',
                              data=body, headers={'Content-Type': 'application/json'})
try:
    resp = urllib.request.urlopen(req, timeout=60)
    r = json.loads(resp.read())
    if 'outputs' in r and len(r['outputs']) > 0:
        o = r['outputs'][0]
        if o['name'] == 'output0' and o['shape'] == [1, 300, 6]:
            print('OK')
        else:
            print('SHAPE_MISMATCH:' + json.dumps(o))
    else:
        print('NO_OUTPUTS:' + json.dumps(r))
except Exception as e:
    print('ERROR:' + str(e))
" 2>/dev/null)

if [ "$RESP" = "OK" ]; then
    pass "Inference returns output0 [1,300,6] via real ONNX Runtime"
else
    fail "Inference failed: $RESP"
fi

# Check 7: Metrics include model label
echo "=== Check 7: Metrics ==="
if curl -s "http://127.0.0.1:${PORT}/metrics" | grep -q 'model="yolo"'; then
    pass "Metrics include model=yolo label"
else
    fail "Metrics missing model label"
fi

echo ""
echo "========================================="
echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}"
echo "========================================="

[ "$FAIL" -eq 0 ]
