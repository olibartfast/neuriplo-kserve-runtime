#!/usr/bin/env bash
set -euo pipefail

# E2E Vision-GGUF Smoke Test (Step 16)
# Verifies: runtime build with real neuriplo GGML backend -> RF-DETR .gguf
# model loads on the TENSOR path (--task-type) -> metadata advertises real
# tensor I/O (not the BYTES prompt/text LLM convention) -> HTTP inference
# returns the expected output shapes.
#
# Model weights are not committed (107+ MB). Provide them via:
#   GGML_RFDETR_MODEL=/path/to/rfdetr.gguf            (required, 1,3,560,560)
#   GGML_RFDETRSEG_MODEL=/path/to/rfdetr_seg.gguf     (optional, 1,3,432,432)
# Produce them with neuriplo/scripts/convert_onnx_to_gguf.py from the
# per-task ONNX exporters (see export/ggml/README.md in neuriplo-tasks).
# To iterate against a local neuriplo checkout instead of the GitHub clone:
#   NEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR=/path/to/neuriplo

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
BUILD_DIR="${REPO_DIR}/build/real-ggml"
PORT=19092
MODEL="${GGML_RFDETR_MODEL:-/tmp/rfdetr.gguf}"
SEG_MODEL="${GGML_RFDETRSEG_MODEL:-}"

# Check 1: Model exists
echo "=== Check 1: Model file ==="
if [ -f "$MODEL" ]; then
    pass "Model found at $MODEL"
else
    fail "Model not found at $MODEL (set GGML_RFDETR_MODEL)"
    exit 1
fi

# Check 2: Build with real neuriplo GGML backend
echo "=== Check 2: Build ==="
CMAKE_EXTRA=()
if [ -n "${NEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR:-}" ]; then
    CMAKE_EXTRA+=("-DNEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR=${NEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR}")
fi
if cmake --preset real-ggml -S "$REPO_DIR" -B "$BUILD_DIR" "${CMAKE_EXTRA[@]}" > /dev/null 2>&1 && \
   cmake --build --preset real-ggml > /dev/null 2>&1; then
    pass "Build succeeds with real neuriplo GGML backend"
else
    fail "Build failed"
    exit 1
fi

start_runtime() {
    local model_name="$1" model_path="$2" task_type="$3" input_sizes="$4" port="$5"
    "$BUILD_DIR/neuriplo-kserve-runtime" \
        --model-name "$model_name" \
        --model-path "$model_path" \
        --backend ggml \
        --task-type "$task_type" \
        --input-sizes "$input_sizes" \
        --port "$port" \
        --instances 1 \
        &>/tmp/e2e-ggml-runtime.log &
    RUNTIME_PID=$!
    for i in $(seq 1 30); do
        if curl -s "http://127.0.0.1:${port}/v2/health/live" > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done
}

stop_runtime() {
    if [ -n "$RUNTIME_PID" ]; then
        kill "$RUNTIME_PID" 2>/dev/null || true
        wait "$RUNTIME_PID" 2>/dev/null || true
        RUNTIME_PID=""
    fi
}

# Check 3: Detection model loads on the tensor path
echo "=== Check 3: RF-DETR detection load ==="
start_runtime rfdetr "$MODEL" rfdetr "1,3,560,560" "$PORT"
if curl -s "http://127.0.0.1:${PORT}/v2/health/ready" | grep -q '"ready":true'; then
    pass "/v2/health/ready returns true (GGUF model loaded)"
else
    fail "/v2/health/ready returned false"
    tail -20 /tmp/e2e-ggml-runtime.log || true
    exit 1
fi

# Check 4: Metadata advertises real tensor I/O, not the LLM convention
echo "=== Check 4: Detection metadata ==="
META=$(curl -s "http://127.0.0.1:${PORT}/v2/models/rfdetr")
if echo "$META" | grep -q '"platform":"neuriplo_ggml"' && \
   echo "$META" | grep -q '"name":"input1"' && \
   echo "$META" | grep -q '"name":"output0"' && \
   ! echo "$META" | grep -q '"name":"prompt"'; then
    pass "Metadata shows neuriplo_ggml with tensor I/O (no BYTES prompt)"
else
    fail "Metadata missing expected tensor I/O"
    echo "$META" | python3 -m json.tool 2>/dev/null || echo "$META"
fi

# Check 5: HTTP inference returns the RF-DETR head shapes
echo "=== Check 5: Detection inference ==="
RESP=$(python3 -c "
import urllib.request, json
body = json.dumps({
    'id': 'e2e-ggml',
    'inputs': [{
        'name': 'input1',
        'shape': [1, 3, 560, 560],
        'datatype': 'FP32',
        'data': [0.0] * 940800
    }],
    'outputs': [{'name': 'output0'}, {'name': 'output1'}]
}).encode()
req = urllib.request.Request('http://127.0.0.1:${PORT}/v2/models/rfdetr/infer',
                              data=body, headers={'Content-Type': 'application/json'})
try:
    resp = urllib.request.urlopen(req, timeout=300)
    r = json.loads(resp.read())
    shapes = {o['name']: o['shape'] for o in r.get('outputs', [])}
    if shapes.get('output0') == [1, 300, 4] and shapes.get('output1') == [1, 300, 91]:
        print('OK')
    else:
        print('SHAPE_MISMATCH:' + json.dumps(shapes))
except Exception as e:
    print('ERROR:' + str(e))
" 2>/dev/null)

if [ "$RESP" = "OK" ]; then
    pass "HTTP inference returns output0 [1,300,4] and output1 [1,300,91]"
else
    fail "HTTP inference failed: $RESP"
fi
stop_runtime

# Check 6: Segmentation slice (optional, needs GGML_RFDETRSEG_MODEL)
if [ -n "$SEG_MODEL" ]; then
    echo "=== Check 6: RF-DETR-seg load + inference ==="
    if [ ! -f "$SEG_MODEL" ]; then
        fail "Seg model not found at $SEG_MODEL"
        exit 1
    fi
    start_runtime rfdetrseg "$SEG_MODEL" rfdetrseg "1,3,432,432" "$PORT"
    SEG_RESP=$(python3 -c "
import urllib.request, json, time
for _ in range(30):
    try:
        urllib.request.urlopen('http://127.0.0.1:${PORT}/v2/health/ready', timeout=5)
        break
    except Exception:
        time.sleep(1)
meta = json.loads(urllib.request.urlopen('http://127.0.0.1:${PORT}/v2/models/rfdetrseg', timeout=30).read())
shapes = {o['name']: o['shape'] for o in meta.get('inputs', []) + meta.get('outputs', [])}
if 'prompt' in shapes:
    print('LLM_METADATA')
else:
    body = json.dumps({
        'id': 'e2e-ggml-seg',
        'inputs': [{
            'name': 'input1',
            'shape': [1, 3, 432, 432],
            'datatype': 'FP32',
            'data': [0.0] * 559872
        }],
        'outputs': [{'name': 'output0'}, {'name': 'output1'}]
    }).encode()
    req = urllib.request.Request('http://127.0.0.1:${PORT}/v2/models/rfdetrseg/infer',
                                  data=body, headers={'Content-Type': 'application/json'})
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    out = {o['name']: o['shape'] for o in r.get('outputs', [])}
    if out.get('output0') == [1, 200, 4] and out.get('output1') == [1, 200, 108, 108]:
        print('OK')
    else:
        print('SHAPE_MISMATCH:' + json.dumps(out))
" 2>/dev/null)
    if [ "$SEG_RESP" = "OK" ]; then
        pass "Seg inference returns output0 [1,200,4] and output1 [1,200,108,108]"
    else
        fail "Seg inference failed: $SEG_RESP"
    fi
    stop_runtime
else
    echo "=== Check 6: RF-DETR-seg (skipped, GGML_RFDETRSEG_MODEL unset) ==="
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
