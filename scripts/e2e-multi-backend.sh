#!/usr/bin/env bash
set -euo pipefail

# E2E Multi-Backend Test (local GPU machine only — not CI)
# One server process serves two models through two different neuriplo
# backends: an ONNX model via the built-in onnx_runtime backend and a
# TensorRT engine via the dlopen'd tensorrt plugin. Exercises admin
# load, inference on both, and drain/reload of one model while the
# other keeps serving.
#
# Requires a build with built-in ONNX_RUNTIME and a TENSORRT plugin, e.g.:
#   cmake -S . -B build/real-trt-plugin -G Ninja \
#     -DCMAKE_BUILD_TYPE=Debug \
#     -DNEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO=ON \
#     -DDEFAULT_BACKEND=ONNX_RUNTIME \
#     -DNEURIPLO_PLUGIN_BACKENDS=TENSORRT \
#     -DTENSORRT_DIR=$HOME/dependencies/TensorRT-10.13.3.9 \
#     -DBUILD_INFERENCE_ENGINE_TESTS=OFF
#   cmake --build build/real-trt-plugin --parallel
#
# Environment overrides: BUILD_DIR, PLUGIN_DIR, MODEL_REPOSITORY, PORT,
# TRT_MODEL_NAME, TRT_MODEL_PATH.
#
# The default TRT model is raft_large_trt from the Triton repository; a
# plan file only loads with the exact TensorRT version that built it, so
# point TRT_MODEL_PATH at a locally built engine when versions differ, e.g.:
#   trtexec --onnx=.../yolo26m_seg_onnx/1/model.onnx --saveEngine=/tmp/yolo.plan
#   TRT_MODEL_NAME=yolo26m_seg_trt TRT_MODEL_PATH=/tmp/yolo.plan scripts/e2e-multi-backend.sh

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
BUILD_DIR="${BUILD_DIR:-${REPO_DIR}/build/real-trt-plugin}"
PLUGIN_DIR="${PLUGIN_DIR:-${BUILD_DIR}/plugins}"
MODEL_REPOSITORY="${MODEL_REPOSITORY:-/home/oli/model_repository}"
PORT="${PORT:-19085}"
BASE="http://127.0.0.1:${PORT}"

YOLO_MODEL_PATH="${MODEL_REPOSITORY}/yolo26m_seg_onnx/1/model.onnx"
TRT_MODEL_NAME="${TRT_MODEL_NAME:-raft_large}"
TRT_MODEL_PATH="${TRT_MODEL_PATH:-${MODEL_REPOSITORY}/${TRT_MODEL_NAME}_trt/1/model.plan}"

# Builds a zero-filled KServe V2 infer payload from the model's metadata.
infer_body() {
    local model="$1"
    curl -s "$BASE/v2/models/$model" | python3 -c '
import json, sys
meta = json.load(sys.stdin)
inputs = []
for tensor in meta["inputs"]:
    shape = [1 if dim < 0 else dim for dim in tensor["shape"]]
    count = 1
    for dim in shape:
        count *= dim
    inputs.append({
        "name": tensor["name"],
        "shape": shape,
        "datatype": tensor["datatype"],
        "data": [0.0] * count,
    })
print(json.dumps({"inputs": inputs}))
'
}

infer_ok() {
    local model="$1" body_file="$2"
    local status resp_file="/tmp/e2e-multi-infer-resp-${model}.json"
    status=$(curl -s -o "$resp_file" -w '%{http_code}' \
        -X POST -H 'Content-Type: application/json' \
        --data-binary "@${body_file}" "$BASE/v2/models/$model/infer")
    [ "$status" = "200" ] && grep -q '"outputs"' "$resp_file"
}

# Check 1: prerequisites
echo "=== Check 1: Prerequisites ==="
if [ -x "$BUILD_DIR/neuriplo-kserve-runtime" ]; then
    pass "Binary found at $BUILD_DIR/neuriplo-kserve-runtime"
else
    fail "Binary not found at $BUILD_DIR/neuriplo-kserve-runtime (see header for build instructions)"
    exit 1
fi
if [ -f "$YOLO_MODEL_PATH" ] && [ -f "$TRT_MODEL_PATH" ]; then
    pass "Model repository has yolo26m_seg_onnx and the TRT engine ($TRT_MODEL_PATH)"
else
    fail "Missing models under $MODEL_REPOSITORY"
    exit 1
fi
if ls "$PLUGIN_DIR"/libneuriplo_backend_*.so >/dev/null 2>&1; then
    pass "Plugin directory has backend plugins: $(ls "$PLUGIN_DIR")"
else
    fail "No plugins found in $PLUGIN_DIR"
    exit 1
fi

# Check 2: server starts with the ONNX model on the built-in backend
echo "=== Check 2: Runtime startup ==="
"$BUILD_DIR/neuriplo-kserve-runtime" \
    --model-name yolo26m_seg \
    --model-path "$YOLO_MODEL_PATH" \
    --backend onnx_runtime \
    --plugin-dir "$PLUGIN_DIR" \
    --port "$PORT" \
    &>/tmp/e2e-multi-runtime.log &
RUNTIME_PID=$!

for _ in $(seq 1 60); do
    if curl -s "$BASE/v2/health/live" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if curl -s "$BASE/v2/health/live" | grep -q '"live":true'; then
    pass "Runtime starts with yolo26m_seg on onnx_runtime"
else
    fail "Runtime failed to start"
    cat /tmp/e2e-multi-runtime.log
    exit 1
fi

# Check 3: admin reports both backends available
echo "=== Check 3: Available backends ==="
ADMIN=$(curl -s "$BASE/v2/admin/models")
if echo "$ADMIN" | grep -q '"onnx_runtime"' && echo "$ADMIN" | grep -q '"tensorrt"'; then
    pass "Admin lists onnx_runtime and tensorrt as available"
else
    fail "Admin backends list unexpected: $ADMIN"
fi

# Check 4: load the TensorRT model through the plugin
# TRT execution contexts are not thread-safe; keep instances=1.
echo "=== Check 4: Load ${TRT_MODEL_NAME} via tensorrt plugin ==="
LOAD_RESP=$(curl -s -X POST -H 'Content-Type: application/json' -d "{
    \"model_name\": \"${TRT_MODEL_NAME}\",
    \"backend\": \"tensorrt\",
    \"model_path\": \"$TRT_MODEL_PATH\",
    \"plugin_dir\": \"$PLUGIN_DIR\",
    \"instances\": 1
}" "$BASE/v2/admin/models/load")
if echo "$LOAD_RESP" | grep -q '"loaded":true' &&
    curl -s "$BASE/v2/models/${TRT_MODEL_NAME}/ready" | grep -q '"ready":true'; then
    pass "${TRT_MODEL_NAME} loaded and ready on tensorrt"
else
    fail "${TRT_MODEL_NAME} load failed: $LOAD_RESP"
    tail -20 /tmp/e2e-multi-runtime.log
    exit 1
fi

# Check 5: inference on both models in one process
echo "=== Check 5: Inference on both backends ==="
infer_body yolo26m_seg >/tmp/e2e-multi-yolo.json
infer_body ${TRT_MODEL_NAME} >/tmp/e2e-multi-raft.json
if infer_ok yolo26m_seg /tmp/e2e-multi-yolo.json; then
    pass "yolo26m_seg inference via onnx_runtime"
else
    fail "yolo26m_seg inference failed"
fi
if infer_ok ${TRT_MODEL_NAME} /tmp/e2e-multi-raft.json; then
    pass "${TRT_MODEL_NAME} inference via tensorrt plugin"
else
    fail "${TRT_MODEL_NAME} inference failed"
fi

# Check 6: reload ${TRT_MODEL_NAME} while yolo26m_seg keeps serving
echo "=== Check 6: Drain/reload under load ==="
YOLO_LOOP_FAILED=/tmp/e2e-multi-yolo-loop-failed
rm -f "$YOLO_LOOP_FAILED"
(
    for _ in $(seq 1 20); do
        if ! infer_ok yolo26m_seg /tmp/e2e-multi-yolo.json; then
            touch "$YOLO_LOOP_FAILED"
            break
        fi
    done
) &
YOLO_LOOP_PID=$!

RELOAD_STATUS=$(curl -s -o /dev/null -w '%{http_code}' -X POST \
    -H 'Content-Type: application/json' -d '{}' \
    "$BASE/v2/admin/models/${TRT_MODEL_NAME}/reload")
wait "$YOLO_LOOP_PID"

if [ "$RELOAD_STATUS" = "200" ] && [ ! -f "$YOLO_LOOP_FAILED" ]; then
    pass "${TRT_MODEL_NAME} reloaded while yolo26m_seg kept serving"
else
    fail "Reload under load failed: reload=$RELOAD_STATUS yolo_loop_failed=$([ -f "$YOLO_LOOP_FAILED" ] && echo yes || echo no)"
fi

if infer_ok ${TRT_MODEL_NAME} /tmp/e2e-multi-raft.json; then
    pass "${TRT_MODEL_NAME} inference works after reload"
else
    fail "${TRT_MODEL_NAME} inference failed after reload"
fi

# Check 7: unload the TRT model, ONNX model unaffected
echo "=== Check 7: Unload isolation ==="
UNLOAD_STATUS=$(curl -s -o /dev/null -w '%{http_code}' -X DELETE "$BASE/v2/admin/models/${TRT_MODEL_NAME}")
if [ "$UNLOAD_STATUS" = "200" ] && infer_ok yolo26m_seg /tmp/e2e-multi-yolo.json; then
    pass "${TRT_MODEL_NAME} unloaded; yolo26m_seg still serving"
else
    fail "Unload isolation failed: unload=$UNLOAD_STATUS"
fi

echo ""
echo "========================================="
echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}"
echo "========================================="

[ "$FAIL" -eq 0 ]
