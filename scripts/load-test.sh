#!/usr/bin/env bash
# Load test script for neuriplo-kserve-runtime.
# Validates that the runtime responds correctly under load and that
# metrics are exposed for autoscaling (HPA/KEDA).
#
# Prerequisites:
#   - A running instance of neuriplo-kserve-runtime
#   - curl, jq, and awk available on PATH
#
# Usage:
#   scripts/load-test.sh [HOST] [PORT] [DURATION_SECONDS] [CONCURRENCY]
#
# Defaults: HOST=127.0.0.1 PORT=8080 DURATION=10 CONCURRENCY=4

set -euo pipefail

HOST="${1:-127.0.0.1}"
PORT="${2:-8080}"
DURATION="${3:-10}"
CONCURRENCY="${4:-4}"
BASE_URL="http://${HOST}:${PORT}"

echo "=== Load Test: ${BASE_URL} ==="
echo "Duration: ${DURATION}s, Concurrency: ${CONCURRENCY}"
echo ""

# 1. Health check
echo "--- Health Check ---"
LIVE=$(curl -sf "${BASE_URL}/v2/health/live" 2>/dev/null || echo "FAIL")
READY=$(curl -sf "${BASE_URL}/v2/health/ready" 2>/dev/null || echo "FAIL")

if [[ "${LIVE}" == "FAIL" || "${READY}" == "FAIL" ]]; then
    echo "ERROR: Runtime not healthy. live=${LIVE} ready=${READY}"
    exit 1
fi
echo "liveness: ${LIVE}"
echo "readiness: ${READY}"

# 2. Model metadata check
echo ""
echo "--- Model Metadata ---"
META=$(curl -sf "${BASE_URL}/v2/models/demo" 2>/dev/null || echo "FAIL")
if [[ "${META}" == "FAIL" ]]; then
    echo "ERROR: Model metadata not available"
    exit 1
fi
echo "${META}" | jq -c '{name: .name, versions: .versions, platform: .platform}'

# 3. Concurrent infer requests
echo ""
echo "--- Concurrent Infer (${CONCURRENCY} clients, ${DURATION}s) ---"

END_TIME=$(($(date +%s) + DURATION))
TOTAL=0
SUCCESS=0
FAIL=0

infer_body='{"id":"load-test","inputs":[{"name":"input","shape":[1,3,224,224],"datatype":"FP32","data":[]}]}'

run_infer() {
    local result
    result=$(curl -sf -o /dev/null -w '%{http_code}' \
        -X POST "${BASE_URL}/v2/models/demo/infer" \
        -H 'Content-Type: application/json' \
        -d "${infer_body}" 2>/dev/null || echo "000")
    echo "${result}"
}

while [[ $(date +%s) -lt ${END_TIME} ]]; do
    pids=()
    for ((i = 1; i <= CONCURRENCY; i++)); do
        run_infer &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        code=$(wait "${pid}" 2>/dev/null; echo $?)
        TOTAL=$((TOTAL + 1))
        # run_infer returns the HTTP status code; 0 means curl failure
        if [[ "${code}" -ge 200 && "${code}" -lt 300 ]] 2>/dev/null; then
            SUCCESS=$((SUCCESS + 1))
        else
            FAIL=$((FAIL + 1))
        fi
    done
done

echo "  total: ${TOTAL}"
echo "  success: ${SUCCESS}"
echo "  fail: ${FAIL}"
echo "  success_rate: $(awk "BEGIN {printf \"%.1f\", ${SUCCESS}/${TOTAL}*100}")%"

# 4. Metrics validation for autoscaling
echo ""
echo "--- Metrics Validation (Autoscaling) ---"
METRICS=$(curl -sf "${BASE_URL}/metrics" 2>/dev/null || echo "FAIL")
if [[ "${METRICS}" == "FAIL" ]]; then
    echo "ERROR: Metrics endpoint not available"
    exit 1
fi

check_metric() {
    local name="$1"
    if echo "${METRICS}" | grep -q "^${name}"; then
        local value
        value=$(echo "${METRICS}" | grep "^${name}" | head -1 | awk '{print $NF}')
        echo "  ${name}: ${value}"
    else
        echo "  ${name}: MISSING"
    fi
}

check_metric "neuriplo_scheduler_queue_depth"
check_metric "neuriplo_scheduler_in_flight_requests"
check_metric "neuriplo_scheduler_requests_accepted_total"
check_metric "neuriplo_scheduler_requests_rejected_total"
check_metric "neuriplo_scheduler_queue_latency_seconds_bucket"
check_metric "neuriplo_scheduler_infer_latency_seconds_bucket"
check_metric "neuriplo_scheduler_total_latency_seconds_bucket"
check_metric "neuriplo_http_infer_requests_total"
check_metric "neuriplo_http_infer_requests_success_total"
check_metric "neuriplo_http_infer_requests_failure_total"
check_metric "neuriplo_process_resident_memory_bytes"

# Check version and deployment labels
if echo "${METRICS}" | grep -q 'version='; then
    echo "  version label: present"
else
    echo "  version label: MISSING"
fi
if echo "${METRICS}" | grep -q 'deployment='; then
    echo "  deployment label: present"
else
    echo "  deployment label: not set (default)"
fi

echo ""
echo "=== Load Test Complete ==="