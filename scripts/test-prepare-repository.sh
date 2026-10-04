#!/usr/bin/env bash
set -uo pipefail

# Tests deploy/prepare/prepare-repository.sh, the serving image's ENTRYPOINT.
#
# Runs with no GPU, no TensorRT, and no runtime binary: trtexec and
# neuriplo-kserve-runtime are replaced by stubs on PATH that record how they were
# called. What is under test is the dispatch and the tree it produces, not
# inference -- so this belongs in CI on an ordinary runner.
#
# PREPARE_SHELL selects the shell that runs the preparer, e.g.
#   PREPARE_SHELL=dash scripts/test-prepare-repository.sh
#   PREPARE_SHELL="busybox sh" scripts/test-prepare-repository.sh
# Unset, the script runs through its own #!/bin/sh shebang.

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

PASS=0
FAIL=0

pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    PASS=$((PASS + 1))
}
fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    FAIL=$((FAIL + 1))
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
PREPARE="$REPO_DIR/deploy/prepare/prepare-repository.sh"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# --- stubs -------------------------------------------------------------------
# trtexec writes a marker file where a real engine would go and appends its
# argument line to a log, so a test can assert both that it ran and how.
BIN="$WORK/bin"
mkdir -p "$BIN"

cat >"$BIN/trtexec" <<'STUB'
#!/usr/bin/env bash
if [ "${1:-}" = "--version" ]; then
    echo "&&&& RUNNING TensorRT.trtexec [TensorRT v${FAKE_TRT_VERSION:-100000}]"
    exit 0
fi
echo "$*" >>"$TRTEXEC_LOG"
if [ "${TRTEXEC_FAIL:-false}" = "true" ]; then
    exit 1
fi
for arg in "$@"; do
    case "$arg" in
        --saveEngine=*) printf 'fake engine' >"${arg#--saveEngine=}" ;;
    esac
done
STUB

cat >"$BIN/neuriplo-kserve-runtime" <<'STUB'
#!/usr/bin/env bash
echo "$*" >>"$SERVE_LOG"
STUB

cat >"$BIN/nvidia-smi" <<'STUB'
#!/usr/bin/env bash
case "$*" in
    *driver_version*) echo "${FAKE_DRIVER:-550.00}" ;;
    *) echo "${FAKE_GPU:-Fake GPU A}" ;;
esac
STUB

chmod +x "$BIN/trtexec" "$BIN/neuriplo-kserve-runtime" "$BIN/nvidia-smi"
export PATH="$BIN:$PATH"

# --- helpers -----------------------------------------------------------------
# Each case gets its own staging dir and repository so nothing leaks between
# them; run_prepare returns the script's exit code without tripping set -e.
CASE=""
STAGE=""
REPO=""

new_case() {
    CASE="$1"
    STAGE="$WORK/$CASE/staging"
    REPO="$WORK/$CASE/repo"
    mkdir -p "$STAGE" "$REPO"
    export TRTEXEC_LOG="$WORK/$CASE/trtexec.log"
    export SERVE_LOG="$WORK/$CASE/serve.log"
    : >"$TRTEXEC_LOG"
    : >"$SERVE_LOG"
}

run_prepare() {
    # PREPARE_SHELL is word-split on purpose ("busybox sh"); empty expands to
    # nothing and the script runs through its shebang.
    # shellcheck disable=SC2086
    STAGE_DIR="$STAGE" MODEL_REPOSITORY="$REPO" ${PREPARE_SHELL:-} "$PREPARE" "$@" \
        >"$WORK/$CASE/out.log" 2>&1
    return $?
}

expect_file() {
    if [ -f "$REPO/$1" ]; then
        pass "$CASE: $1 exists"
    else
        fail "$CASE: $1 missing"
        find "$REPO" -mindepth 1 | sed 's/^/      /'
    fi
}

expect_absent() {
    if [ -e "$REPO/$1" ]; then
        fail "$CASE: $1 should not exist"
    else
        pass "$CASE: $1 absent"
    fi
}

expect_exit() {
    if [ "$1" -eq "$2" ]; then
        pass "$CASE: exit $2"
    else
        fail "$CASE: expected exit $2, got $1"
        sed 's/^/      /' "$WORK/$CASE/out.log"
    fi
}

expect_log_contains() {
    if grep -qF -- "$2" "$1"; then
        pass "$CASE: log contains '$2'"
    else
        fail "$CASE: log missing '$2'"
        sed 's/^/      /' "$1"
    fi
}

# =============================================================================
echo "=== Case 1: heterogeneous flat staging ==="
new_case heterogeneous
printf 'onnx' >"$STAGE/detector.onnx"
printf 'pte' >"$STAGE/ecdet.pte"
printf 'tflite' >"$STAGE/classifier.tflite"
printf 'xml' >"$STAGE/segmenter.xml"
printf 'bin' >"$STAGE/segmenter.bin"
printf 'dali' >"$STAGE/preprocess.dali"
printf 'graph' >"$STAGE/yolo_ensemble.json"
printf 'prebuilt' >"$STAGE/depth.plan"

run_prepare --port 8080
expect_exit $? 0
# The ONNX is compiled; every portable format is copied under a name whose
# extension selects its backend.
expect_file "detector/1/model.plan"
expect_file "ecdet/1/model.pte"
expect_file "classifier/1/model.tflite"
expect_file "segmenter/1/model.xml"
expect_file "segmenter/1/model.bin"
expect_file "preprocess/1/model.dali"
expect_file "yolo_ensemble/1/model.json"
expect_file "depth/1/model.plan"
expect_log_contains "$SERVE_LOG" "--models=$REPO"
expect_log_contains "$SERVE_LOG" "--port 8080"

# =============================================================================
echo "=== Case 2: OpenVINO weights are renamed to match the .xml ==="
new_case openvino
printf 'xml' >"$STAGE/seg.xml"
printf 'weights' >"$STAGE/seg.bin"
run_prepare
expect_exit $? 0
# OpenVINO resolves weights by basename, so seg.bin must land as model.bin.
if [ "$(cat "$REPO/seg/1/model.bin" 2>/dev/null)" = "weights" ]; then
    pass "$CASE: .bin renamed alongside .xml"
else
    fail "$CASE: .bin not renamed to model.bin"
fi
expect_absent "seg/1/seg.bin"

# =============================================================================
echo "=== Case 3: orphan .bin is an error, not a silent skip ==="
new_case orphan-bin
printf 'weights' >"$STAGE/lonely.bin"
run_prepare
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "no matching .xml"

# =============================================================================
echo "=== Case 4: unrecognized artifact fails loudly, opt-out available ==="
new_case unknown
printf 'onnx' >"$STAGE/good.onnx"
printf 'labels' >"$STAGE/labels.names"
run_prepare
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "unrecognized staged file"

new_case unknown-ignored
printf 'onnx' >"$STAGE/good.onnx"
printf 'labels' >"$STAGE/labels.names"
PREPARE_IGNORE_UNKNOWN=true run_prepare
expect_exit $? 0
expect_file "good/1/model.plan"

# =============================================================================
echo "=== Case 5: ONNX_BACKEND=onnx_runtime skips compilation ==="
new_case onnx-passthrough
printf 'onnx' >"$STAGE/detector.onnx"
ONNX_BACKEND=onnx_runtime run_prepare
expect_exit $? 0
expect_file "detector/1/model.onnx"
expect_absent "detector/1/model.plan"
if [ -s "$TRTEXEC_LOG" ]; then
    fail "$CASE: trtexec ran despite ONNX_BACKEND=onnx_runtime"
else
    pass "$CASE: trtexec never invoked"
fi

# =============================================================================
echo "=== Case 6: preparation is idempotent across restarts ==="
new_case idempotent
printf 'onnx' >"$STAGE/detector.onnx"
run_prepare
expect_exit $? 0
first_count=$(wc -l <"$TRTEXEC_LOG")
run_prepare
expect_exit $? 0
second_count=$(wc -l <"$TRTEXEC_LOG")
if [ "$first_count" -eq 1 ] && [ "$second_count" -eq 1 ]; then
    pass "$CASE: engine built once, reused on restart"
else
    fail "$CASE: trtexec ran $second_count times across two starts (expected 1)"
fi

# =============================================================================
echo "=== Case 7: a failed conversion publishes nothing ==="
new_case atomic
printf 'onnx' >"$STAGE/detector.onnx"
TRTEXEC_FAIL=true run_prepare
expect_exit $? 1
# The half-built directory must not look like a finished model to the next start.
expect_absent "detector/1"

# A retry after the failure succeeds, rather than tripping over leftovers.
run_prepare
expect_exit $? 0
expect_file "detector/1/model.plan"

# =============================================================================
echo "=== Case 8: a stale temp directory never gets published ==="
new_case stale-tmp
printf 'onnx' >"$STAGE/detector.onnx"
mkdir -p "$REPO/detector/.prepare-tmp.1.999"
printf 'truncated' >"$REPO/detector/.prepare-tmp.1.999/model.plan"
run_prepare
expect_exit $? 0
expect_file "detector/1/model.plan"
if [ "$(cat "$REPO/detector/1/model.plan")" = "fake engine" ]; then
    pass "$CASE: published the rebuilt engine, not the leftover"
else
    fail "$CASE: leftover temp content was published"
fi

# =============================================================================
echo "=== Case 9: tree-form staging is copied verbatim ==="
new_case tree-form
mkdir -p "$STAGE/raft/3"
printf 'engine' >"$STAGE/raft/3/model.plan"
printf 'labels' >"$STAGE/raft/3/labels.txt"
printf 'config' >"$STAGE/raft/config.pbtxt"
run_prepare
expect_exit $? 0
# Verbatim means the version and the extra files survive: this is the escape
# hatch for anything the flat form cannot express.
expect_file "raft/3/model.plan"
expect_file "raft/3/labels.txt"
expect_file "raft/config.pbtxt"
expect_absent "raft/1"

# =============================================================================
echo "=== Case 10: flat .pbtxt lands beside the version directory ==="
new_case pbtxt-overlay
printf 'pte' >"$STAGE/ecdet.pte"
printf 'input { name: "images" }' >"$STAGE/ecdet.pbtxt"
run_prepare
expect_exit $? 0
expect_file "ecdet/1/model.pte"
expect_file "ecdet/config.pbtxt"
expect_absent "ecdet/1/model.pbtxt"

new_case pbtxt-orphan
printf 'config' >"$STAGE/nobody.pbtxt"
run_prepare
expect_exit $? 1

# =============================================================================
echo "=== Case 11: per-model TRT overrides ==="
new_case per-model-shapes
printf 'onnx' >"$STAGE/static_det.onnx"
printf 'onnx' >"$STAGE/raft-large.onnx"
# One static model and one dynamic model in the same repository: a single global
# TRT_SHAPES cannot express this, which is what the override exists for.
TRT_SHAPES_RAFT_LARGE="input:1x3x480x640" TRT_PRECISION_STATIC_DET=fp32 run_prepare
expect_exit $? 0
expect_log_contains "$TRTEXEC_LOG" "--shapes=input:1x3x480x640"
if grep -F -- "static_det" "$TRTEXEC_LOG" | grep -qF -- "--shapes="; then
    fail "$CASE: static model was given explicit shapes"
else
    pass "$CASE: static model built without --shapes"
fi
if grep -F -- "static_det" "$TRTEXEC_LOG" | grep -qF -- "--fp16"; then
    fail "$CASE: TRT_PRECISION_STATIC_DET=fp32 did not override the default"
else
    pass "$CASE: per-model precision override applied"
fi

# =============================================================================
echo "=== Case 12: empty staging refuses to serve ==="
new_case empty
run_prepare
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "no servable artifacts staged"
if [ -s "$SERVE_LOG" ]; then
    fail "$CASE: server started on an empty repository"
else
    pass "$CASE: server never started"
fi

# =============================================================================
echo "=== Case 13: missing trtexec is not a silent substitution ==="
new_case no-trtexec
printf 'onnx' >"$STAGE/detector.onnx"
mv "$BIN/trtexec" "$BIN/trtexec.hidden"
run_prepare
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "trtexec not found"

new_case no-trtexec-fallback
printf 'onnx' >"$STAGE/detector.onnx"
TRT_FALLBACK_ONNX=true run_prepare
expect_exit $? 0
expect_file "detector/1/model.onnx"
mv "$BIN/trtexec.hidden" "$BIN/trtexec"

# =============================================================================
echo "=== Case 14: PREPARE_ONLY builds the tree without starting anything ==="
new_case prepare-only
printf 'onnx' >"$STAGE/detector.onnx"
printf 'pte' >"$STAGE/ecdet.pte"
PREPARE_ONLY=true run_prepare
expect_exit $? 0
expect_file "detector/1/model.plan"
expect_file "ecdet/1/model.pte"
# The whole point of the init-container form: the tree is the deliverable, and
# no server is launched by the thing that built it.
if [ -s "$SERVE_LOG" ]; then
    fail "$CASE: a server was started despite PREPARE_ONLY=true"
else
    pass "$CASE: no server started"
fi

# =============================================================================
echo "=== Case 15: SERVER_EXEC drives an arbitrary server ==="
new_case server-exec
printf 'onnx' >"$STAGE/detector.onnx"
cat >"$BIN/tritonserver" <<'STUB'
#!/usr/bin/env bash
echo "$*" >>"$SERVE_LOG"
STUB
chmod +x "$BIN/tritonserver"
SERVER_EXEC="tritonserver --model-repository=$REPO" run_prepare --strict-readiness=true
expect_exit $? 0
expect_log_contains "$SERVE_LOG" "--model-repository=$REPO"
# Container arguments still reach the server, so a manifest's args: keep working
# whichever server is wrapped.
expect_log_contains "$SERVE_LOG" "--strict-readiness=true"

# =============================================================================
echo "=== Case 16: Triton layout renames to Triton's conventions ==="
new_case triton-layout
printf 'onnx' >"$STAGE/detector.onnx"
printf 'ts' >"$STAGE/pose.torchscript"
printf 'graph' >"$STAGE/legacy.pb"
printf 'xml' >"$STAGE/seg.xml"
printf 'bin' >"$STAGE/seg.bin"
REPOSITORY_LAYOUT=triton PREPARE_ONLY=true run_prepare
expect_exit $? 0
expect_file "detector/1/model.plan"
# Triton detects platform by filename, so a TorchScript must be model.pt and a
# frozen graph model.graphdef whatever the staged extension was.
expect_file "pose/1/model.pt"
expect_absent "pose/1/model.torchscript"
expect_file "legacy/1/model.graphdef"
expect_absent "legacy/1/model.pb"
expect_file "seg/1/model.xml"
expect_file "seg/1/model.bin"

# =============================================================================
echo "=== Case 17: OVMS layout, and formats it cannot serve are refused ==="
new_case ovms-layout
printf 'xml' >"$STAGE/seg.xml"
printf 'bin' >"$STAGE/seg.bin"
REPOSITORY_LAYOUT=ovms ONNX_BACKEND=onnx_runtime PREPARE_ONLY=true run_prepare
expect_exit $? 0
expect_file "seg/1/model.xml"
expect_file "seg/1/model.bin"

new_case ovms-refuses-engine
printf 'onnx' >"$STAGE/detector.onnx"
# OVMS cannot load a TensorRT engine. Compiling one and dropping it in the tree
# would produce a model that silently never appears, so this must fail here.
REPOSITORY_LAYOUT=ovms PREPARE_ONLY=true run_prepare
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "cannot serve a .plan model"
# Refusing must leave the repository untouched, not a half-built model directory
# that the next start would mistake for finished work.
expect_absent "detector/1"
if [ -s "$TRTEXEC_LOG" ]; then
    fail "$CASE: compiled an engine for a layout that cannot load one"
else
    pass "$CASE: refused before doing any work"
fi

new_case triton-refuses-executorch
printf 'pte' >"$STAGE/ecdet.pte"
REPOSITORY_LAYOUT=triton PREPARE_ONLY=true run_prepare
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "cannot serve a .pte model"

# =============================================================================
echo "=== Case 18: an unknown layout is rejected up front ==="
new_case bad-layout
printf 'onnx' >"$STAGE/detector.onnx"
REPOSITORY_LAYOUT=torchserve run_prepare
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "unsupported REPOSITORY_LAYOUT"

# =============================================================================
echo "=== Case 19: publishing into an existing empty version dir does not nest ==="
new_case empty-target
printf 'onnx' >"$STAGE/detector.onnx"
mkdir -p "$REPO/detector/1"
run_prepare
expect_exit $? 0
expect_file "detector/1/model.plan"
expect_file "detector/1/.prepared"
if find "$REPO/detector" -name '.prepare-tmp.*' | grep -q .; then
    fail "$CASE: work directory nested or left behind"
else
    pass "$CASE: no nested work directory"
fi

new_case nonempty-unstamped
printf 'onnx' >"$STAGE/detector.onnx"
mkdir -p "$REPO/detector/1"
printf 'legacy' >"$REPO/detector/1/junk.txt"
run_prepare
expect_exit $? 0
expect_file "detector/1/model.plan"
expect_absent "detector/1/junk.txt"
if [ "$(wc -l <"$TRTEXEC_LOG")" -eq 1 ]; then
    pass "$CASE: unstamped directory is rebuilt"
else
    fail "$CASE: unstamped directory was not rebuilt"
fi

# =============================================================================
echo "=== Case 20: the stamp records provenance and gates re-preparation ==="
new_case stamp
printf 'onnx-v1' >"$STAGE/detector.onnx"
run_prepare
expect_exit $? 0
STAMP_PATH="$REPO/detector/1/.prepared"
expect_log_contains "$STAMP_PATH" "source_sha256="
expect_log_contains "$STAMP_PATH" "tensorrt=100000"
expect_log_contains "$STAMP_PATH" "gpu=Fake GPU A;"
expect_log_contains "$STAMP_PATH" "driver=550.00;"
expect_log_contains "$STAMP_PATH" "precision=fp16"
expect_log_contains "$STAMP_PATH" "layout=neuriplo"

count_builds() { wc -l <"$TRTEXEC_LOG" | tr -d ' '; }

run_prepare
expect_exit $? 0
if [ "$(count_builds)" = 1 ]; then
    pass "$CASE: identical inputs reuse the engine"
else
    fail "$CASE: identical inputs rebuilt"
fi

printf 'onnx-v2' >"$STAGE/detector.onnx"
run_prepare
if [ "$(count_builds)" = 2 ]; then
    pass "$CASE: changed source rebuilds"
else
    fail "$CASE: changed source did not rebuild"
fi

TRT_PRECISION=fp32 run_prepare
if [ "$(count_builds)" = 3 ]; then
    pass "$CASE: changed precision rebuilds"
else
    fail "$CASE: changed precision did not rebuild"
fi

TRT_PRECISION=fp32 TRT_SHAPES="images:1x3x64x64" run_prepare
if [ "$(count_builds)" = 4 ]; then
    pass "$CASE: changed shapes rebuild"
else
    fail "$CASE: changed shapes did not rebuild"
fi

TRT_PRECISION=fp32 TRT_SHAPES="images:1x3x64x64" TRT_EXTRA_ARGS="--workspace=64" run_prepare
if [ "$(count_builds)" = 5 ]; then
    pass "$CASE: changed extra args rebuild"
else
    fail "$CASE: changed extra args did not rebuild"
fi

export TRT_PRECISION=fp32 TRT_SHAPES="images:1x3x64x64" TRT_EXTRA_ARGS="--workspace=64"
FAKE_TRT_VERSION=100001 run_prepare
if [ "$(count_builds)" = 6 ]; then
    pass "$CASE: changed TensorRT version rebuilds"
else
    fail "$CASE: changed TensorRT version did not rebuild"
fi

FAKE_TRT_VERSION=100001 FAKE_GPU="Fake GPU B" run_prepare
if [ "$(count_builds)" = 7 ]; then
    pass "$CASE: changed GPU rebuilds"
else
    fail "$CASE: changed GPU did not rebuild"
fi

FAKE_TRT_VERSION=100001 FAKE_GPU="Fake GPU B" FAKE_DRIVER=560.00 run_prepare
if [ "$(count_builds)" = 8 ]; then
    pass "$CASE: changed driver rebuilds"
else
    fail "$CASE: changed driver did not rebuild"
fi

FAKE_TRT_VERSION=100001 FAKE_GPU="Fake GPU B" FAKE_DRIVER=560.00 run_prepare
if [ "$(count_builds)" = 8 ]; then
    pass "$CASE: unchanged environment reuses the engine"
else
    fail "$CASE: unchanged environment rebuilt"
fi
unset TRT_PRECISION TRT_SHAPES TRT_EXTRA_ARGS

rm "$REPO/detector/1/.prepared"
FAKE_TRT_VERSION=100001 FAKE_GPU="Fake GPU B" FAKE_DRIVER=560.00 TRT_PRECISION=fp32 \
    TRT_SHAPES="images:1x3x64x64" TRT_EXTRA_ARGS="--workspace=64" run_prepare
if [ "$(count_builds)" = 9 ]; then
    pass "$CASE: missing stamp rebuilds"
else
    fail "$CASE: missing stamp did not rebuild"
fi

new_case stamp-layout
printf 'pte' >"$STAGE/ecdet.pte"
run_prepare
expect_exit $? 0
printf 'pte2' >"$STAGE/ecdet.pte"
run_prepare
expect_exit $? 0
if [ "$(cat "$REPO/ecdet/1/model.pte")" = "pte2" ]; then
    pass "$CASE: changed copy-through artifact is re-prepared"
else
    fail "$CASE: stale copy-through artifact served"
fi

new_case stamp-tree
mkdir -p "$STAGE/raft/3"
printf 'engine' >"$STAGE/raft/3/model.plan"
run_prepare
expect_exit $? 0
printf 'engine2' >"$STAGE/raft/3/model.plan"
run_prepare
expect_exit $? 0
if [ "$(cat "$REPO/raft/3/model.plan")" = "engine2" ]; then
    pass "$CASE: changed tree is re-prepared"
else
    fail "$CASE: stale tree served"
fi
expect_file "raft/3/.prepared"

new_case removed-not-pruned
printf 'onnx' >"$STAGE/a.onnx"
printf 'onnx' >"$STAGE/b.onnx"
run_prepare
rm "$STAGE/b.onnx"
run_prepare
expect_exit $? 0
expect_file "b/1/model.plan"

# =============================================================================
echo "=== Case 21: duplicate model names fail loudly ==="
new_case duplicate-name
printf 'onnx' >"$STAGE/detector.onnx"
printf 'plan' >"$STAGE/detector.plan"
run_prepare
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "detector.onnx"
expect_log_contains "$WORK/$CASE/out.log" "detector.plan"
expect_log_contains "$WORK/$CASE/out.log" "model name 'detector'"
expect_absent "detector"

new_case duplicate-tree-flat
printf 'onnx' >"$STAGE/raft.onnx"
mkdir -p "$STAGE/raft/2"
printf 'engine' >"$STAGE/raft/2/model.plan"
run_prepare
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "both resolve to model name 'raft'"

# =============================================================================
echo "=== Case 22: versions must be numeric ==="
new_case bad-model-version
printf 'onnx' >"$STAGE/detector.onnx"
MODEL_VERSION=../../escape run_prepare
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "MODEL_VERSION must be"
if [ -e "$WORK/$CASE/escape" ]; then
    fail "$CASE: wrote outside the repository"
else
    pass "$CASE: nothing written outside the repository"
fi

new_case empty-model-version
printf 'onnx' >"$STAGE/detector.onnx"
MODEL_VERSION="" run_prepare
# An empty variable falls back to the default of 1.
expect_exit $? 0
expect_file "detector/1/model.plan"

new_case bad-tree-version
mkdir -p "$STAGE/raft/v1"
printf 'engine' >"$STAGE/raft/v1/model.plan"
run_prepare
expect_exit $? 1
expect_log_contains "$WORK/$CASE/out.log" "version directory must be a non-negative integer"
expect_absent "raft"

new_case non-integer-tree-version
mkdir -p "$STAGE/raft/1.5"
printf 'engine' >"$STAGE/raft/1.5/model.plan"
run_prepare
expect_exit $? 1

# =============================================================================
echo "=== Case 23: a failed rebuild keeps the served version ==="
new_case failed-rebuild-keeps-old
printf 'onnx-v1' >"$STAGE/detector.onnx"
run_prepare
expect_exit $? 0
cp "$REPO/detector/1/model.plan" "$WORK/$CASE/plan.before"
cp "$REPO/detector/1/.prepared" "$WORK/$CASE/stamp.before"
printf 'onnx-v2' >"$STAGE/detector.onnx"
TRTEXEC_FAIL=true run_prepare
expect_exit $? 1
if cmp -s "$REPO/detector/1/model.plan" "$WORK/$CASE/plan.before" &&
    cmp -s "$REPO/detector/1/.prepared" "$WORK/$CASE/stamp.before"; then
    pass "$CASE: old model.plan and .prepared untouched"
else
    fail "$CASE: failed rebuild altered the served version"
fi

# =============================================================================
echo
echo "=== Summary ==="
echo "passed: $PASS"
echo "failed: $FAIL"
[ "$FAIL" -eq 0 ]
