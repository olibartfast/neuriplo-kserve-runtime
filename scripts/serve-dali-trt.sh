#!/usr/bin/env bash
# Serve a TensorRT engine with DALI GPU preprocessing available as a `dali`
# model for ensembles.
#
# Dependency story, so nobody has to rediscover it:
#   - TensorRT ships its own libs (TENSORRT_DIR/lib). TensorRT 10 needs no cuDNN.
#   - DALI bundles every third-party lib it needs in DALI_DIR/.libs, resolved
#     via rpath; only DALI_DIR itself goes on the path.
#   - libcuda comes from the NVIDIA driver. Nothing else is required, and in
#     particular nothing from any other vendor's package tree.
set -euo pipefail

TENSORRT_DIR="${TENSORRT_DIR:?set TENSORRT_DIR to the TensorRT install root}"
DALI_DIR="${DALI_DIR:?set DALI_DIR to the DALI install directory}"
BINARY="${BINARY:-$(dirname "$0")/../build/real-dali-trt/neuriplo-kserve-runtime}"

if [ ! -x "$BINARY" ]; then
    echo "error: runtime binary not found or not executable: $BINARY (set BINARY)" >&2
    exit 1
fi

export LD_LIBRARY_PATH="${TENSORRT_DIR}/lib:${DALI_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "${BINARY}" "$@"
