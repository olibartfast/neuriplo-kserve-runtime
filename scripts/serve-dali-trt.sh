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

TENSORRT_DIR="${TENSORRT_DIR:-/home/oli/dependencies/TensorRT-10.13.3.9}"
DALI_DIR="${DALI_DIR:-/home/oli/dependencies/dali}"
BINARY="${BINARY:-$(dirname "$0")/../build/real-dali-trt/neuriplo-kserve-runtime}"

export LD_LIBRARY_PATH="${TENSORRT_DIR}/lib:${DALI_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "${BINARY}" "$@"
