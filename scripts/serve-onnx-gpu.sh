#!/usr/bin/env bash
# Serve models through ONNX Runtime's CUDA execution provider.
#
# The CUDA EP needs cuDNN 9 and cuBLAS at run time. These come from NVIDIA's
# own pip wheels installed into a dedicated directory:
#
#   pip install --target "$NVIDIA_RUNTIME_DIR/.." nvidia-cudnn-cu12
#
# Do NOT source them from another product's environment (an OpenVINO or torch
# venv that happens to contain them): that couples the server's runtime to an
# unrelated package's upgrade cycle.
set -euo pipefail

ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR:?set ONNXRUNTIME_DIR to the onnxruntime GPU package root}"
NVIDIA_RUNTIME_DIR="${NVIDIA_RUNTIME_DIR:?set NVIDIA_RUNTIME_DIR to the nvidia pip wheel directory}"
BINARY="${BINARY:-$(dirname "$0")/../build/real-onnx-ensemble/neuriplo-kserve-runtime}"

if [ ! -x "$BINARY" ]; then
    echo "error: runtime binary not found or not executable: $BINARY (set BINARY)" >&2
    exit 1
fi

export LD_LIBRARY_PATH="${NVIDIA_RUNTIME_DIR}/cudnn/lib:${NVIDIA_RUNTIME_DIR}/cublas/lib:${NVIDIA_RUNTIME_DIR}/cuda_nvrtc/lib:${ONNXRUNTIME_DIR}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "${BINARY}" "$@"
