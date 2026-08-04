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

ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR:-/home/oli/dependencies/onnxruntime-linux-x64-gpu-1.19.2}"
NVIDIA_RUNTIME_DIR="${NVIDIA_RUNTIME_DIR:-/home/oli/dependencies/nvidia-cuda-runtime/nvidia}"
BINARY="${BINARY:-$(dirname "$0")/../build/real-onnx-ensemble/neuriplo-kserve-runtime}"

export LD_LIBRARY_PATH="${NVIDIA_RUNTIME_DIR}/cudnn/lib:${NVIDIA_RUNTIME_DIR}/cublas/lib:${NVIDIA_RUNTIME_DIR}/cuda_nvrtc/lib:${ONNXRUNTIME_DIR}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "${BINARY}" "$@"
