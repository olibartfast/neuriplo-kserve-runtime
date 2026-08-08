#!/bin/sh
# Build TensorRT engines from staged ONNX files, place them in a model
# repository tree, then serve the tree.
#
# This runs inside the serving container, on the node that will serve, because a
# TensorRT engine is specific to the GPU, driver, and TensorRT version it was
# built against. An engine built anywhere else is not portable here, which is why
# conversion cannot move to image build time or to the host.
#
# Every *.onnx in STAGE_DIR is converted; the file's basename becomes the model
# name. Layout produced (the tree --models expects):
#   $MODEL_REPOSITORY/<model-name>/$MODEL_VERSION/model.plan
#
# Environment:
#   STAGE_DIR         directory the init containers copied .onnx files into
#   MODEL_REPOSITORY  repository root to build and serve (default /models/repo)
#   MODEL_VERSION     version directory to write (default 1)
#   TRT_PRECISION     fp16 (default), fp32, or best
#   TRT_SHAPES        trtexec shape spec for models with DYNAMIC axes, e.g.
#                     images:1x3x768x768. Leave unset for a static ONNX --
#                     trtexec rejects explicit shapes on a static model with
#                     "Static model does not take explicit shapes".
#   TRT_EXTRA_ARGS    additional trtexec arguments
#   TRT_FALLBACK_ONNX when true, serve the ONNX if trtexec is unavailable
set -eu

STAGE_DIR="${STAGE_DIR:-/staging}"
MODEL_REPOSITORY="${MODEL_REPOSITORY:-/models/repo}"
MODEL_VERSION="${MODEL_VERSION:-1}"
TRT_PRECISION="${TRT_PRECISION:-fp16}"
TRT_FALLBACK_ONNX="${TRT_FALLBACK_ONNX:-false}"

if [ ! -d "$STAGE_DIR" ]; then
    echo "error: staging directory not found: $STAGE_DIR" >&2
    exit 1
fi

staged=""
for source_onnx in "$STAGE_DIR"/*.onnx; do
    [ -f "$source_onnx" ] || continue
    staged="yes"
done
if [ -z "$staged" ]; then
    echo "error: no .onnx files staged in $STAGE_DIR" >&2
    ls -l "$STAGE_DIR" >&2 || true
    exit 1
fi

convert_one() {
    source_onnx="$1"
    model_name=$(basename "$source_onnx" .onnx)
    target_dir="$MODEL_REPOSITORY/$model_name/$MODEL_VERSION"
    target_engine="$target_dir/model.plan"

    mkdir -p "$target_dir"

    if [ -f "$target_engine" ]; then
        # Surviving a restart without rebuilding matters: conversion of a real
        # model takes minutes, and the startup probe has to tolerate it once.
        echo "engine already present, skipping conversion: $target_engine"
        return 0
    fi

    if command -v trtexec >/dev/null 2>&1; then
        echo "converting $source_onnx -> $target_engine (precision=$TRT_PRECISION)"

        set -- --onnx="$source_onnx" --saveEngine="$target_engine.tmp"
        case "$TRT_PRECISION" in
            fp16) set -- "$@" --fp16 ;;
            best) set -- "$@" --best ;;
            fp32) ;;
            *)
                echo "error: unsupported TRT_PRECISION: $TRT_PRECISION" >&2
                exit 1
                ;;
        esac
        if [ -n "${TRT_SHAPES:-}" ]; then
            set -- "$@" --shapes="$TRT_SHAPES"
        fi
        if [ -n "${TRT_EXTRA_ARGS:-}" ]; then
            # Intentionally unquoted: TRT_EXTRA_ARGS carries multiple arguments.
            # shellcheck disable=SC2086
            set -- "$@" $TRT_EXTRA_ARGS
        fi

        trtexec "$@"
        # Publish atomically so a conversion killed partway through is not
        # mistaken for a finished engine on the next start.
        mv "$target_engine.tmp" "$target_engine"
        echo "engine written: $target_engine"
    elif [ "$TRT_FALLBACK_ONNX" = "true" ]; then
        # Explicitly opted into: serving the ONNX is a different execution path
        # with different latency, so it is never a silent substitution.
        echo "warning: trtexec not found; TRT_FALLBACK_ONNX=true, serving ONNX" >&2
        cp "$source_onnx" "$target_dir/model.onnx"
    else
        echo "error: trtexec not found in PATH and TRT_FALLBACK_ONNX is not true" >&2
        echo "       this image needs TensorRT installed to build an engine" >&2
        exit 1
    fi
}

for source_onnx in "$STAGE_DIR"/*.onnx; do
    [ -f "$source_onnx" ] || continue
    convert_one "$source_onnx"
done

echo "serving repository $MODEL_REPOSITORY"
exec neuriplo-kserve-runtime --models="$MODEL_REPOSITORY" "$@"
