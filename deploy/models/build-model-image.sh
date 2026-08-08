#!/usr/bin/env bash
# Build a model artifact image from exported ONNX files.
#
# Exists so the image that feeds a deployment is reproducible from this repo.
# Model files are deliberately not committed (see .gitignore and the export
# guidance in neuriplo-tasks), so the recipe is committed instead.
#
# Sources may be given as:
#   - a path to an existing .onnx file
#   - an Ultralytics model name, which is exported on demand
#
#   ./build-model-image.sh --tag neuriplo-models:depth-v1 \
#       yolo26n-depth yolo26l-depth
#
#   ./build-model-image.sh --tag neuriplo-models:mixed-v1 \
#       /path/to/detector.onnx yolo26n-depth
#
# Nothing about the resulting image is model-specific: the serving side reads
# whatever it contains and takes each model's name from its filename.
set -euo pipefail

TAG=""
IMGSZ=768
IMPORT_CLUSTER=""
SOURCES=()

usage() {
    cat >&2 <<'USAGE'
usage: build-model-image.sh --tag <image:tag> [--imgsz N] [--k3d-import <cluster>]
                            <source> [<source> ...]

  --tag           image tag to build (required)
  --imgsz         export size for Ultralytics sources (default 768)
  --k3d-import    import the built image into the named k3d cluster
  <source>        path to an .onnx file, or an Ultralytics model name
USAGE
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --tag) TAG="${2:?--tag needs a value}"; shift 2 ;;
        --imgsz) IMGSZ="${2:?--imgsz needs a value}"; shift 2 ;;
        --k3d-import) IMPORT_CLUSTER="${2:?--k3d-import needs a value}"; shift 2 ;;
        -h|--help) usage ;;
        --*) echo "unknown option: $1" >&2; usage ;;
        *) SOURCES+=("$1"); shift ;;
    esac
done

[ -n "$TAG" ] || usage
[ ${#SOURCES[@]} -gt 0 ] || usage

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
context=$(mktemp -d)
# The context holds copies of large model files, so clean it up on every exit
# path rather than leaving it in /tmp.
trap 'rm -rf "$context"' EXIT

for source in "${SOURCES[@]}"; do
    if [ -f "$source" ]; then
        echo "using existing export: $source"
        cp "$source" "$context/"
        continue
    fi

    if ! command -v yolo >/dev/null 2>&1; then
        echo "error: '$source' is not a file and the Ultralytics CLI is not installed" >&2
        echo "       install it with: python -m pip install 'ultralytics[export]'" >&2
        exit 1
    fi

    echo "exporting $source at imgsz=$IMGSZ"
    (
        cd "$context"
        yolo export "model=${source}.pt" format=onnx "imgsz=${IMGSZ}"
    )
    # The checkpoint is a build input, not an artifact to ship.
    rm -f "$context/${source}.pt"
done

cp "$script_dir/Dockerfile" "$context/Dockerfile"
docker build -t "$TAG" "$context"
echo "built $TAG with: $(cd "$context" && ls ./*.onnx | tr '\n' ' ')"

if [ -n "$IMPORT_CLUSTER" ]; then
    k3d image import "$TAG" -c "$IMPORT_CLUSTER"
fi
