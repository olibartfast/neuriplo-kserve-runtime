# Standalone model-repository preparer.
#
# Builds a versioned model repository tree from staged artifacts and exits. It
# serves nothing and links to no server, so it works as an init container in
# front of Triton, OpenVINO Model Server, neuriplo-kserve-runtime, or anything
# else that reads <model>/<version>/<file>. Point REPOSITORY_LAYOUT at whichever
# one will mount the volume next.
#
#   docker build -f deploy/prepare/Dockerfile -t neuriplo-prepare:trt .
#
# The base carries trtexec because compiling ONNX to a TensorRT engine is the one
# preparation step that cannot be done anywhere else: an engine is specific to
# the GPU, driver, and TensorRT version that built it. Two consequences follow.
#
#   - This container needs a GPU. As a Kubernetes init container it must request
#     nvidia.com/gpu, and it holds the device for the length of the build.
#   - Its TensorRT version must match the server's. Triton 25.12 and this base
#     both ship TensorRT 10.14; pairing mismatched versions produces an engine
#     the server refuses to load, reported as a version mismatch at load time.
#
# For a repository with no TensorRT in it, a much smaller base is enough --
# preparation is then only copying and renaming:
#
#   docker build -f deploy/prepare/Dockerfile --build-arg BASE_IMAGE=busybox:glibc .
ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt:25.12-py3

FROM ${BASE_IMAGE}

COPY deploy/prepare/prepare-repository.sh /usr/local/bin/prepare-repository.sh
RUN chmod +x /usr/local/bin/prepare-repository.sh

ENV STAGE_DIR=/staging
ENV MODEL_REPOSITORY=/models/repo
# Build and exit. This image never starts a server; that is the whole point of
# it being separable from one.
ENV PREPARE_ONLY=true

ENTRYPOINT ["/usr/local/bin/prepare-repository.sh"]
