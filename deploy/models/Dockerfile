# Model artifact image.
#
# Carries exported ONNX files and stages them on request. Staging is this
# image's own job so that no orchestrator -- Kubernetes, compose, or a bare
# docker run -- has to describe the step.
#
# The build context is a directory of *.onnx files. Nothing here names a model;
# whatever is in the context is what gets served, and each file's basename
# becomes the served model name.
#
# Build with deploy/models/build-model-image.sh, which produces the context.
# Pinned to a version tag and digest so a rebuild cannot silently change the base.
FROM busybox:1.37.0-glibc@sha256:42a9ff19af33811ee466b24c66384461d37c32dcc28d1d27d49b8aca816bd5ee

COPY *.onnx /artifacts/

ENV STAGE_DIR=/staging

# The staging directory is cleared first: it is discardable by contract, and a
# persistent volume would otherwise keep artifacts from a previous image.
CMD ["sh", "-c", "mkdir -p \"${STAGE_DIR:?}\" && rm -rf \"${STAGE_DIR:?}\"/* \"${STAGE_DIR:?}\"/.[!.]* && cp /artifacts/*.onnx \"$STAGE_DIR\"/ && ls -l \"$STAGE_DIR\""]
