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
FROM busybox:glibc

COPY *.onnx /artifacts/

ENV STAGE_DIR=/staging

CMD ["sh", "-c", "mkdir -p \"$STAGE_DIR\" && cp /artifacts/*.onnx \"$STAGE_DIR\"/ && ls -l \"$STAGE_DIR\""]
