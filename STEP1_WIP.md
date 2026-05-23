# Step 1 WIP: KServe Packaging Plan

This file tracks the work-in-progress plan for Step 1. When all exit
criteria are implemented and validated, this should be replaced by a completed
`STEP1.md` snapshot similar to `STEP0.md`.

## Goal

Make the current scaffold deployable as a KServe custom runtime, while still
using the stub model execution path. Step 1 should not add real `neuriplo`
execution yet; it should make the runtime package, configure, and run in the
shape KServe expects.

## Scope

Step 1 covers:

- Container image definition.
- KServe `ClusterServingRuntime` manifest.
- Example KServe `InferenceService`.
- KServe/container configuration defaults.
- `/mnt/models` model path convention.
- Basic deployment documentation.
- Tests for new configuration behavior.

Step 1 does not cover:

- Real `neuriplo` backend execution.
- Full V2 infer request parsing.
- Scheduler, batching, or backpressure.
- gRPC.
- Metrics.
- llama.cpp, Cactus, or OpenAI-compatible endpoints.

## Deliverables

### Dockerfile

Add:

```text
docker/Dockerfile
```

Requirements:

- Build the C++ runtime from source.
- Produce a small runtime image containing `neuriplo-kserve-runtime`.
- Run as a non-root user if practical.
- Expose port `8080`.
- Default command should start the server with KServe-friendly defaults.
- Do not include model artifacts in the image.

Initial command shape:

```text
neuriplo-kserve-runtime --host 0.0.0.0 --port 8080
```

Runtime configuration should come from explicit CLI flags or environment
defaults in the binary.

### Runtime Config Defaults

Update `RuntimeConfig` parsing so container/KServe defaults work without a long
argument list.

Expected behavior:

- `MODEL_NAME` can set the default model name.
- `MODEL_PATH` can set the default model path.
- `STORAGE_URI` is not downloaded by the server, but may be logged or retained
  for diagnostics.
- If `MODEL_PATH` is unset and `/mnt/models` exists, default `model_path` to
  `/mnt/models`.
- `BACKEND` can set the default backend.
- CLI flags override environment defaults.

Keep existing local defaults:

```text
model_name: demo
backend: stub
host: 0.0.0.0
port: 8080
```

### ClusterServingRuntime Manifest

Add:

```text
deploy/kserve/cluster-serving-runtime.yaml
```

Requirements:

- Define a `ClusterServingRuntime`.
- Use runtime name `neuriplo-kserve-runtime`.
- Declare protocol version `v2`.
- Declare conservative model format:

```yaml
supportedModelFormats:
  - name: neuriplo
    version: "1"
    autoSelect: false
```

- Do not claim generic `onnx`, `openvino`, `tensorrt`, or `gguf` auto-selection
  in Step 1.
- Container should listen on port `8080`.
- Arguments/env should map KServe model configuration into the runtime.

### InferenceService Example

Add:

```text
deploy/kserve/inferenceservice.yaml
```

Requirements:

- Use `modelFormat.name: neuriplo`.
- Use `protocolVersion: v2`.
- Explicitly select `runtime: neuriplo-kserve-runtime`.
- Use a placeholder `storageUri` that documents where users should put their
  own model artifact.
- Include CPU-friendly resource requests/limits.
- Keep the example safe for local KServe/minikube-style testing.

### Deployment Notes

Add documentation to `README.md` or a file under `deploy/kserve/`.

It should explain:

- Build the container image.
- Apply the `ClusterServingRuntime`.
- Apply the example `InferenceService`.
- Confirm readiness.
- Curl the basic V2 endpoints.
- Current limitation: inference is still a stub.

### Tests

Add focused tests for configuration behavior:

- Environment defaults are read.
- CLI flags override environment defaults.
- `/mnt/models` default is selected when appropriate.
- Existing CLI parsing behavior still works.

If direct environment mutation would make tests brittle, add a small injectable
environment abstraction around config parsing.

## Exit Criteria

Step 1 is complete when:

- `docker/Dockerfile` exists and builds the runtime image.
- `deploy/kserve/cluster-serving-runtime.yaml` exists.
- `deploy/kserve/inferenceservice.yaml` exists.
- Runtime config supports KServe/container defaults.
- Tests cover the new config behavior.
- Local build and tests pass:

```bash
cmake --build --preset debug
ctest --preset debug
```

- Documentation clearly says that inference is still stubbed.

## Completion Process

When all exit criteria pass:

1. Replace this WIP file with `STEP1.md`.
2. Document what was actually implemented.
3. Include the validation commands that passed.
4. Remove or archive any stale WIP notes.
