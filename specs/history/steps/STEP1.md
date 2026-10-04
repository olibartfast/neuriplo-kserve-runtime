# Step 1: KServe Packaging

This document records what was implemented for Step 1. The runtime is still on
the stub model execution path, but it can now be packaged and configured in the
shape expected by a KServe custom runtime.

## Current Scope

Step 1 adds:

- Container image definition.
- KServe `ClusterServingRuntime` manifest.
- Example KServe `InferenceService`.
- KServe/container configuration defaults.
- `/mnt/models` model path convention.
- Basic deployment documentation.
- Tests for new configuration behavior.

Step 1 does not add real `neuriplo` execution, full V2 infer parsing,
scheduling, batching, metrics, gRPC, or LLM endpoints.

## Container Image

Implemented in `docker/Dockerfile`.

- Uses a multi-stage Debian build.
- Builds `neuriplo-kserve-runtime` from source with CMake and Ninja.
- Copies only the runtime binary into the final image.
- Runs as a non-root `runtime` user.
- Exposes port `8080`.
- Starts with:

```text
neuriplo-kserve-runtime --host 0.0.0.0 --port 8080
```

Model artifacts are not included in the image.

## Runtime Configuration

Implemented in `src/RuntimeConfig.cpp` and `src/RuntimeConfig.hpp`.

Environment defaults now supported:

```text
MODEL_NAME
MODEL_PATH
BACKEND
STORAGE_URI
```

Behavior:

- Existing local defaults are preserved:

```text
host:       0.0.0.0
port:       8080
model-name: demo
backend:    stub
```

- CLI flags override environment defaults.
- `STORAGE_URI` is retained in `RuntimeConfig` for diagnostics, but the server
  does not download it.
- If `MODEL_PATH` is unset and `/mnt/models` exists, `model_path` defaults to
  `/mnt/models`.
- Tests use an injectable environment/path abstraction instead of mutating the
  process environment.

## KServe Manifests

Implemented files:

```text
deploy/kserve/cluster-serving-runtime.yaml
deploy/kserve/inferenceservice.yaml
```

The `ClusterServingRuntime`:

- Uses runtime name `neuriplo-kserve-runtime`.
- Declares protocol version `v2`.
- Declares only the conservative model format:

```yaml
supportedModelFormats:
  - name: neuriplo
    version: "1"
    autoSelect: false
```

- Does not claim generic `onnx`, `openvino`, `tensorrt`, or `gguf`
  auto-selection.
- Listens on port `8080`.
- Passes the KServe InferenceService name to `--model-name` using the KServe
  `{{.Name}}` template variable.
- Sets `MODEL_PATH=/mnt/models` and `BACKEND=stub`.

The example `InferenceService`:

- Uses `modelFormat.name: neuriplo`.
- Uses `protocolVersion: v2`.
- Explicitly selects `runtime: neuriplo-kserve-runtime`.
- Uses placeholder `storageUri: pvc://your-model-pvc/path/to/model`.
- Includes CPU-friendly requests and limits.

## Documentation

`README.md` now documents:

- Local image build command.
- Applying the `ClusterServingRuntime`.
- Applying the example `InferenceService`.
- Basic readiness and metadata curl checks.
- The current limitation that inference is still stubbed.

## Tests

New focused coverage in `tests/RuntimeConfigTest.cpp` checks:

- Environment defaults are read.
- CLI flags override environment defaults.
- `/mnt/models` is selected when present and `MODEL_PATH` is unset.
- `MODEL_PATH` overrides the `/mnt/models` convention.
- Existing defaults, CLI parsing, and invalid port validation still work.

## Validation

Validation commands run for this step:

```bash
cmake --build --preset debug
ctest --preset debug
docker build -f docker/Dockerfile -t neuriplo-kserve-runtime:step1 .
docker run --rm neuriplo-kserve-runtime:step1 --version
docker run --rm neuriplo-kserve-runtime:step1 --help
```
