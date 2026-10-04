# Step 1: KServe Packaging - plan

> Retrospective packet. Step 1 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 1
> snapshot (`plan/STEP1.md`, later `specs/history/steps/STEP1.md`) and the
> Step 1 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Tasks are the step's substeps. Each `T-` section is the implementation
record from the snapshot.

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

> This step was delivered as one unit; the sections above are its single
> task group.
