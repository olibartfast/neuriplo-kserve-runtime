# Step 3: Model Registry And Executor Abstraction

This document records what was implemented for Step 3. Inference remains stubbed,
but routing now delegates execution to a replaceable executor behind a model
handle with real lifecycle state.

## Current Scope

Step 3 adds:

- `Executor` interface and `StubExecutor` implementation.
- `ModelHandle` with `Loading`, `Ready`, `Failed`, and `Unavailable` states.
- Startup model loading through `ModelRegistry` with injectable executor factory
  for tests.
- Metadata ownership in model handles and executors.
- KServe route inference through executors instead of codec-generated stub data.
- `nlohmann/json` serialization for model metadata and inference responses.
- Error mapping for failed model load and not-ready models.
- Step 2 review follow-up tests for invalid `Content-Length`, zero/excessive
  `max_request_bytes`, and declared `Content-Length` over-limit rejection.

Step 3 does not add real `neuriplo` integration, scheduler/queueing, dynamic
batching, metrics, gRPC, or Kubernetes end-to-end deployment tests.

## Module Layout

New and updated files:

```text
src/ModelMetadata.hpp
src/ModelState.hpp
src/Executor.hpp
src/StubExecutor.hpp
src/StubExecutor.cpp
src/ModelHandle.hpp
src/ModelRegistry.hpp
src/ModelRegistry.cpp
src/KServeV2Codec.hpp
src/KServeV2Codec.cpp
src/KServeRuntime.cpp
```

## Executor Boundary

`Executor` owns execution behavior. Routes validate and dispatch; executors
produce output tensors.

```cpp
struct ExecutionRequest {
    std::optional<std::string> id;
    std::vector<std::string> requested_outputs;
};

struct ExecutionResponse {
    std::vector<OutputTensor> outputs;
};

class Executor {
  public:
    virtual const ModelMetadata &metadata() const = 0;
    virtual ExecutionResponse infer(const ExecutionRequest &request) = 0;
};
```

`StubExecutor` returns deterministic zero-filled outputs shaped to match model
metadata (`FP32 [1, 1000]` for the demo model output).

## Model Registry And State

`ModelRegistry` constructs one `ModelHandle` at startup:

- Creates an executor through a factory (`stub` backend only for now).
- Transitions to `Ready` on success or `Failed` with a load error message.
- Exposes metadata and readiness from handle state instead of hardcoded booleans.

Readiness mapping:

- Missing model: `404 MODEL_NOT_FOUND`
- Not ready (loading): `409 MODEL_NOT_READY`
- Failed/unavailable: `503 UNAVAILABLE`
- Runtime `/v2/health/ready`: `503 UNAVAILABLE` when any startup model is not ready

Unknown backends (other than `stub`) fail executor creation at startup.

## Codec Refactor

`KServeV2Codec` now:

- Parses and validates V2 inference requests.
- Serializes executor `ExecutionResponse` values.
- Serializes model metadata with `nlohmann/json`.
- Does not generate stub tensor data.

## Tests

Added:

```text
tests/StubExecutorTest.cpp
tests/ModelRegistryTest.cpp
```

Updated:

```text
tests/KServeRuntimeTest.cpp
tests/KServeV2CodecTest.cpp
tests/RuntimeConfigTest.cpp
tests/HttpIntegrationTest.cpp
```

Coverage includes:

- Stub executor metadata and deterministic inference output shape.
- Registry startup load success and injected failure.
- Injected executor used by runtime without route changes.
- Codec serializes executor responses (not locally invented stub tensors).
- Failed model load affects `/v2/health/ready` and model ready routes.
- Invalid `Content-Length`, zero/excessive `max_request_bytes`, and declared
  `Content-Length` over-limit HTTP behavior.

## Validation

Validation commands run for this step:

```bash
cmake --preset debug
scripts/check-format.sh
cmake --build --preset debug
ctest --preset debug
docker build -f docker/Dockerfile -t neuriplo-kserve-runtime:step3 .
```

## Remaining Assumptions For Step 4

Step 4 (`neuriplo` integration) should:

- Add `NeuriploExecutor` implementing `Executor`.
- Move tensor buffer conversion into the executor/backend layer.
- Load configured backend/model once at startup through the registry factory.
- Keep KServe routes unchanged; only swap the executor implementation.
- Source metadata from `neuriplo` backend APIs instead of stub metadata.
