# Step 3: Model Registry And Executor Abstraction - plan

> Retrospective packet. Step 3 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 3
> snapshot (`plan/STEP3.md`, later `specs/history/steps/STEP3.md`) and the
> Step 3 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Tasks are the step's substeps. Each `T-` section is the implementation
record from the snapshot.

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

> This step was delivered as one unit; the sections above are its single
> task group.
