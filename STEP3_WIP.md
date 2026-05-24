# Step 3 WIP: Model Registry And Executor Abstraction Plan

This file tracks the work-in-progress plan for Step 3. When all exit criteria
are implemented and validated, this should be replaced by a completed
`STEP3.md` snapshot similar to the earlier step snapshots.

## Goal

Introduce a replaceable execution boundary and real model lifecycle state while
keeping inference stubbed. After Step 3, the KServe route layer should no longer
generate stub outputs itself. It should validate V2 requests, call an executor
through an interface, and serialize the executor response. This prepares the
runtime for `neuriplo` integration in Step 4 without changing route behavior.

## Current Baseline

Step 2 completed protocol correctness:

- V2 inference request parsing and validation.
- Request `id` echo.
- Versioned routes.
- Metadata `versions`.
- Request-size enforcement.
- Unit and real-socket HTTP integration tests.

Remaining relevant Step 2 review follow-ups:

- Add tests for invalid `Content-Length`.
- Add tests for `max_request_bytes = 0` and excessive values.
- Consider using `nlohmann/json` for metadata serialization too.
- Avoid keeping stub output generation inside the codec once an executor exists.

## Scope

Step 3 covers:

- `Executor` interface.
- `StubExecutor` implementation.
- Native inference request/response structs between routing, codec, and
  executors.
- Model states and readiness transitions.
- Startup model loading through the registry.
- Metadata ownership in model handles.
- Error mapping from executor/model load failures to structured KServe errors.
- Tests proving `StubExecutor` can be replaced without route changes.

Step 3 does not cover:

- Linking or calling real `neuriplo`.
- Backend-specific tensor buffer conversion.
- Scheduler, queueing, dynamic batching, or worker pools.
- Multiple model hot reload.
- Metrics.
- gRPC.
- Kubernetes end-to-end deployment tests.

## Deliverables

### Module Layout

Move toward the roadmap structure without over-rotating the repository in one
step. Likely files:

```text
src/execution/Executor.hpp
src/execution/StubExecutor.hpp
src/execution/StubExecutor.cpp
src/model/ModelHandle.hpp
src/model/ModelState.hpp
```

If moving existing files would create too much churn, keep the current flat
layout for this step and introduce clear type names first.

### Executor Interface

Add an executor boundary that owns execution behavior, not routing behavior.

Expected shape:

```cpp
struct ExecutionRequest {
    std::optional<std::string> id;
    std::vector<ParsedTensor> inputs;
    std::vector<std::string> requested_outputs;
};

struct ExecutionResponse {
    std::vector<OutputTensor> outputs;
};

class Executor {
  public:
    virtual ~Executor() = default;
    virtual const ModelMetadata &metadata() const = 0;
    virtual ExecutionResponse infer(const ExecutionRequest &request) = 0;
};
```

The exact types should match the final codec design, but the important boundary
is that routes validate/dispatch and executors produce output tensors.

### Stub Executor

Add `StubExecutor` that:

- Owns static model metadata currently hardcoded in `ModelRegistry`.
- Returns deterministic output tensors.
- Honors requested output filtering.
- Uses output metadata consistently.

Decide whether the stub output should remain `[1, 1]` for simple deterministic
tests or move to `[1, 1000]` to match metadata. Prefer matching metadata unless
there is a strong reason not to.

### Codec Refactor

Refactor `KServeV2Codec` so it:

- Parses V2 JSON into native request structs.
- Validates request tensors against metadata.
- Serializes a native `ExecutionResponse`.
- Does not generate stub tensor data itself.
- Uses `nlohmann/json` for model metadata serialization if practical.

This is where the Step 2 review recommendation should land: once executors
exist, the codec serializes executor output rather than inventing output.

### Model Registry And State

Replace the single always-ready metadata-only registry with a model handle:

```text
ModelHandle
  - name
  - versions
  - state
  - metadata
  - executor
  - load error, if any
```

Initial model states:

```text
Loading
Ready
Failed
Unavailable
```

Expected behavior:

- Registry constructs the configured model at startup.
- Registry calls model load/initialization for the selected executor.
- Ready endpoints return true only when the model is `Ready`.
- Failed/unavailable models return stable structured errors.
- `allReady()` reflects model state, not a hardcoded boolean.

For Step 3, `StubExecutor` loading can always succeed unless a test injects a
failure.

### Runtime Configuration

Keep existing CLI/env behavior. Consider adding only what Step 3 needs:

- Backend selection still defaults to `stub`.
- Unknown backend can return a startup/config error or select stub until Step 4.
- Do not add real backend flags yet.

### Error Mapping

Define stable mappings for Step 3:

- Model missing: `404 MODEL_NOT_FOUND`.
- Model loading/unavailable: `503 UNAVAILABLE` or `409 MODEL_NOT_READY`.
- Model failed to load: `503 UNAVAILABLE` with useful message.
- Executor inference error: `500 BACKEND_ERROR` or `500 INTERNAL`.

Pick one mapping and cover it in tests. Keep messages concise and do not leak
paths, secrets, or model internals.

### HTTP And Config Follow-Up Tests

Add the low-risk coverage gaps from Step 2 review:

- Invalid `Content-Length` returns `400 INVALID_ARGUMENT`.
- `--max-request-bytes 0` is rejected.
- `MAX_REQUEST_BYTES=0` is rejected.
- Excessive `max_request_bytes` value is rejected or handled predictably.
- Oversized request rejection covers the declared `Content-Length` path, not
  only accumulated raw bytes.

### Tests

Add focused tests for:

- `StubExecutor` metadata and deterministic inference.
- Runtime inference calls the executor rather than codec-generated output.
- Requested output filtering works through executor response serialization.
- Model readiness reflects model state.
- Failed model load affects `/v2/health/ready` and model ready routes.
- Existing Step 2 route and integration tests continue to pass.

Avoid broad refactors unless tests require them.

## Exit Criteria

Step 3 is complete when:

- A replaceable `Executor` interface exists.
- `StubExecutor` implements the current deterministic inference behavior.
- KServe routes call an executor without changing public endpoint behavior.
- The codec no longer owns stub output generation.
- Model metadata and readiness come from a model handle/state, not hardcoded
  route assumptions.
- Readiness tracks model state.
- Stub executor can be replaced in tests without route changes.
- Step 2 HTTP/config coverage gaps listed above are addressed.
- Local validation passes:

```bash
cmake --preset debug
scripts/check-format.sh
cmake --build --preset debug
ctest --preset debug
```

- Container build still passes:

```bash
docker build -f docker/Dockerfile -t neuriplo-kserve-runtime:step3 .
```

## Completion Process

When all exit criteria pass:

1. Replace this WIP file with `STEP3.md`.
2. Document what was actually implemented.
3. Include validation commands that passed.
4. Document remaining Step 4 integration assumptions for `neuriplo`.
