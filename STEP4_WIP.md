# Step 4 WIP: neuriplo Integration

This document tracks the planned Step 4 work. Step 3 created the executor
boundary and model lifecycle state; Step 4 should replace the stub executor
with a real `neuriplo` executor while keeping KServe routing unchanged.

## Goal

Load one configured model through `neuriplo` at startup and execute KServe V2
`/infer` requests through the selected backend.

## Planned Scope

- Link the runtime against the `neuriplo` library or supported local checkout.
- Add `NeuriploExecutor` implementing the existing `Executor` interface.
- Extend the default executor factory to select:
  - `stub` -> `StubExecutor`
  - supported `neuriplo` backend ids -> `NeuriploExecutor`
- Load the configured backend and model once during `ModelRegistry` startup.
- Source `ModelMetadata` from `neuriplo` backend/model metadata APIs.
- Convert validated KServe request tensors into `neuriplo` input buffers.
- Convert `neuriplo` inference outputs into `ExecutionResponse` tensors.
- Map backend load and inference errors to stable KServe error responses.

## Out Of Scope

- Request queueing, batching, timeouts, and overload behavior.
- Multi-model registry support.
- gRPC V2.
- Metrics and tracing.
- OpenAI-compatible `/v1/*` endpoints.
- LLM token scheduling.

## Integration Constraints

- Do not add `neuriplo` types to route handlers or HTTP parsing code.
- Keep KServe routes delegating only through `ModelRegistry` and `Executor`.
- Keep tensor protocol details in codec types and backend conversion code.
- Preserve existing `stub` backend behavior and tests.
- Keep startup readiness tied to `ModelHandle` state.

## Acceptance Criteria

- `--backend stub` continues to pass the existing test suite.
- A supported real backend can load a smoke model at startup.
- `/v2/health/ready` is false when real model loading fails.
- `/v2/models/{model}` metadata comes from `neuriplo` for real backends.
- A single `/infer` request returns backend-produced output data.
- Runtime output for the smoke model matches direct backend execution.
- Unknown backend ids fail predictably at startup or readiness.

## WIP Tasks

- Identify the exact `neuriplo` CMake target and include paths.
- Decide how local development locates `neuriplo`:
  - vendored subdirectory
  - CMake package
  - sibling checkout path
- Add minimal `NeuriploExecutor` skeleton behind a build option if needed.
- Add conversion helpers for supported datatypes and tensor shapes.
- Add focused tests with an injected fake `neuriplo` adapter before linking the
  real backend where practical.
- Add one smoke/integration test path for a small supported model.

## Open Questions

- Which backend should be the first smoke target: ONNX Runtime or OpenCV DNN?
- Does `neuriplo` expose stable metadata before the first inference call?
- What ownership and thread-safety guarantees does a loaded backend instance
  provide?
- Are outputs always dense tensors compatible with KServe JSON `data`, or do
  some backends require binary tensor support first?

## Step 3 Handoff Notes

- Version routing now follows executor metadata through
  `ModelRegistry::defaultVersion`.
- The current implementation still returns synthetic partial metadata for a
  failed startup load. Before real backend integration, decide whether metadata
  should return `503 UNAVAILABLE` for failed models or expose last-known-good
  metadata only when deliberately cached.

## Validation Checklist

```bash
cmake --preset debug
scripts/check-format.sh
cmake --build --preset debug
ctest --preset debug
```

Add the real-backend smoke command here once the first `neuriplo` backend is
linked and a small model fixture is selected.
