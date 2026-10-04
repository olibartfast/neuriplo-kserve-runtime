# Step 4: Atomic neuriplo Integration - requirements

> Retrospective packet. Step 4 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 4
> snapshot (`plan/STEP4.md`, later `specs/history/steps/STEP4.md`) and the
> Step 4 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Roadmap phase: [Phase 0 - Serving Foundation](../roadmap.md#phase-0---serving-foundation-steps-0-11)
Status: **Complete**
Specified: 2026-05-27 (snapshot date)

## Goal

This snapshot records the Step 4 implementation state. The runtime now has an
executor-facing tensor request path, a `neuriplo` adapter boundary, a
`NeuriploExecutor`, backend selection for known real backend ids, and optional
CMake wiring for a sibling `neuriplo` checkout.

## In Scope

Step 4 integrates real `neuriplo` execution in independently buildable
substeps. Routes already delegate through `ModelRegistry` and `Executor`, but
`ExecutionRequest` does not yet preserve parsed input tensor data. Real backend
execution must start by carrying validated KServe inputs through the executor
boundary.

### 4.1 Preserve KServe Input Tensors

- Add an input tensor representation to executor-facing request types.
- Have `KServeV2Codec` validate and pass input `name`, `datatype`, `shape`, and
  `data` into `ExecutionRequest`.
- Keep `StubExecutor` behavior unchanged.

### 4.2 Define A neuriplo Adapter Boundary

- Add a small internal adapter interface or wrapper so tests can fake
  `neuriplo` behavior before linking the real library.
- Keep `neuriplo`-specific types out of HTTP, routing, and registry code.
- Keep `Executor` as the only runtime execution interface used by
  `ModelRegistry` and routes.

### 4.3 Add NeuriploExecutor Skeleton

- Implement `Executor` using the adapter boundary.
- Load model metadata from the adapter.
- Support construction or load failure with stable registry failed-state
  behavior.

### 4.4 Wire Backend Selection

- Keep `stub` mapped to `StubExecutor`.
- Map explicitly supported real backend ids to `NeuriploExecutor`.
- Keep unknown backend ids failing predictably.

### 4.5 Convert Request Inputs

- Convert supported KServe JSON tensor datatypes and shapes into `neuriplo`
  input buffers.
- Start with the minimum datatype needed by the chosen smoke backend, likely
  `FP32`.
- Return stable inference errors for unsupported or malformed backend-bound
  tensors.

### 4.6 Convert Backend Outputs

- Convert dense `neuriplo` outputs into `ExecutionResponse`.
- Preserve requested-output filtering semantics.
- Keep KServe response JSON shape compatible with existing tests.

### 4.7 Link Real neuriplo

- Add the selected CMake discovery path: package, sibling checkout, or vendored
  path.
- Keep the build usable without real `neuriplo` unless the real backend option
  is enabled.

### 4.8 Add Smoke Model Validation

- Pick one first backend, preferably ONNX Runtime unless the `neuriplo` API
  strongly favors OpenCV DNN.
- Add a small model fixture or documented local smoke command.
- Compare runtime `/infer` output against direct backend execution.

## Requirements

Each requirement is an exit criterion of the original step plan; the
`(n.m)` tag names the substep that owned it.

- [R-1] (4.1) Codec tests prove parsed input tensor data reaches `ExecutionRequest`.
- [R-2] (4.1) Stub inference responses remain compatible with existing tests.
- [R-3] (4.2) Unit tests can inject fake backend metadata, load failures, and inference results without linking real `neuriplo`.
- [R-4] (4.2) Public route and registry types do not include `neuriplo` headers.
- [R-5] (4.3) Fake-adapter tests cover load success, load failure, metadata mapping, and readiness behavior.
- [R-6] (4.3) Failed real-backend startup keeps `/v2/health/ready` false.
- [R-7] (4.4) Factory tests cover `stub`, supported real backend ids, and unknown backend ids.
- [R-8] (4.4) Unsupported backend configuration produces a stable startup or readiness failure.
- [R-9] (4.5) Fake-adapter tests cover inference success, unsupported datatypes, malformed shapes, and backend-bound tensor validation errors.
- [R-10] (4.5) Dense JSON tensor data is supported for the first smoke backend.
- [R-11] (4.6) Fake-adapter tests cover dense output conversion and requested-output filtering.
- [R-12] (4.6) Existing KServe response shape tests continue to pass.
- [R-13] (4.7) Default `stub` builds still work without `neuriplo` installed.
- [R-14] (4.7) Enabling the real backend option links the selected `neuriplo` target.
- [R-15] (4.8) ONNX Runtime or OpenCV DNN smoke model runs through `/infer`.
- [R-16] (4.8) Metadata comes from `neuriplo`.
- [R-17] (4.8) Single request output matches direct backend output.

## Out of Scope

Recorded at the end of the step as remaining gaps; later steps picked them
up as noted in [`../roadmap.md`](../roadmap.md).

### Remaining Gaps

The real adapter currently assumes dense `FP32` inputs and dense tensor outputs
because that is the minimum safe conversion surface exposed by the current
sibling `neuriplo` API.
