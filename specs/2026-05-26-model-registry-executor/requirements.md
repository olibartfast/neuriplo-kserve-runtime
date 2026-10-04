# Step 3: Model Registry And Executor Abstraction - requirements

> Retrospective packet. Step 3 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 3
> snapshot (`plan/STEP3.md`, later `specs/history/steps/STEP3.md`) and the
> Step 3 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Roadmap phase: [Phase 0 - Serving Foundation](../roadmap.md#phase-0---serving-foundation-steps-0-11)
Status: **Complete**
Specified: 2026-05-26 (snapshot date)

## Goal

This document records what was implemented for Step 3. Inference remains stubbed,
but routing now delegates execution to a replaceable executor behind a model
handle with real lifecycle state.

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

## In Scope

- Add `Executor` interface.
- Add `StubExecutor`.
- Add model states.
- Add startup model loading.
- Add metadata conversion.

## Requirements

Each requirement is an exit criterion of the original step plan; the
`(n.m)` tag names the substep that owned it.

- [R-1] Stub executor can be replaced without route changes.
- [R-2] Readiness tracks model state.

## Out of Scope

Recorded at the end of the step as remaining gaps; later steps picked them
up as noted in [`../roadmap.md`](../roadmap.md).

### Remaining Assumptions For Step 4

Step 4 (`neuriplo` integration) should:

- Add `NeuriploExecutor` implementing `Executor`.
- Move tensor buffer conversion into the executor/backend layer.
- Load configured backend/model once at startup through the registry factory.
- Keep KServe routes unchanged; only swap the executor implementation.
- Source metadata from `neuriplo` backend APIs instead of stub metadata.
