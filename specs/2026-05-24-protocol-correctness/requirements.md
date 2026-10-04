# Step 2: Protocol Correctness - requirements

> Retrospective packet. Step 2 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 2
> snapshot (`plan/STEP2.md`, later `specs/history/steps/STEP2.md`) and the
> Step 2 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Roadmap phase: [Phase 0 - Serving Foundation](../roadmap.md#phase-0---serving-foundation-steps-0-11)
Status: **Complete**
Specified: 2026-05-24 (snapshot date)

## Goal

This document records what was implemented for Step 2. The runtime still uses
stub inference, but the KServe/Open Inference Protocol V2 request and response
surface is now parsed and validated instead of ignoring inference request
bodies.

Step 2 adds:

- KServe V2 inference request parsing.
- Deterministic stub inference response serialization.
- Request `id` preservation.
- Model metadata `versions`.
- Versioned model metadata, readiness, and inference routes.
- Structured `400`, `404`, `405`, `409`, and `413` error behavior.
- Runtime `max_request_bytes` configuration.
- HTTP-layer oversized request and invalid `Content-Length` handling.
- Unit tests for codec, routing, config, and validation behavior.
- HTTP integration tests that exercise the server over a real socket.

Step 2 does not add real `neuriplo` execution, executor abstraction, scheduler,
batching, queueing, metrics, gRPC, Kubernetes end-to-end tests, or LLM/OpenAI
endpoints.

## In Scope

- Add KServe V2 request/response codec.
- Add structured errors.
- Add route tests.
- Add content-length/body-size enforcement.
- Add request id handling.
- Add model version route parsing and metadata `versions`.
- Validate tensor names, shapes, datatypes, and requested outputs.

## Requirements

Each requirement is an exit criterion of the original step plan; the
`(n.m)` tag names the substep that owned it.

- [R-1] Invalid model returns stable 404.
- [R-2] Invalid JSON returns stable 400.
- [R-3] Stub inference echoes deterministic tensor response.
- [R-4] Request `id` is preserved in the response.
- [R-5] Unsupported datatype/output requests fail predictably.

## Out of Scope

Recorded at the end of the step as remaining gaps; later steps picked them
up as noted in [`../roadmap.md`](../roadmap.md).

### Remaining Gaps

Deferred to later steps:

- Real tensor buffer conversion.
- Tensor data length validation.
- Executor abstraction and backend errors.
- Scheduler, queue limits, timeouts, and concurrency controls.
- Full Kubernetes end-to-end deployment validation.
