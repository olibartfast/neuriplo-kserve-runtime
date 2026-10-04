# Step 11: Deployment Validation - requirements

> Retrospective packet. Step 11 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 11
> snapshot (`plan/STEP11.md`, later `specs/history/steps/STEP11.md`) and the
> Step 11 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Roadmap phase: [Phase 0 - Serving Foundation](../roadmap.md#phase-0---serving-foundation-steps-0-11)
Status: **Complete**
Specified: 2026-06-05 (snapshot date)

## Goal

**Completed:** 2026-06-04
**Snapshot:** All 4 substeps implemented and tested.

Completed deployment validation: InferenceGraph routing compatibility tests, canary rollout
metrics labels, autoscaling manifests and load testing, and comprehensive error documentation.

## In Scope

Step 11 validates the runtime end-to-end in a KServe/Kubernetes context
and documents behaviour for external consumers.

### 11.1 InferenceGraph Routing Validation

- Integration test proving the runtime response shape survives InferenceGraph
  `Sequence`, `Switch`, and `Splitter` routing.
- Validate that metadata and error responses are compatible with graph
  routing semantics.

### 11.2 Canary Rollout Validation

- Test that `canaryTrafficPercent` works correctly with runtime-image
  and model-artifact changes.
- Validate metrics labels distinguish canary vs stable traffic.

### 11.3 Autoscaling Integration Tests

- Validate HPA/KEDA/custom autoscaler behaviour with the runtime's
  exposed metrics (queue depth, inflight, latency).
- Document recommended autoscaling thresholds.

### 11.4 Structured Error Documentation

- Document the full error taxonomy for external clients.
- Include HTTP status codes, error body shapes, and recovery guidance.
- Add examples for each error category.

## Requirements

Each requirement is an exit criterion of the original step plan; the
`(n.m)` tag names the substep that owned it.

- [R-1] (11.1) An InferenceGraph with the runtime as a node completes successfully.
- [R-2] (11.1) Error responses from the runtime do not break graph routing.
- [R-3] (11.2) Canary traffic split produces distinguishable metrics.
- [R-4] (11.2) Rollout does not cause spurious readiness flips.
- [R-5] (11.3) Autoscaler scales up under load and down when idle.
- [R-6] (11.3) Scaling events are visible in metrics/logs.
- [R-7] (11.4) Client-facing error docs exist and cover every `KServeErrors` category.
- [R-8] (11.4) Docs are kept in the repository.

Not every requirement was fully met; the delivered status of each is in
[`validation.md`](validation.md#deviations).

## Out of Scope

Nothing recorded beyond the later steps in the target design.
