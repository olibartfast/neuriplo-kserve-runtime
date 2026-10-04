# Step 9: Production Hardening - Tensor Path - requirements

> Retrospective packet. Step 9 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 9
> snapshot (`plan/STEP9.md`, later `specs/history/steps/STEP9.md`) and the
> Step 9 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Roadmap phase: [Phase 0 - Serving Foundation](../roadmap.md#phase-0---serving-foundation-steps-0-11)
Status: **Complete**
Specified: 2026-06-04 (snapshot date)

## Goal

**Completed:** 2026-06-04
**Snapshot:** All 6 substeps implemented and tested.

Production-hardened the tensor serving path: multi-datatype adapter pipeline, formal model state machine, composable request pipeline, UUID-based request IDs, Kubernetes probe documentation, and latency SLO benchmarks.

## In Scope

Step 9 closes the gaps blocking production tensor/CV serving. Each substep is
atomic: independent to build, validate, and ship.

### 9.1 Multi-Datatype Neuriplo Adapter

- Extend `NeuriploExecutor` beyond dense `FP32` to INT8, FP16, INT32, and
  the datatypes used by the first target model.
- Add dtype-conversion validation at the adapter boundary.
- Reject unsupported datatypes with stable error codes.

### 9.2 Formal Model State Machine

- Implement explicit `UNLOADED → LOADING → READY → UNLOADING → UNLOADED`
  transitions in `ModelRegistry`/`ModelHandle`.
- Replace scattered readiness checks with transition helpers.
- Reject or log invalid state transitions.
- Align drain behavior (`beginDrain()`) with `UNAVAILABLE` state.

### 9.3 Composable Request Pipeline

- Refactor `KServeRuntime::handleInfer()` into a chain:
  `decode → validate → admit → schedule → encode`.
- Each stage is a plain function or small callable; no framework required.
- Extension points: request ID injection, rate limits, auth hooks.

### 9.4 Collision-Resistant Request IDs

- Replace counter-based request ID generation with UUID v4 or similar.
- Preserve client-supplied `id` passthrough.
- Propagate IDs consistently through all trace spans and structured logs.

### 9.5 Container Startup Probe Documentation

- Document Kubernetes `startupProbe`, `initialDelaySeconds`, and
  `readinessGates` integration for model load-and-ready window.
- Validate liveness/readiness under load in a real or mock cluster.
- Smoke-test the `ClusterServingRuntime` manifest end-to-end.

### 9.6 Latency SLO Benchmarks

- Add p50/p95/p99 latency benchmarks for tensor inference path.
- Run under controlled concurrency; record queue, infer, and total
  latencies.
- Publish baseline numbers in CI or documentation.

## Requirements

Each requirement is an exit criterion of the original step plan; the
`(n.m)` tag names the substep that owned it.

- [R-1] (9.1) At least two non-FP32 datatypes convert correctly end-to-end through `/v2/models/{model}/infer`.
- [R-2] (9.1) Unsupported datatypes produce `INVALID_ARGUMENT` with a clear message.
- [R-3] (9.2) Readiness endpoints reflect state machine state deterministically.
- [R-4] (9.2) Invalid transition attempts are logged.
- [R-5] (9.3) New cross-cutting infer behavior can be added without editing every route handler.
- [R-6] (9.3) Existing route tests pass unchanged.
- [R-7] (9.4) Two concurrent requests never share a generated ID.
- [R-8] (9.4) Client-supplied IDs are unchanged.
- [R-9] (9.5) Startup probe config is documented and known to work with a model that takes >1s to load.
- [R-10] (9.5) `/v2/health/ready` does not flip true until all configured models are loaded.
- [R-11] (9.6) Baseline latencies are measurable and repeatable.
- [R-12] (9.6) Regression is detectable from benchmark output.

Not every requirement was fully met; the delivered status of each is in
[`validation.md`](validation.md#deviations).

## Out of Scope

Nothing recorded beyond the later steps in the target design.
