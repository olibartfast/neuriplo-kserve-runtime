# Step 5: Scheduler And Autoscaling Behavior - requirements

> Retrospective packet. Step 5 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 5
> snapshot (`plan/STEP5.md`, later `specs/history/steps/STEP5.md`) and the
> Step 5 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Roadmap phase: [Phase 0 - Serving Foundation](../roadmap.md#phase-0---serving-foundation-steps-0-11)
Status: **Complete**
Specified: 2026-05-29 (snapshot date)

## Goal

This snapshot records the Step 5 implementation state. The runtime now routes
inference through a per-model scheduler with bounded admission, worker
execution, deadline handling, overload responses, drain-aware readiness, and
internal autoscaling metrics.

## In Scope

- Add bounded request queue.
- Add worker thread.
- Add timeouts.
- Add overload behavior.
- Add least-inflight instance selection.
- Add graceful shutdown/draining behavior.
- Add concurrency, queue depth, and latency metrics needed for HPA/KEDA/custom
  autoscaling.

## Requirements

Each requirement is an exit criterion of the original step plan; the
`(n.m)` tag names the substep that owned it.

- [R-1] Queue depth is bounded.
- [R-2] Overload produces `429` or `503`.
- [R-3] Multiple concurrent requests complete without data races.
- [R-4] Readiness flips false while draining.

## Out of Scope

Recorded at the end of the step as remaining gaps; later steps picked them
up as noted in [`../roadmap.md`](../roadmap.md).

### Remaining Work

- Dynamic batching remains Step 6 work.
- Prometheus `/metrics` export remains Step 7 work.
