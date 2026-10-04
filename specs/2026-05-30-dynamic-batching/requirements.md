# Step 6: Dynamic Batching - requirements

> Retrospective packet. Step 6 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 6
> snapshot (`plan/STEP6.md`, later `specs/history/steps/STEP6.md`) and the
> Step 6 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Roadmap phase: [Phase 0 - Serving Foundation](../roadmap.md#phase-0---serving-foundation-steps-0-11)
Status: **Complete**
Specified: 2026-05-30 (snapshot date)

## Goal

This snapshot records the Step 6 implementation state. Step 5 established
per-model scheduling with bounded admission, worker execution, deadlines,
overload behavior, drain-aware readiness, and internal autoscaling metrics;
Step 6 adds compatible-request grouping, bounded batch formation, batched
execution, and output splitting for tensor models.

## In Scope

- Add compatible-request grouping.
- Add max batch size and max queue delay.
- Split batched outputs.
- Add tests comparing batched vs single outputs.

## Requirements

Each requirement is an exit criterion of the original step plan; the
`(n.m)` tag names the substep that owned it.

- [R-1] Batched outputs are shape/schema equivalent to single-request outputs.
- [R-2] Batch size metrics are exposed.

Not every requirement was fully met; the delivered status of each is in
[`validation.md`](validation.md#deviations).

## Out of Scope

Recorded at the end of the step as remaining gaps; later steps picked them
up as noted in [`../roadmap.md`](../roadmap.md).

### Remaining Work

- Prometheus `/metrics` export remains Step 7 work.
- LLM/token-aware scheduling remains Step 8 work.
