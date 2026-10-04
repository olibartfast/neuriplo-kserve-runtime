# Step 13: Control Plane / Data Plane Split - requirements

> Retrospective packet. Step 13 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 13
> snapshot (`plan/STEP13.md`, later `specs/history/steps/STEP13.md`) and the
> Step 13 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Roadmap phase: [Phase 1 - Platform E2E and Production Track](../roadmap.md#phase-1---platform-e2e-and-production-track-steps-12-14)
Status: **Complete**
Specified: 2026-06-10 (snapshot date)

## Goal

Separate model load/drain/unload/reload (control plane) from registry lookup
and infer routing (data plane), so reload and drain never block in-flight
inference beyond the defined drain semantics. Required before multi-model hot
reload (the "Post-Step 8: Control plane vs data plane split" item of the
target design).

## In Scope

## Requirements

Each requirement is an exit criterion of the original step plan; the
`(n.m)` tag names the substep that owned it.

- [R-1] (13.1) Load/drain/unload/reload logic lives in `ModelLifecycle`, not `ModelRegistry::loadModel()`.
- [R-2] (13.1) `ModelRegistry` exposes `reload()` and `completeUnload()` without changing infer route APIs.
- [R-3] (13.2) The infer path reads an immutable snapshot without taking a registry mutex on `submit()`.
- [R-4] (13.2) Reload atomically publishes a new snapshot; requests holding the old snapshot complete on the old scheduler.
- [R-5] (13.3) Reload and unload return without joining retired scheduler workers; the workers join off the hot path.

## Out of Scope

Nothing recorded beyond the later steps in the target design.
