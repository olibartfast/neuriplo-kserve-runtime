# Step 14: Multi-Model Hot Reload - requirements

> Retrospective packet. Step 14 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 14
> snapshot (`plan/STEP14.md`, later `specs/history/steps/STEP14.md`) and the
> Step 14 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Roadmap phase: [Phase 1 - Platform E2E and Production Track](../roadmap.md#phase-1---platform-e2e-and-production-track-steps-12-14)
Status: **Complete**
Specified: 2026-06-11 (snapshot date)

## Goal

Let one runtime hold several models that load, serve, reload and unload
independently at runtime, and switch a model's active version without
dropping in-flight requests on the old version. Builds on the Step 13 split.

## In Scope

## Requirements

Each requirement is an exit criterion of the original step plan; the
`(n.m)` tag names the substep that owned it.

- [R-1] (14.1) The registry maps model name to slot; models load, serve and unload independently.
- [R-2] (14.2) Models can be listed, loaded, unloaded, reloaded and version-switched over `/v2/admin/models` routes.
- [R-3] (14.2) Admin errors are stable: 400 invalid body, 404 unknown model or route, 409 failed load or reload.
- [R-4] (14.3) Activating a version publishes the new snapshot atomically, keeps the old version's snapshot for versioned routes, and drains the old scheduler in the background.
- [R-5] (14.3) Plain loads keep executor-reported metadata versions.

## Out of Scope

Nothing recorded beyond the later steps in the target design.
