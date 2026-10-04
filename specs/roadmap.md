# Neuriplo KServe Runtime Roadmap

> Status: living brownfield roadmap, reconstructed on 2026-10-04 from
> `plan/ROADMAP.md`, `plan/NEXT_STEPS.md`, and `CHANGELOG.md`. Phase order after
> the current phase is a working sequence to be confirmed.

This roadmap is scoped to the KServe runtime. `plan/` stays in place as the
historical implementation record: step snapshots (`plan/STEP0.md` to
`plan/STEP14.md`) are not copied here and are not retired.

## Status Key

- **Complete** - the intended repository capability has landed; ongoing
  maintenance may remain.
- **Current** - the phase being worked now.
- **Next** - the next phase to specify and implement.
- **Specified** - a dated feature packet exists and validation is defined;
  implementation has not started.
- **Planned** - ordered but not yet specified as a feature packet.
- **Blocked** - cannot proceed without an identified decision or dependency.

## Specification Rule

Create a dated `specs/YYYY-MM-DD-feature-name/` packet for active work that is
multi-phase, changes public behavior or architecture, or has low reversibility.
The packet contains `requirements.md`, `plan.md`, and `validation.md`, with
validation defined before implementation and evidence recorded after execution.
Work that spans repositories is specified in
[neuriplo-platform](https://github.com/olibartfast/neuriplo-platform) `specs/`
and linked from here. Small contained fixes may use a concise PR-level note;
trivial fixes need no packet. Do not create speculative packets for inactive
items or backfill packets for completed work.

## Phase 0 - Serving Foundation (Steps 0-11)

**Status: Complete**

Scaffold, KServe packaging, V2 protocol, model registry and executors, atomic
neuriplo integration, scheduler, dynamic batching, observability, LLM backends
and path completion, production hardening, and deployment validation. Record:
[plan/ROADMAP.md](../plan/ROADMAP.md), `plan/STEP0.md` to `plan/STEP11.md`.

## Phase 1 - Platform E2E and Production Track (Steps 12-14)

**Status: Complete**

YOLO end-to-end through neuriplo-infer, HTTP and gRPC parity, control/data
plane split, and multi-model hot reload with zero-downtime version switch.
Record: [plan/NEXT_STEPS.md](../plan/NEXT_STEPS.md), [plan/E2E_YOLO.md](../plan/E2E_YOLO.md),
`plan/STEP12.md` to `plan/STEP14.md`.

## Phase 2 - Multi-Backend and Raw Output Path

**Status: Complete**

Backend registry, plugin loader integration, typed byte-buffer input, and the
raw output hot path (Step 15 adapter work). The Step 15.4 item (LLM on raw
output) stays deferred until neuriplo exposes it. Released as v0.1.0 to v0.3.2
(see [CHANGELOG.md](../CHANGELOG.md)).

## Phase 3 - v0.4.0 Release

**Status: Current** - part of the cross-repo packet
[2026-10-03-kserve-dynamic-dim-encoded-image](https://github.com/olibartfast/neuriplo-platform/tree/main/specs/2026-10-03-kserve-dynamic-dim-encoded-image),
which is in its Phase 2 (pre-release audit of `v0.3.2..develop`).

Goal: ship the unreleased work on `develop` as v0.4.0 after a full audit.

Scope: everything on `develop` since v0.3.2. That is pipeline (ensemble)
models, `NEURIPLO_RUNTIME_ENABLE_TASKS`, the `dali` backend id, `--use-gpu`,
the dynamic-dimension and encoded-image fixes, the executor error-propagation
fix, and model repository serving: `--models` / `--model-repository` /
`MODEL_REPOSITORY`, `--model-control-mode`, the KServe repository extension
(`/v2/repository/*`), the init-container preparation procedure and its deploy
tooling (#14-#16). The pre-release audit found that `[Unreleased]` in
[CHANGELOG.md](../CHANGELOG.md) did not list the repository work, and that the
`versions.env` pins (neuriplo v0.8.0, neuriplo-tasks v0.8.0) are stale: tasks
v0.8.0 lacks `decodeImage`, so a task-enabled build fails. Both are fixed in the
release batch.

Exit criteria: the audit findings are resolved, `scripts/release-patch.sh` or
the GitFlow release branch merges to `master`, `v0.4.0` is tagged, and
`develop` is back-merged. The `versions.env` pins are confirmed against the
platform version matrix.

## Phase 4 - Deferred Follow-Ups

**Status: Planned**

Items from `plan/NEXT_STEPS.md`, in no committed order:

- `--model-repository` Triton-layout scan (`config.pbtxt` to backend id).
- `device_id` and multi-GPU placement.
- Per-backend OBJECT-lib isolation in built-in mode.
- LLM raw output once neuriplo exposes it (Step 15.4).

Each needs a packet before work starts.

## Phase 5 - Optimization Track

**Status: Planned, scale-triggered**

Async HTTP reactor, zero-copy tensor buffers, arena/PMR allocators, and
multi-datatype gRPC codec. Each starts only on its trigger in
`plan/NEXT_STEPS.md` (concurrency, profiled copy cost, allocation churn, client
demand).

## Assumptions to Confirm

- [A-10] The Step 15 status in `AGENTS.md` ("on `feature/step-15-raw-output`")
  is stale, because the raw output path is already used by the adapter.
- [A-11] Phase 4 and 5 ordering is a working sequence, not a commitment.

_Revision: 2026-10-04 - initial brownfield roadmap._
