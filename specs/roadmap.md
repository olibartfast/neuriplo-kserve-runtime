# Neuriplo KServe Runtime Roadmap

> Status: living brownfield roadmap, reconstructed on 2026-10-04 from
> the former target design, the former next-steps file, and `CHANGELOG.md`. Phase order after
> the current phase is a working sequence to be confirmed.

This roadmap is scoped to the KServe runtime. Design patterns are in
`architecture.md`. The step snapshots (`STEP0.md` to `STEP14.md`), the original
target design and the YOLO e2e procedure were removed from the tree; they remain
in git history at [e77f6f0](https://github.com/olibartfast/neuriplo-kserve-runtime/tree/e77f6f0/specs/). Step status that was in the former
next-steps file is merged into the phases below.

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
the target design and `STEP0.md` to `STEP11.md` in [git history](https://github.com/olibartfast/neuriplo-kserve-runtime/tree/e77f6f0/specs/history).

## Phase 1 - Platform E2E and Production Track (Steps 12-14)

**Status: Complete**

YOLO end-to-end through neuriplo-infer, HTTP and gRPC parity, control/data
plane split, and multi-model hot reload with zero-downtime version switch.
Record: the YOLO e2e procedure and `STEP12.md` to `STEP14.md` in
[git history](https://github.com/olibartfast/neuriplo-kserve-runtime/tree/e77f6f0/specs/).

Summary: Step 12 ran real neuriplo-infer to KServe HTTP to runtime to neuriplo
(ONNX Runtime) to YOLO, with gRPC parity (`real-onnx-grpc`, `scripts/e2e-yolo.sh`).
Step 13 split control and data plane (`ModelLifecycle`, `SchedulerRetireQueue`).
Step 14 added the multi-model registry, `/v2/admin/models` load/unload/reload,
and zero-downtime version activation.

## Phase 2 - Multi-Backend and Raw Output Path

**Status: Complete**

Backend registry, plugin loader integration, typed byte-buffer input, and the
raw output hot path (Step 15: `RealNeuriploAdapter::infer()` uses
`get_infer_results_raw()` and maps `RawOutputTensor` directly to `OutputTensor`;
needs neuriplo PR #14 on `develop`, which gates real-* CI). The Step 15.4 item (LLM on raw
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

Items carried from the former next-steps file, in no committed order:

- `--model-repository` Triton-layout scan (`config.pbtxt` to backend id);
  auto-load a model repository without admin POSTs.
- `device_id` and multi-GPU placement (per-backend device selection).
- Per-backend OBJECT-lib isolation in built-in mode.
- LLM raw output once neuriplo exposes it (Step 15.4): keep `llmInfer()` on
  `get_infer_results()` until then.
- CI hardening: pin the neuriplo SHA in runtime real-* jobs (cross-repo API
  drift guard; in PR #7 / follow-up); free-disk-space step before
  ROCm/MIGraphX Docker builds (neuriplo-side).
- Optional byte-identical comparison test of the raw output path versus the old
  `get_infer_results()` path.
- Release housekeeping: finish the GitFlow `release/0.3.0` leftovers if not yet
  done (merge to `master`, tag `v0.3.0`, back-merge to `develop`), and cut the
  matching neuriplo release once its raw-output API gate is merged and CI is
  stable. Superseded in practice by Phase 3 (v0.4.0) if already released.
- Out of scope unless requested: Step 13.2 data-plane work on branch
  `feature/step-13-2-infer-data-plane` (separate from Step 15).

Each needs a packet before work starts.

## Phase 5 - Optimization Track

**Status: Planned, scale-triggered**

Async HTTP reactor, zero-copy tensor buffers, arena/PMR allocators,
`std::expected`-style result types at executor/scheduler boundaries, and
multi-datatype gRPC codec. Each starts only on its trigger: connection
concurrency above O(100) for the reactor (epoll/io_uring), copy overhead visible
in profiling for zero-copy buffers, allocation churn under load for arena/PMR,
error-handling bugs for structured result types,
and client demand for INT8/FP16 in the gRPC codec.

## Infrastructure and Cross-Repo Notes

CI is done: parallel sanitizer matrix (`fail-fast: false`), `real-onnx-grpc`
preset job, `scripts/e2e-stub.sh` smoke job (11 checks incl. admin lifecycle),
triggers on `master` and `develop`, `enable_testing()` fix, `.gitignore` entries.
Cross-repo: neuriplo-infer `feature/neuriplo-kserve-runtime` wires the KServe
HTTP/gRPC client through `InferenceInterface` (`--kserve_endpoint`,
`--kserve_model_name`, `--kserve_transport=http|grpc`; no local `--weights`
needed remotely). A real HTTP E2E with `yolo26s.onnx` on `--backend onnx_runtime`
succeeded, and neuriplo-infer's 22/22 tests passed.

## Assumptions to Confirm

- [A-10] The Step 15 status in `AGENTS.md` ("on `feature/step-15-raw-output`")
  is stale, because the raw output path is already used by the adapter.
- [A-11] Phase 4 and 5 ordering is a working sequence, not a commitment.

## Standing risks

- Backend thread-safety is per-backend and not uniformly documented upstream.
- Tensor dtype mapping must stay explicit and tested; silent widening is the
  failure mode.
- Dynamic batching can silently change output splitting if compatibility rules
  are relaxed carelessly.
- LLM scheduling is not tensor batching. Token, context, cancellation, and
  memory-pressure policies have to stay explicit.
- Pulling server dependencies into `neuriplo` core, or the task layer into the
  default runtime build, would create dependency creep. The
  `NEURIPLO_RUNTIME_ENABLE_TASKS=OFF` default is the guard.

_Revision: 2026-10-04 - initial brownfield roadmap._
