# Neuriplo KServe Runtime Roadmap

> Status: living brownfield roadmap, reconstructed on 2026-10-04 from
> the former target design, the former next-steps file, and `CHANGELOG.md`. Phase order after
> the current phase is a working sequence to be confirmed.

This roadmap is scoped to the KServe runtime. Design patterns are in
`architecture.md`. Steps 0-14 are recorded as retrospective packets under
`specs/` (see [README.md](README.md)), ported from the step snapshots, the
original target design and the YOLO e2e procedure. Step status that was in the
former next-steps file is merged into the phases below.

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
items. The Step 0-14 retrospective packets are the only backfill; completed
work after them is recorded in its own packet or in `CHANGELOG.md`.

## Phase 0 - Serving Foundation (Steps 0-11)

**Status: Complete**

Scaffold, KServe packaging, V2 protocol, model registry and executors, atomic
neuriplo integration, scheduler, dynamic batching, observability, LLM backends
and path completion, production hardening, and deployment validation. Record:
[2026-05-23-runtime-target-design](2026-05-23-runtime-target-design/plan.md)
and the Step 0-11 packets it indexes. Their `validation.md` files list the exit
criteria that were only partly met (Steps 1, 6-11), notably the LLM decode path
(Step 10) and cluster-level deployment checks (Step 11).

## Phase 1 - Platform E2E and Production Track (Steps 12-14)

**Status: Complete**

YOLO end-to-end through neuriplo-infer, HTTP and gRPC parity, control/data
plane split, and multi-model hot reload with zero-downtime version switch.
Record: [2026-06-05-yolo-platform-e2e](2026-06-05-yolo-platform-e2e/validation.md)
(including the YOLO e2e procedure),
[2026-06-10-control-data-plane-split](2026-06-10-control-data-plane-split/plan.md),
[2026-06-11-multi-model-hot-reload](2026-06-11-multi-model-hot-reload/plan.md).

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

**Status: Complete** - released as v0.4.0 on 2026-10-05, as part of the
cross-repo packet
[2026-10-03-kserve-dynamic-dim-encoded-image](https://github.com/olibartfast/neuriplo-platform/tree/main/specs/2026-10-03-kserve-dynamic-dim-encoded-image).

Shipped everything on `develop` since v0.3.2: pipeline (ensemble) models,
`NEURIPLO_RUNTIME_ENABLE_TASKS`, the `dali` backend id, `--use-gpu`, the
dynamic-dimension and encoded-image fixes, the executor error-propagation fix,
and model repository serving with the KServe repository extension and the
init-container tooling (#14-#16). The pre-release audit (4 blockers, 16 major,
31 minor) and the release review (3 integration defects) were fixed in #19 and #21-#24; see the packet's validation
record and [CHANGELOG.md](../CHANGELOG.md).

## Phase 4 - Deferred Follow-Ups

**Status: Planned**

In no committed order.

Deferred from the v0.4.0 pre-release audit:

- Pipeline: propagate request cancellation into steps (A-8); map step failures
  to precise status codes (A-9); check edge datatypes between steps beyond
  FRAME_SIZE (A-10); reject unknown pipeline config keys (A-12).
- Ensembles stay ready when a step model is unloaded, and cached step metadata
  goes stale (B-18).
- Ensemble contract questions (`platform` for model-first graphs,
  `max_batch_size`): raise against the platform ensemble contract (A-13).
- Run images and pods as non-root with `fsGroup` on the PVC, and build
  `Dockerfile.tensorrt` from a pinned source; needs a k3d/GPU validation run
  (C-7, C-9).
- Authentication for admin and repository routes, and an allowlist for
  `model_path` / `plugin_dir`.
- Metadata validation in the gRPC codec.
- Remove the `data: []` validation bypass (test fixtures depend on it).
- A pre-decode size cap for TGA (no signature to detect it by).
- `scheduler_skips_incompatible_queue_neighbors_during_batch_formation` is
  timing-sensitive and failed once under valgrind in CI (PR #25, no leak);
  make it deterministic.

Carried from the former next-steps file:

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
