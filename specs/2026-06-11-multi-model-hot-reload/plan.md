# Step 14: Multi-Model Hot Reload - plan

> Retrospective packet. Step 14 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 14
> snapshot (`plan/STEP14.md`, later `specs/history/steps/STEP14.md`) and the
> Step 14 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Tasks are the step's substeps. Each `T-` section is the implementation
record from the snapshot.

## [T-1] 14.1 Multi-Model Registry — Complete

**Goal:** Registry holds a map of model name → slot so multiple models load, serve,
and unload independently.

### Changes

**`src/ModelRegistry.hpp/.cpp`**:

- `ModelSlot` — per-model `ModelHandle`, active `InferSnapshot`, and retained
  per-version snapshots
- `models_` map guarded by `std::shared_mutex`; infer lookups take the shared lock
  only to resolve the slot, then read the snapshot via `atomic_load`
- `loadModel()` / `unloadModel()` add and remove slots at runtime; `listModels()`
  enumerates loaded models

**`tests/MultiModelRegistryTest.cpp`** — load/list two models, unload one without
affecting the other, reload retires the old scheduler in the background.

## [T-2] 14.2 Admin Endpoints — Complete

**Goal:** Model add/remove/reload at runtime via HTTP admin surface.

### Routes (`src/KServeRuntime.cpp::handleAdmin`)

| Method | Path | Action |
|--------|------|--------|
| GET | `/v2/admin/models` | List models with default version + readiness |
| POST | `/v2/admin/models/load` | Load a model from a JSON config body |
| DELETE | `/v2/admin/models/{name}` | Drain + unload a model |
| POST | `/v2/admin/models/{name}/reload` | Reload with new config, background-retire old scheduler |
| POST | `/v2/admin/models/{name}/versions/{version}/activate` | Switch active version |

**`src/AdminCodec.hpp/.cpp`** — JSON body parsing (`parseLoadModelRequest`,
`parseReloadModelRequest`, `parseSwitchVersionRequest`) layered over `RuntimeConfig`
defaults.

**`tests/AdminEndpointTest.cpp`** — route coverage including error paths
(400 invalid body, 404 unknown model/route, 409 failed load/reload).

## [T-3] 14.3 Version Switch With Zero-Downtime Drain — Complete

**Goal:** Activate a new model version without dropping in-flight requests on the
old version.

### Changes

- `ModelRegistry::switchVersion()` reloads the slot pinned to the requested version
  (`ModelLifecycle::reload` `version_override` parameter), publishes the new
  snapshot atomically, retains the old version's snapshot in
  `ModelSlot::version_snapshots` for versioned routes, and hands the old scheduler
  to the retire queue for background drain.
- `ModelLifecycle::load/reload` take an optional `version_override`: explicit
  switches pin the published version; plain loads keep executor-reported metadata
  versions (config default `"1"` no longer clobbers them).

**Tests** — `multi_model_registry_switch_version_keeps_old_version_snapshot`,
`model_registry_resolves_version_from_executor_metadata`, plus versioned-route
coverage in `tests/NeuriploPlatformE2eTest.cpp`.

## Related Fixes Landed With This Step

- `enable_testing()` was missing from `CMakeLists.txt`, so `ctest --preset <p>`
  discovered zero tests and CI's test steps were silently green. Tests now run
  through CTest again.
- `tests/NeuriploPlatformE2eTest.cpp` (in-process platform e2e suite: HTTP server,
  state machine, scheduler, batching, codec, pipeline) is wired into the test
  binary and passing.
