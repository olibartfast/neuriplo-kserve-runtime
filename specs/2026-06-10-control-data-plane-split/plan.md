# Step 13: Control Plane / Data Plane Split - plan

> Retrospective packet. Step 13 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 13
> snapshot (`plan/STEP13.md`, later `specs/history/steps/STEP13.md`) and the
> Step 13 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Tasks are the step's substeps. Each `T-` section is the implementation
record from the snapshot.

## [T-1] 13.1 Extract ModelLifecycle — Complete

**Goal:** Separate model load/drain/unload/reload (control plane) from registry lookup and infer routing (data plane).

### Changes

**`src/ModelLifecycle.hpp/.cpp`** — new control-plane type:

- `load()` — executor creation, scheduler wiring, state transitions (`Unloaded → Loading → Ready/Failed`)
- `beginDrain()` — `Ready → Unloading`, scheduler drain
- `completeUnload()` — tear down scheduler, clear metadata, `Unloading → Unloaded`
- `reload()` — drain + unload when ready, reset failed/unavailable, then load

**`src/ModelRegistry.hpp/.cpp`** — data-plane registry:

- Owns `ModelHandle` + `ModelLifecycle`
- Constructor delegates to `lifecycle_.load()`
- `beginDrain()` / new `completeUnload()` / `reload()` delegate to lifecycle
- Lookup, readiness, and metrics methods unchanged

**`tests/ModelLifecycleTest.cpp`** — lifecycle coverage:

- Load success and failure paths
- Drain and complete-unload transitions
- Reload replaces executor output
- Registry delegation for `reload()` and `completeUnload()`

### [T-2] 13.2 Infer Snapshot — Complete

**Goal:** Infer routes use a stable read-only scheduler view; lifecycle swaps in new
schedulers without tearing down in-flight work.

#### Changes

**`src/InferSnapshot.hpp`** — immutable data-plane view:

- `shared_ptr<Scheduler>` kept alive by snapshot holders
- `fromHandle()` builds snapshot from control-plane `ModelHandle`

**`src/ModelRegistry.hpp/.cpp`** — atomic snapshot publish:

- `active_snapshot_` swapped via `std::atomic_store` / `std::atomic_load`
- `findHandle()` / `findHandleVersion()` return `shared_ptr<const InferSnapshot>`
- `publishSnapshot()` called after load, reload, drain, and unload

**`src/ModelHandle.hpp`** — scheduler ownership via `shared_ptr` (snapshot shares refs)

**`src/ModelLifecycle.cpp`** — reload swap semantics:

- `stopAccepting()` on retired scheduler (no worker join)
- build replacement handle, swap scheduler in place
- failed reload restores valid state transitions (`Unloaded → Loading → Ready`)

**`src/Scheduler.hpp`**, **`ModelScheduler`**, **`LlmScheduler`**:

- `stopAccepting()` — reject new submits; in-flight work continues
- `beginDrain()` — idempotent full drain (always joins workers)

**`src/KServeRuntime.cpp`**, **`src/GrpcServer.cpp`**, **`src/RequestPipeline.hpp`**:

- infer hot path uses `InferSnapshot` instead of `const ModelHandle *`

**`tests/ModelLifecycleTest.cpp`**:

- `infer_snapshot_keeps_old_scheduler_alive_during_reload` — concurrent infer +
  reload: in-flight completes on old marker, new requests use new marker

### [T-3] 13.3 Background Drain Retire Queue — Complete

**Goal:** Drain + reload without blocking in-flight inference; lifecycle never joins
retired scheduler workers on the control path.

#### Changes

**`src/SchedulerRetireQueue.hpp/.cpp`** — drained-scheduler graveyard:

- `retire(shared_ptr<Scheduler>)` — runs `beginDrain()` (full drain + worker join)
  on a background `std::async` task; control plane returns immediately
- `pendingCount()` — prunes completed drains, reports in-flight retirements
- destructor waits for all pending drains (clean shutdown)

**`src/ModelRegistry.hpp/.cpp`** — retirement wiring:

- owns `SchedulerRetireQueue`; `reload()`, `switchVersion()`, and `completeUnload()`
  hand the displaced scheduler to the queue instead of joining inline
- `retiredSchedulerCount()` exposes queue depth for tests/observability

**`tests/SchedulerRetireQueueTest.cpp`** — retire queue drains in background and
completes on destruction.
