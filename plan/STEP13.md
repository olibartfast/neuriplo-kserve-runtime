# Step 13: Control Plane / Data Plane Split

## 13.1 Extract ModelLifecycle — Complete

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

### Exit Criteria Met

- Load/drain/unload/reload logic lives in `ModelLifecycle`, not `ModelRegistry::loadModel()`.
- `ModelRegistry` exposes `reload()` and `completeUnload()` without changing infer route APIs.
- Existing registry and runtime tests remain compatible.

### 13.2 Infer Snapshot — Complete

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

#### Exit Criteria Met

- Infer path reads snapshot via lock-free `atomic_load`; no registry mutex on `submit()`.
- Reload atomically publishes new snapshot; retired scheduler drains in background.
- In-flight requests holding an old snapshot complete without contention.

### 13.3 Background Drain Retire Queue — Complete

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

#### Exit Criteria Met

- Reload/unload return without joining retired scheduler workers.
- In-flight requests on a retired scheduler finish; workers join off the hot path.
- Step 13 (control plane / data plane split) is complete; Step 14 builds on it
  (see `plan/STEP14.md`).
