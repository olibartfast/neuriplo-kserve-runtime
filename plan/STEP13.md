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

### Next (13.2)

- Decouple infer hot path from lifecycle mutex/serialization so load does not contend with `submit()`.
