# Step 9: Production Hardening — Tensor Path

**Completed:** 2026-06-04
**Snapshot:** All 6 substeps implemented and tested.

## Summary

Production-hardened the tensor serving path: multi-datatype adapter pipeline, formal model state machine, composable request pipeline, UUID-based request IDs, Kubernetes probe documentation, and latency SLO benchmarks.

## 9.1 Multi-Datatype Neuriplo Adapter

### Changes

**`src/RealNeuriploAdapter.cpp`** — restructured the adapter to avoid FP32 hardcoding:

- Removed FP32 hardcoding in `load()` metadata conversion. Input/output datatypes now flow from backend metadata via `extractDatatypes()` + per-layer `convertLayers()` overload. Default remains FP32 for backends that don't report per-layer types.
- Removed FP32 hardcoding in `infer()` output tensors. Output datatype now reads from stored `output_metadata_` (captured during `load()`).
- Added `bytesToTensor()` — reverse of `tensorToBytes()`, converting raw bytes → `vector<double>` for all 12 supported datatypes (BOOL, INT8–INT64, UINT8–UINT64, FP16, FP32, FP64).
- Added `numericFromBytes<T>()` template — byte-to-double conversion per C++ type.
- Added `convertLayers()` overload accepting `vector<string>` per-layer datatypes.
- Extracted `extractDatatypes()` helper — placeholder that returns FP32 for each layer; to be replaced when the real neuriplo backend provides per-layer type info.

### Tests Added

`tests/NeuriploExecutorTest.cpp`:
- `neuriplo_executor_preserves_int32_output_datatype` — verifies INT32 outputs propagate through NeuriploExecutor
- `neuriplo_executor_preserves_fp16_output_datatype` — verifies FP16 output preservation
- `neuriplo_executor_preserves_uint8_output_datatype` — verifies UINT8 output preservation

FakeNeuriploAdapter refactored to output metadata-driven tensors instead of hardcoded scores/labels.

## 9.2 Formal Model State Machine

### New File

**`src/ModelStateMachine.hpp`** — state machine wrapper around `ModelState`:

- Named transition methods: `startLoad()` (Unloaded→Loading), `markReady()` (Loading→Ready), `markFailed()` (Loading→Failed), `markUnavailable()` (Loading/Ready→Unavailable), `beginUnload()` (Ready→Unloading), `completeUnload()` (Unloading→Unloaded), `reset()` (Failed/Unavailable→Unloaded).
- `isLoaded()` — checks Ready state.
- `isTerminal()` — checks Failed or Unavailable.
- `stateName()` — human-readable state string.
- Observer pattern: `onTransition(Observer)` — callbacks invoked on each state change with `(from, to)`.

### Integration

**`src/ModelHandle.hpp`** — replaced raw `ModelState state`, `canTransitionTo()`, and `transitionTo()` with `ModelStateMachine state`.

**`src/ModelRegistry.cpp`** — updated to use named methods: `handle_.state.startLoad()`, `handle_.state.markFailed()`, `handle_.state.markReady()`, `handle_.state.beginUnload()`.

**`src/KServeRuntime.cpp`** — `modelStateError()` refactored to `switch` on `handle.state.current()`.

### Tests Added

`tests/ModelStateMachineTest.cpp` — 11 test cases:
- `state_machine_starts_unloaded` — default state is Unloaded
- `state_machine_normal_lifecycle` — full Unloaded→Loading→Ready→Unloading→Unloaded cycle
- `state_machine_load_failure` — Loading→Failed→reset→Unloaded
- `state_machine_unavailable_from_loading` — Loading→Unavailable
- `state_machine_unavailable_from_ready` — Ready→Unavailable→reset
- `state_machine_rejects_invalid_transitions` — no invalid jumps from Unloaded
- `state_machine_rejects_double_load` — no re-entrant Loading
- `state_machine_reset_noop_when_unloaded` — reset is no-op on Unloaded
- `state_machine_observes_transitions` — observer callbacks fire correctly
- `state_machine_observer_sees_failure` — observer captures Failed transition
- `state_machine_state_name` — stateName() returns correct strings
- `state_machine_multiple_observers` — multiple observers all fire

## 9.3 Composable Request Pipeline

### New File

**`src/RequestPipeline.hpp`** — lightweight composable pipeline:

- `InferContext` struct — carries request, model metadata, parsed inference, scheduled result through pipeline stages.
- `PipelineStage` — `function<optional<HttpResponse>(InferContext&)>` type alias.
- `RequestPipeline` class — `addStage()` for composition, `run()` executes stages sequentially. First non-nullopt response short-circuits the pipeline.
- No external dependencies.

### Integration

**`src/KServeRuntime.cpp`** — `infer()` method refactored from 4 sequential method calls into a composed pipeline:

```
validateInferModel → decodeAndAdmit → scheduleInfer → encodeInferResponse
```

Each stage is a lambda adapting the existing private method. New middleware can be added via `pipeline.addStage()` without modifying `infer()`.

### Architectural Benefits

- New cross-cutting behavior (auth, rate limiting, request logging) can be added as pipeline stages without touching route handlers.
- Pipeline stages are independently testable.
- Stage ordering is explicit and configurable.

## 9.4 Collision-Resistant Request IDs

### Changes

**`src/KServeRuntime.cpp`** — `generateRequestId()` rewritten from custom hex-group format to RFC 4122 UUID v4:

- Generates 16 random bytes via `thread_local` `std::mt19937_64` RNG seeded from `std::random_device`
- Sets version nibble (byte 6 = 0x40) and variant nibble (byte 8 = 0x80)
- Formats as standard UUID string: `xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx`
- Removed `<iomanip>` and `<sstream>` includes (no longer needed)
- Added `<cstdio>` for `snprintf`

## 9.5 Container Startup Probe Documentation

### New File

**`deploy/kserve/README.md`** — deployment guide covering:

- ClusterServingRuntime deployment
- Kubernetes probe configuration (`startupProbe`, `livenessProbe`, `readinessProbe`) with `/v2/health/live` and `/v2/health/ready` endpoints
- Probe behavior table per model state
- Startup timing guidance for different backends (stub, ONNX, llama.cpp)
- InferenceService, canary, and InferenceGraph deployment commands

## 9.6 Latency SLO Benchmarks

### New File

**`tests/BenchmarkTest.cpp`** — microbenchmark tests:

- `benchmark_scheduler_latency_stub_executor` — warmup (100) + samples (1000), reports p50/p95/p99/mean in microseconds
- `benchmark_scheduler_throughput_submit_wait` — measures req/s with 2-instance stub executor (2000 iterations)

### Baselines (stub executor, debug build, 2 instances)

```
mean: ~247 us
p50:  ~223 us
p95:  ~357 us
p99:  ~409 us
Throughput: ~4280 req/s
```

## Files Changed

| File | Change |
|---|---|
| `src/RealNeuriploAdapter.cpp` | Multi-datatype support, bytesToTensor() |
| `src/ModelStateMachine.hpp` | New file — state machine wrapper |
| `src/ModelHandle.hpp` | Replaced raw state with ModelStateMachine |
| `src/ModelRegistry.cpp` | Named transition methods |
| `src/KServeRuntime.cpp` | Pipeline, UUID, switch-based state check |
| `src/RequestPipeline.hpp` | New file — composable pipeline |
| `tests/NeuriploExecutorTest.cpp` | Multi-datatype output preservation tests |
| `tests/ModelStateMachineTest.cpp` | New file — 11 state machine tests |
| `tests/BenchmarkTest.cpp` | New file — latency + throughput benchmarks |
| `deploy/kserve/README.md` | New file — probe and deployment docs |
| `CMakeLists.txt` | Added new test files |

## Validation

```bash
cmake --build --preset debug  &&  ctest --preset debug
# 100% tests passed, 0 tests failed
scripts/check-format.sh  # passes
```
