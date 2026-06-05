# Design Patterns in neuriplo-kserve-runtime

Architecture cheat sheet for the runtime. Prefer symbol names over line numbers when
navigating — line refs drift quickly. Step snapshots live in `plan/STEP0.md` through
`plan/STEP6.md`.

## Strategy / Interface

Abstract base class defines contract; concrete implementations vary behavior.

```
src/Executor.hpp           — class Executor { virtual infer() = 0; }
src/Scheduler.hpp          — class Scheduler { virtual submit() = 0; }
src/NeuriploAdapter.hpp    — class NeuriploAdapter { virtual load/infer() = 0; }
```

Production implementations:
- `StubExecutor`, `NeuriploExecutor` (`Executor`)
- `ModelScheduler` (`Scheduler`, hidden in anonymous namespace in `ModelScheduler.cpp`)
- `RealNeuriploAdapter` (`NeuriploAdapter`, in `RealNeuriploAdapter.cpp`)

Test-only executors (`CountingExecutor`, `SlowExecutor`, `BlockingExecutor`,
`BatchTrackingExecutor`, `ThrowingExecutor`, `RecordingBatchExecutor`) inject fakes
without touching consumers.

`NeuriploExecutor` adapts `NeuriploAdapter` to `Executor`. Switching backend or test
harness never touches `KServeRuntime` or `ModelRegistry`.

---

## Factory Function

Free functions return `std::unique_ptr<Interface>`, hiding concrete type and
construction details.

```
src/Scheduler.hpp          — makeModelScheduler(...)
src/StubExecutor.hpp       — makeStubExecutor(...)
src/NeuriploExecutor.hpp   — makeNeuriploExecutor(...)
src/NeuriploAdapter.hpp    — makeRealNeuriploAdapter()
```

Consumers include headers for the interface and factory only. They never depend on
`ModelScheduler` or `StubExecutor` types directly.

---

## PIMPL / Handle-Body

Interface declared in `.hpp`; implementation hidden in anonymous namespace inside
`.cpp`.

```
src/Scheduler.hpp          — class Scheduler (pure virtual)
src/ModelScheduler.cpp     — class ModelScheduler : public Scheduler (in namespace {})
```

Zero transitive includes leak to consumers. `ModelScheduler` is fully hidden.

---

## Dependency Injection

Constructors accept callbacks or factory functions instead of hardwired dependencies.

```
src/ModelRegistry.hpp      — ExecutorFactory = function<unique_ptr<Executor>(...)>
src/ModelRegistry.cpp      — ModelRegistry(config, factory)
src/HttpServer.hpp         — Handler = function<HttpResponse(const HttpRequest&)>
src/HttpServer.cpp         — HttpServer(host, port, handler, ...)
src/RuntimeConfig.hpp      — RuntimeEnvironment { get, pathExists }
src/RuntimeConfig.cpp      — parseRuntimeConfig(argc, argv, environment)
src/NeuriploExecutor.cpp   — makeNeuriploExecutor(..., adapter) for test injection
```

Tests inject fake executors, adapters, environments, and handlers without modifying
production code.

---

## Facade

Single entry point hides subsystem wiring.

```
src/KServeRuntime.cpp      — route(), handleInfer(), health/metadata handlers
```

`KServeRuntime` orchestrates `ModelRegistry`, `KServeV2Codec`, and scheduler error
mapping behind one HTTP handler surface. `main.cpp` wires config → registry → runtime
→ server.

---

## RAII (Resource Acquisition Is Initialization)

Ownership tied to object lifetime. Destructors guarantee cleanup.

```
src/ModelScheduler.cpp     — ~ModelScheduler() calls beginDrain()
src/ModelScheduler.cpp     — beginDrain() joins workers and rejects queued work
src/HttpServer.hpp         — ~HttpServer(); deleted copy/assign
src/ModelHandle.hpp        — unique_ptr<Scheduler> scheduler
src/ModelRegistry.cpp      — vector<unique_ptr<Executor>> executor pool
```

Lock guards in `ModelScheduler`: `queue_mutex_`, `inflight_mutex_`, `metrics_mutex_`.

`HttpServer` uses `std::lock_guard` for client thread bookkeeping and joins threads
on stop/destruction.

---

## Adapter / Anti-Corruption Layer

Translates between external protocol/library and internal domain types.

```
src/KServeV2Codec.hpp/cpp  — HTTP/JSON ↔ ExecutionRequest/ExecutionResponse
src/GrpcV2Codec.hpp/cpp    — gRPC/protobuf ↔ ExecutionRequest/ExecutionResponse
src/NeuriploExecutor.cpp   — NeuriploAdapter ↔ Executor
src/RealNeuriploAdapter.cpp — real neuriplo library ↔ NeuriploAdapter
src/DynamicBatcher.hpp/cpp — N requests ↔ merged request + split responses
```

KServe protocol types never leak into `Executor` or `Scheduler`. `neuriplo` types never
leak into HTTP/routing or gRPC.

---

## Specification / Policy

Pure validation and scheduling rules with no mutable state.

```
src/BatchCompatibility.hpp/cpp — batchCompatibilityError(), areBatchCompatible()
src/DynamicBatcher.hpp/cpp     — shouldDispatchBatchSize(merged_batch_dimension, ...)
```

`BatchCompatibility` checks whether two requests may share a batch. `shouldDispatchBatchSize`
compares accumulated **merged tensor batch dimension** (sum of input `shape[0]`), not
raw request count, against `max_batch_size` and `preferred_batch_sizes`.

---

## Producer/Consumer

Bounded queue with condition variable. `submit()` produces; worker threads consume.

```
src/ModelScheduler.cpp     — submit() enqueues PendingRequest, notify_one
src/ModelScheduler.cpp     — workerLoop() → popNextPending() → formBatch() → processBatch()
src/ModelScheduler.cpp     — queue_.size() >= max_queue_size → OVERLOADED
```

Synchronization: `queue_mutex_`, `queue_cv_`, `deque<PendingRequest> queue_`.

Batch formation (`formBatch`) scans the queue (not front-only):
- skips incompatible neighbors
- removes cancelled entries
- fulfills expired entries immediately with timeout
- waits until formation deadline or next queued deadline

See `plan/STEP6.md` for dynamic batching behavior.

---

## Promise/Future

Decouples request submission from result delivery. Submitter blocks on future; worker
sets promise via `fulfillResult()`.

```
src/ModelScheduler.cpp     — PendingRequest { promise, future.wait_until(deadline) }
src/ModelScheduler.cpp     — fulfillResult() → promise->set_value(...)
```

Cancellation via atomic flag on `PendingRequest::cancelled`:
- submitter sets `cancelled` after client-side deadline expiry
- workers skip cancelled entries during batch formation and processing

---

## Object Pool + Load-Balancing Policy

Multiple executor instances per model (`--instances`). Shared queue; per-instance
inflight counters.

```
src/ModelRegistry.cpp      — creates N executors, passes vector to makeModelScheduler
src/ModelScheduler.cpp     — selectExecutorIndex() picks least-inflight executor
src/ModelScheduler.cpp     — decrementInflight() after infer completes
```

Not a generic pool — fixed-size vector created at model load. Policy is least-inflight,
not round-robin.

---

## Value Object / Configuration Object

Plain structs carry data; defaults set at declaration.

```
src/Executor.hpp           — InputTensor, OutputTensor, ExecutionRequest, ExecutionResponse
src/RuntimeConfig.hpp      — RuntimeConfig, RuntimeEnvironment
src/Scheduler.hpp          — SchedulerConfig, SchedulerResult
src/SchedulerMetrics.hpp   — SchedulerMetricsSnapshot
src/DynamicBatcher.hpp     — DynamicBatchingConfig, MergedBatch
src/ModelMetadata.hpp      — ModelMetadata, TensorMetadata
```

All fields public. Copy/move via compiler defaults. Behavioral logic lives in free
functions or classes elsewhere (`DynamicBatcher`, `BatchCompatibility`, codecs).

---

## Optional

`std::optional<T>` represents presence/absence without pointers or sentinels.

```
src/RuntimeConfig.hpp      — RuntimeEnvironment::get → optional<string>
src/ModelHandle.hpp        — optional<string> load_error
src/ModelRegistry.hpp      — optional<ModelMetadata> find(...)
src/Executor.hpp           — optional<string> id on ExecutionRequest
src/BatchCompatibility.hpp — optional<string> batchCompatibilityError(...)
src/DynamicBatcher.hpp     — optional<string> mergeExecutionRequests(...) error
```

---

## State

Model lifecycle tracked explicitly.

```
src/ModelState.hpp         — enum class ModelState { Loading, Ready, Failed, ... }
src/ModelHandle.hpp        — ModelHandle { state, load_error, isReady() }
src/ModelRegistry.cpp      — loadModel() transitions state on success/failure
```

Readiness combines model state, scheduler readiness, and drain status.

---

## Error Translation

Scheduler and backend errors map to HTTP status codes.

```
src/Scheduler.hpp          — enum class SchedulerError { None, Overloaded, Timeout,
                             Draining, Unavailable }
src/ModelScheduler.cpp     — makeTimeoutResult(), makeDrainingResult(), makeOverloadedResult()
src/KServeRuntime.cpp      — switch (scheduled.scheduler_error):
                               Overloaded → 429, Timeout → 504,
                               Draining/Unavailable → 503
src/KServeRuntime.cpp      — execution_response.ok == false → 400/500
```

---

## Composite / Batch Processing

Merge N domain objects into one, dispatch, split results back.

```
src/DynamicBatcher.cpp     — mergeExecutionRequests(), splitExecutionResponse()
src/ModelScheduler.cpp     — processBatch() merge → infer → split
```

Single-request fast path bypasses merge/split when `active_batch.size() == 1`
(`processSingle()`).

`splitExecutionResponse()` validates batched output shape/data consistency before
slicing per-request responses.

---

## Separation of Concerns (Module Layout)

| Module | Responsibility | Boundary |
|--------|---------------|----------|
| `KServeRuntime` | HTTP routing, error mapping | RPC protocol |
| `GrpcServer`/`GrpcV2Codec` | gRPC routing, protobuf conversion | RPC protocol |
| `KServeV2Codec` | JSON ↔ tensor types | Serialization |
| `ModelRegistry` | Model lifecycle, executor factory | State machine |
| `ModelHandle` | Model identity + scheduler ref | Aggregate root |
| `Scheduler`/`ModelScheduler` | Admission, queuing, batching, deadlines | Scheduling policy |
| `Executor` | Model inference | Backend boundary |
| `NeuriploAdapter` | Backend library boundary | Integration |
| `DynamicBatcher` | Merge/split logic | Batch transformation |
| `BatchCompatibility` | Compatibility checks | Input validation |
| `HttpServer` | TCP/HTTP transport | I/O boundary |
| `RuntimeConfig` | CLI/env parsing | Configuration |

Each module depends only on interfaces below it. No cyclical dependencies.

Related docs: `plan/STEP5.md` (scheduler), `plan/STEP6.md` (dynamic batching), `AGENTS.md`
(build/test conventions).
