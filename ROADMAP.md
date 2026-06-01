# neuriplo-kserve-runtime Roadmap

## Goal

Build a production-oriented, KServe-compatible C++ inference runtime that serves
models through `neuriplo` backends while keeping application behavior and task
semantics outside the server unless explicitly added later.

The runtime should be usable in two deployment modes:

```text
same host:
  vision-inference -> http://127.0.0.1:8080 -> neuriplo-kserve-runtime -> neuriplo

cluster:
  vision-inference/client -> KServe endpoint -> neuriplo-kserve-runtime pod -> neuriplo
```

The first implementation target is raw tensor inference over KServe V2. Task
preprocessing and postprocessing stay in `vision-core` / `vision-inference`
unless a later task-aware serving layer is explicitly designed. LLM serving is a
separate first-class track because `neuriplo` already has llama.cpp and Cactus
backends, but it should use token-aware scheduling instead of the tensor-model
dynamic batching path.

## Repository Boundary

### This Repository Owns

- KServe/Open Inference Protocol V2 HTTP surface.
- Optional gRPC V2 surface later.
- Runtime process lifecycle.
- Model registry and model state.
- Request admission and backpressure.
- Per-model scheduling.
- Execution instance pools.
- Dynamic batching for compatible tensor models.
- LLM/token scheduler for llama.cpp, Cactus, and GGUF-style local models.
- Metrics, health, readiness, and structured errors.
- Container image and KServe `ServingRuntime` manifests.
- KServe deployment examples for `InferenceService`, canary rollout, and
  InferenceGraph-friendly request/response behavior.

### `neuriplo` Owns

- Backend abstraction.
- Backend adapters.
- Runtime dependency compatibility.
- Model loading/execution for local backends.
- Backend-specific metadata extraction.
- Thread-safety guarantees or documented limitations.

### `vision-core` Owns

- Task contracts.
- Model type strings.
- Tensor shape/dtype expectations.
- Preprocessing and postprocessing semantics.
- Result variants and output schema.

### `vision-inference` Owns

- CLI/application flow.
- Source/video input.
- User configuration.
- Calling a remote KServe endpoint.
- Visualization and local output.

## Non-Goals For The First Milestone

- No task-aware output schema in the server.
- No OpenAI-compatible API in the first CV milestone. OpenAI-compatible LLM
  endpoints are allowed only in the explicit LLM milestone for llama.cpp and
  Cactus backends.
- No multi-model hot reload.
- No distributed model cache.
- No ModelMesh replacement.
- No auth/TLS beyond being reverse-proxy friendly.
- No changes to tensor shapes, dtypes, task decoding, NMS, or output semantics.

## Compatibility Target

Implement the KServe V2 protocol surface required by a custom serving runtime:

```text
GET  /v2
GET  /v2/health/live
GET  /v2/health/ready
GET  /v2/models/{model_name}
GET  /v2/models/{model_name}/ready
POST /v2/models/{model_name}/infer
```

Initial response bodies should match the Open Inference Protocol shape closely
enough that KServe clients and simple HTTP tests can validate readiness,
metadata, and inference behavior.

The runtime should also be deployable through KServe's model spec using a
conservative custom model format:

```yaml
predictor:
  model:
    modelFormat:
      name: neuriplo
    protocolVersion: v2
    runtime: neuriplo-kserve-runtime
    storageUri: pvc://...
```

Do not advertise automatic selection for generic formats such as `onnx`,
`openvino`, `tensorrt`, or `gguf` until metadata conversion and backend behavior
are validated per format.

## Step Index

The roadmap is organized as implementation steps. Each completed step should
have a matching `STEP<N>.md` snapshot that records what was actually built and
validated. Work-in-progress step planning should use `STEP<N>_WIP.md`.

```text
Step 0: Scaffold
Step 1: KServe Packaging
Step 2: Protocol Correctness
Step 3: Model Registry And Executor Abstraction
Step 4: Atomic neuriplo Integration
Step 5: Scheduler And Autoscaling Behavior
Step 6: Dynamic Batching
Step 7: Observability And KServe Deployment Examples
Step 8: LLM Backends (llama.cpp and Cactus)
```

Current architecture patterns are documented in `DESIGN_PATTERNS.md`. That file
records what exists today; this section records what to grow into and when.

## Architecture And Design Pattern Evolution

The runtime already uses a sound baseline: Strategy (`Executor`, `Scheduler`,
`NeuriploAdapter`), factory functions, PIMPL, dependency injection, bounded
producer/consumer scheduling, adapter/anti-corruption layers, and composite
batch merge/split. These are appropriate for a C++17 KServe runtime and do not
need replacement.

The gap versus production-oriented serving stacks is mostly operational and
systems patterns, not missing GoF abstractions. The roadmap below adds modern
serving mechanics incrementally without over-engineering the current scaffold.

### Current Patterns (Keep)

| Pattern | Location | Status |
|---------|----------|--------|
| Strategy / interface boundaries | `Executor`, `Scheduler`, `NeuriploAdapter` | In use |
| Factory + PIMPL | `makeModelScheduler`, `makeStubExecutor`, `ModelScheduler.cpp` | In use |
| Dependency injection | `ModelRegistry::ExecutorFactory`, `HttpServer::Handler` | In use |
| Facade | `KServeRuntime` | In use |
| Producer/consumer + promise/future | `ModelScheduler` | In use |
| Specification / policy | `BatchCompatibility`, `shouldDispatchBatchSize` | In use |
| Composite batch processing | `DynamicBatcher` merge/split | In use |
| Object pool + least-inflight | `--instances`, `selectExecutorIndex()` | In use |

### Near-Term Pattern Additions

#### Step 7: Observability patterns

- Add Prometheus metrics export and latency histograms (already planned).
- Add structured logging with request id, model, queue/infer/total latency.
- Add trace spans for admit → queue → batch form → infer → split → respond.
  Prefer OpenTelemetry-compatible span names even if export starts minimal.
- Stabilize error taxonomy (`INVALID_ARGUMENT`, `QUEUE_FULL`, etc.) across HTTP,
  scheduler, and backend layers.

Exit criteria additions:

- A single request can be traced through queue, batch, and infer stages in logs
  or metrics labels.
- Error categories are stable and documented.

#### Step 7–8: Cancellation and deadline propagation

- Propagate deadline/cancel context from HTTP submit through scheduler into
  executor/backend calls.
- Extend beyond the current client-side timeout + `PendingRequest::cancelled`
  flag.
- Prefer `std::stop_token` or an internal cancel scope once C++20 usage is
  allowed in the build.

Exit criteria:

- Slow backend work can be abandoned when the client deadline expires.
- LLM decode loops honor cancellation (required for Step 8).

#### Step 7: Request pipeline / middleware chain

- Refactor infer handling into composable stages instead of growing
  `KServeRuntime::handleInfer()` monolithically:

```text
decode → validate → admit → schedule → encode
```

- Keep stages as plain functions or small callables; no framework required.
- Extension points: request id, rate limits, auth hooks, payload size checks.

Exit criteria:

- New cross-cutting infer behavior can be added without editing every route
  handler.

#### Step 3–7: Formal model state machine

- Make `ModelState` transitions explicit in `ModelRegistry`/`ModelHandle`:

```text
UNLOADED → LOADING → READY → UNLOADING → UNLOADED
                     ↓
                  UNAVAILABLE / FAILED
```

- Replace scattered readiness checks with transition helpers.
- Align with drain behavior from Step 5.

Exit criteria:

- Invalid state transitions are rejected or logged.
- Readiness endpoints reflect state machine state deterministically.

### Mid-Term Pattern Additions

#### Step 8: Separate scheduling strategies

- Do not extend tensor dynamic batching for LLM/token workloads.
- Add a second scheduler strategy (or scheduler policy interface) for:
  prefill, decode, KV cache slots, streaming, context limits.
- Share admission, metrics, cancellation, and error mapping; not batch merge/split.

Exit criteria:

- Tensor and LLM paths share lifecycle/metrics but not batch-formation logic.

#### Step 8+: Backend plugin registry

- Evolve `ExecutorFactory` into an explicit backend registry keyed by backend id
  and capabilities (tensor vs token).
- Support test doubles and optional compile-time backend registration.

Exit criteria:

- New backend ids can be registered without editing a central switch statement.

#### Post–Step 8: Control plane vs data plane split

- Separate model load/drain/reload (control) from infer hot path (data).
- Required before multi-model hot reload and version-specific policies.

Exit criteria:

- Model reload/drain does not block in-flight inference beyond defined drain
  semantics.

### Long-Term / Scale-Only Patterns

Add these only when profiling or deployment pressure justifies the complexity:

| Pattern | Trigger | Notes |
|---------|---------|-------|
| Async HTTP reactor | High connection concurrency | Replace thread-per-client `HttpServer` with epoll/io_uring/Asio-style event loop |
| Zero-copy / borrowed tensor buffers | Copy overhead in hot path | Reduce `vector<double>` copies; consider external buffer lifetime rules |
| Arena / PMR allocators | Allocation churn under load | Per-request or per-batch memory pools for tensor staging |
| Structured result types (`std::expected` / typed `Result`) | Error-handling bugs | Gradual adoption at executor/scheduler boundaries |
| gRPC V2 surface | Client demand | Parallel adapter layer; reuse scheduler and executor boundaries |

### Explicit Non-Goals

Do not introduce these into this repository unless requirements change materially:

- In-process microservices or event sourcing
- CQRS across infer requests
- Heavy DDD aggregate hierarchies
- Actor-model rewrite of the scheduler
- Template-heavy CRTP frameworks for hot paths

Prefer correctness, stable protocol behavior, and operability before transport
or memory micro-optimizations.


```text
HttpServer
  -> KServeRuntime
    -> ModelRegistry
      -> ModelHandle
        -> Scheduler
          -> ExecutionInstancePool
            -> neuriplo InferenceInterface
```

Planned modules:

```text
src/
  config/
    RuntimeConfig
  http/
    HttpServer
    HttpRequest
    HttpResponse
  kserve/
    KServeRuntime
    KServeV2Codec
    KServeErrors
  model/
    ModelRegistry
    ModelHandle
    ModelMetadata
    ModelState
  scheduler/
    RequestQueue
    Scheduler
    DynamicBatcher
    LlmScheduler
  execution/
    ExecutionInstance
    ExecutionInstancePool
    NeuriploExecutor
    StubExecutor
  metrics/
    MetricsRegistry
  util/
    Json
    Logging
    Time
```

The current scaffold is intentionally smaller. The module layout should evolve
toward this structure as soon as the protocol and model lifecycle need tests.

## Runtime Configuration

Initial CLI flags:

```text
--host 0.0.0.0
--port 8080
--model-name <name>
--model-path <path>
--backend <neuriplo-backend-id>
--use-gpu true|false
--batch-size 1
--input-size <shape>
--max-request-bytes <bytes>
--max-queue-size <count>
--request-timeout-ms <ms>
--instances <count>
```

KServe/container environment defaults:

```text
MODEL_NAME=<name>
STORAGE_URI=<original KServe storage URI, if provided>
PROTOCOL=v2
```

When running inside KServe, `--model-path` should default to `/mnt/models` so
the runtime works with KServe's storage initializer and PVC/S3/GCS/HF-backed
model delivery without custom download logic inside the server.

Later config file:

```yaml
server:
  host: 0.0.0.0
  port: 8080
  max_request_bytes: 67108864

models:
  - name: yolo
    path: /models/yolo.onnx
    backend: onnx_runtime
    batch_size: 1
    instances:
      - device: cpu
        count: 2
    scheduler:
      max_queue_size: 1024
      timeout_ms: 1000
      dynamic_batching:
        enabled: false
        max_batch_size: 8
        preferred_batch_sizes: [2, 4, 8]
        max_queue_delay_us: 2000
```

## Request Flow

### Raw Tensor Model

```text
1. Client calls POST /v2/models/{model}/infer.
2. KServeRuntime validates route and model readiness.
3. KServeV2Codec parses tensors into internal tensor buffers.
4. Admission control checks request size, queue depth, and timeout budget.
5. Scheduler enqueues request.
6. Scheduler forms a batch if dynamic batching is enabled.
7. ExecutionInstance runs neuriplo inference.
8. Scheduler splits batched outputs back to individual requests.
9. KServeV2Codec serializes response.
```

### Metadata Flow

```text
1. ModelRegistry loads one model during startup.
2. NeuriploExecutor calls backend metadata APIs.
3. Metadata is converted to KServe model metadata.
4. GET /v2/models/{model} returns stable metadata.
```

### Same-Machine Local Serving

```text
neuriplo-kserve-runtime --model-name yolo --model-path /models/yolo.onnx --backend onnx_runtime
vision-inference --endpoint http://127.0.0.1:8080 --model yolo --type yolo --source data/dog.jpg
```

`vision-inference` does not link `neuriplo` in this design. It only needs a
KServe client and task preprocessing/postprocessing through `vision-core`.

### KServe Custom Runtime Serving

```text
1. User creates a KServe InferenceService with modelFormat: neuriplo.
2. KServe selects the explicit neuriplo ServingRuntime.
3. KServe storage initialization makes the model artifact available at /mnt/models.
4. Kubernetes probes call /v2/health/live and /v2/health/ready.
5. Clients call KServe ingress, which forwards V2 requests to the runtime.
```

## Model Lifecycle

Model states:

```text
UNLOADED
LOADING
READY
UNAVAILABLE
UNLOADING
```

First milestone supports only:

```text
startup load -> READY or process failure
```

Later:

```text
reload
graceful unload
multiple models
version selection
```

Readiness rules:

- `/v2/health/live` returns healthy if the process event loop is running.
- `/v2/health/ready` returns ready only if all configured startup models are
  ready and schedulers can accept work.
- `/v2/models/{model}/ready` returns ready only for a loaded model whose
  execution pool is available.

## Scheduler Design

There should be one scheduler per model. A single global thread pool is not
enough because batching, queue limits, and device pressure are model-specific.

Initial scheduler:

```text
bounded FIFO queue
N execution instances
least-inflight instance selection
request timeout
overload response
```

Dynamic batching scheduler:

```text
queue first compatible request
wait up to max_queue_delay_us
scan queue for compatible candidates; skip incompatible; fulfill expired inline
append compatible requests until merged tensor batch dimension reaches max_batch_size
honor preferred_batch_sizes against merged tensor batch dimension
dispatch batch
split outputs
```

Compatibility check:

- same model name and version
- same input count
- same tensor names
- same dtype
- same shape except batch dimension
- same relevant request parameters

Backpressure:

- reject immediately when request body exceeds configured limit
- reject when model queue is full
- reject or expire when deadline cannot be met
- return `429` or `503` with structured error body

## LLM / llama.cpp / Cactus Support

LLM serving is a separate policy from tensor model batching. It covers existing
`neuriplo` llama.cpp and Cactus backends, including GGUF-style local models and
mobile/edge-oriented runtimes where Cactus is the execution backend.

KServe V2 can represent prompts as `BYTES` tensors, but LLM serving needs
token-aware scheduling:

```text
prompt prefill
decode loop
KV cache slots
streaming
cancellation
context length enforcement
sampling parameters
```

Planned LLM endpoints:

```text
POST /v2/models/{model}/infer
POST /v1/completions
POST /v1/chat/completions
POST /v1/embeddings
```

The `/v1/*` endpoints are optional and should not be required for KServe
compliance. They are useful for ecosystem compatibility for llama.cpp and
Cactus-backed LLM models, but they must share model lifecycle, metrics,
timeouts, and cancellation behavior with the KServe V2 path.

LLM request convention for KServe V2:

```json
{
  "inputs": [
    {
      "name": "prompt",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["Explain KServe briefly."]
    }
  ],
  "parameters": {
    "max_tokens": 128,
    "temperature": 0.7
  }
}
```

LLM response convention:

```json
{
  "outputs": [
    {
      "name": "text",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["..."]
    }
  ]
}
```

## Observability

Expose Prometheus metrics on:

```text
GET /metrics
```

Required metrics:

- request count by model, method, and status
- request latency histogram
- queue latency histogram
- backend inference latency histogram
- batch size histogram
- queue depth gauge
- in-flight requests gauge
- model load success/failure counters
- backend error counters
- process memory gauge

Structured logs should include:

- timestamp
- severity
- request id
- model name
- model version
- backend
- route
- status
- queue latency
- inference latency
- total latency
- batch size
- error code

Payload logging should be opt-in and redaction-aware. The runtime should emit
request/response byte counts by default, not raw tensors or prompts.

## Error Model

Use stable error categories:

```text
INVALID_ARGUMENT
MODEL_NOT_FOUND
MODEL_NOT_READY
QUEUE_FULL
DEADLINE_EXCEEDED
BACKEND_ERROR
UNAVAILABLE
INTERNAL
```

HTTP mapping:

```text
400 INVALID_ARGUMENT
404 MODEL_NOT_FOUND
409 MODEL_NOT_READY
429 QUEUE_FULL
504 DEADLINE_EXCEEDED
500 BACKEND_ERROR / INTERNAL
503 UNAVAILABLE
```

Error response:

```json
{
  "error": {
    "code": "MODEL_NOT_READY",
    "message": "model yolo is not ready"
  }
}
```

## Container And KServe Manifests

Initial files:

```text
docker/Dockerfile
deploy/kserve/cluster-serving-runtime.yaml
deploy/kserve/inferenceservice.yaml
deploy/kserve/inferencegraph.yaml
deploy/kserve/canary-inferenceservice.yaml
```

The `ServingRuntime` should declare:

```yaml
protocolVersions:
  - v2
supportedModelFormats:
  - name: neuriplo
    version: "1"
    autoSelect: false
```

The runtime may later add explicit backend-specific formats after validation:

```yaml
supportedModelFormats:
  - name: onnx
    version: "1"
    autoSelect: false
  - name: openvino
    version: "1"
    autoSelect: false
  - name: tensorrt
    version: "1"
    autoSelect: false
  - name: gguf
    version: "1"
    autoSelect: false
```

Do not claim broad auto-selection until backend behavior and metadata mapping
are validated.

The example `InferenceService` should demonstrate:

- explicit `runtime: neuriplo-kserve-runtime`
- `protocolVersion: v2`
- `storageUri` mounted model artifacts
- resource requests/limits suitable for CPU-only smoke tests
- readiness/liveness probe compatibility

The canary example should demonstrate changing either the runtime image or model
artifact with `canaryTrafficPercent`. The InferenceGraph example should keep the
runtime response shape stable enough for sequence, switch, splitter, and
ensemble routing.

## Sibling Repo Work

### `neuriplo`

Branch: `neuriplo-kserve-runtime`

Likely changes:

- Add a stable serving-facing factory API if the current setup function is too
  CLI-oriented.
- Document backend thread-safety.
- Ensure one backend instance per execution instance is safe.
- Expose enough metadata to build KServe metadata responses.
- Add tests for metadata and repeated inference calls per backend.

Avoid:

- KServe protocol types in core backend interfaces.
- Pulling HTTP/gRPC dependencies into the core library.

### `vision-core`

Branch: `neuriplo-kserve-runtime`

Likely changes:

- Only add shared tensor/protocol-neutral types if needed by both clients and
  server.
- Keep task result semantics unchanged.

Avoid:

- Changing model type strings.
- Changing preprocessing/postprocessing outputs.
- Changing result variants for server convenience.

### `vision-inference`

Branch: `neuriplo-kserve-runtime`

Likely changes:

- Add KServe V2 client abstraction.
- Replace direct `neuriplo` fetch/link in remote-only mode.
- Add CLI flags for endpoint/model/version/timeout.
- Keep local app flow and output behavior stable.

Avoid:

- In-process backend fallback if the design goal is server-only inference.
- Changing CLI behavior for existing task options unless explicitly migrated.

## Step Roadmap

### Step 0: Scaffold

- Create repo.
- Add CMake project.
- Add basic HTTP server.
- Add health and metadata endpoints.
- Add placeholder `/infer`.
- Add plan and README.

Exit criteria:

- Builds locally.
- `curl /v2/health/live` returns 200.
- `curl /v2/health/ready` returns 200 for stub model.
- `curl /v2/models/demo` returns model metadata.

### Step 1: KServe Packaging

- Add Dockerfile.
- Add `ClusterServingRuntime` manifest for `modelFormat: neuriplo`.
- Add example `InferenceService`.
- Add `/mnt/models` and KServe environment default handling.
- Document local KServe deployment flow.

Exit criteria:

- Runtime image can be referenced by a KServe `InferenceService`.
- KServe storage-initialized model path resolves to `/mnt/models`.
- Health/readiness endpoints are compatible with Kubernetes probes.

### Step 2: Protocol Correctness

- Add KServe V2 request/response codec.
- Add structured errors.
- Add route tests.
- Add content-length/body-size enforcement.
- Add request id handling.
- Add model version route parsing and metadata `versions`.
- Validate tensor names, shapes, datatypes, and requested outputs.

Exit criteria:

- Invalid model returns stable 404.
- Invalid JSON returns stable 400.
- Stub inference echoes deterministic tensor response.
- Request `id` is preserved in the response.
- Unsupported datatype/output requests fail predictably.

### Step 3: Model Registry And Executor Abstraction

- Add `Executor` interface.
- Add `StubExecutor`.
- Add model states.
- Add startup model loading.
- Add metadata conversion.

Exit criteria:

- Stub executor can be replaced without route changes.
- Readiness tracks model state.

### Step 4: Atomic neuriplo Integration

Step 4 integrates real `neuriplo` execution in independently buildable
substeps. Routes already delegate through `ModelRegistry` and `Executor`, but
`ExecutionRequest` does not yet preserve parsed input tensor data. Real backend
execution must start by carrying validated KServe inputs through the executor
boundary.

#### Step 4.1: Preserve KServe Input Tensors

- Add an input tensor representation to executor-facing request types.
- Have `KServeV2Codec` validate and pass input `name`, `datatype`, `shape`, and
  `data` into `ExecutionRequest`.
- Keep `StubExecutor` behavior unchanged.

Exit criteria:

- Codec tests prove parsed input tensor data reaches `ExecutionRequest`.
- Stub inference responses remain compatible with existing tests.

#### Step 4.2: Define A neuriplo Adapter Boundary

- Add a small internal adapter interface or wrapper so tests can fake
  `neuriplo` behavior before linking the real library.
- Keep `neuriplo`-specific types out of HTTP, routing, and registry code.
- Keep `Executor` as the only runtime execution interface used by
  `ModelRegistry` and routes.

Exit criteria:

- Unit tests can inject fake backend metadata, load failures, and inference
  results without linking real `neuriplo`.
- Public route and registry types do not include `neuriplo` headers.

#### Step 4.3: Add NeuriploExecutor Skeleton

- Implement `Executor` using the adapter boundary.
- Load model metadata from the adapter.
- Support construction or load failure with stable registry failed-state
  behavior.

Exit criteria:

- Fake-adapter tests cover load success, load failure, metadata mapping, and
  readiness behavior.
- Failed real-backend startup keeps `/v2/health/ready` false.

#### Step 4.4: Wire Backend Selection

- Keep `stub` mapped to `StubExecutor`.
- Map explicitly supported real backend ids to `NeuriploExecutor`.
- Keep unknown backend ids failing predictably.

Exit criteria:

- Factory tests cover `stub`, supported real backend ids, and unknown backend
  ids.
- Unsupported backend configuration produces a stable startup or readiness
  failure.

#### Step 4.5: Convert Request Inputs

- Convert supported KServe JSON tensor datatypes and shapes into `neuriplo`
  input buffers.
- Start with the minimum datatype needed by the chosen smoke backend, likely
  `FP32`.
- Return stable inference errors for unsupported or malformed backend-bound
  tensors.

Exit criteria:

- Fake-adapter tests cover inference success, unsupported datatypes, malformed
  shapes, and backend-bound tensor validation errors.
- Dense JSON tensor data is supported for the first smoke backend.

#### Step 4.6: Convert Backend Outputs

- Convert dense `neuriplo` outputs into `ExecutionResponse`.
- Preserve requested-output filtering semantics.
- Keep KServe response JSON shape compatible with existing tests.

Exit criteria:

- Fake-adapter tests cover dense output conversion and requested-output
  filtering.
- Existing KServe response shape tests continue to pass.

#### Step 4.7: Link Real neuriplo

- Add the selected CMake discovery path: package, sibling checkout, or vendored
  path.
- Keep the build usable without real `neuriplo` unless the real backend option
  is enabled.

Exit criteria:

- Default `stub` builds still work without `neuriplo` installed.
- Enabling the real backend option links the selected `neuriplo` target.

#### Step 4.8: Add Smoke Model Validation

- Pick one first backend, preferably ONNX Runtime unless the `neuriplo` API
  strongly favors OpenCV DNN.
- Add a small model fixture or documented local smoke command.
- Compare runtime `/infer` output against direct backend execution.

Exit criteria:

- ONNX Runtime or OpenCV DNN smoke model runs through `/infer`.
- Metadata comes from `neuriplo`.
- Single request output matches direct backend output.

### Step 5: Scheduler And Autoscaling Behavior

- Add bounded request queue.
- Add worker thread.
- Add timeouts.
- Add overload behavior.
- Add least-inflight instance selection.
- Add graceful shutdown/draining behavior.
- Add concurrency, queue depth, and latency metrics needed for HPA/KEDA/custom
  autoscaling.

Exit criteria:

- Queue depth is bounded.
- Overload produces `429` or `503`.
- Multiple concurrent requests complete without data races.
- Readiness flips false while draining.

### Step 6: Dynamic Batching

- Add compatible-request grouping.
- Add max batch size and max queue delay.
- Split batched outputs.
- Add tests comparing batched vs single outputs.

Exit criteria:

- Batched outputs are shape/schema equivalent to single-request outputs.
- Batch size metrics are exposed.

### Step 7: Observability And KServe Deployment Examples

- Add `/metrics`.
- Add structured logs.
- Add optional payload logging controls.
- Add request-path trace spans (admit, queue, batch, infer, split).
- Add canary rollout example.
- Add InferenceGraph example.
- Add deployment documentation.

Exit criteria:

- Runtime deploys as a KServe custom serving runtime.
- Health/readiness work in Kubernetes.
- Metrics scrape succeeds.
- Logs or metrics can follow one request through queue and infer stages.
- Canary example documents runtime-image and model-artifact rollout.
- InferenceGraph example can route to the runtime without custom response
  adapters.

### Step 8: LLM Backends (llama.cpp and Cactus)

- Add LLM model config for llama.cpp and Cactus backends.
- Add GGUF/local model artifact conventions where applicable.
- Add backend selection for llama.cpp vs Cactus through neuriplo.
- Add token-aware scheduler strategy separate from tensor dynamic batching.
- Add prefill, decode, cancellation, and deadline propagation into backend
  calls.
- Add KServe V2 `BYTES` prompt convention.
- Add streaming response design for generated tokens.
- Add OpenAI-compatible `/v1/completions`, `/v1/chat/completions`, and
  `/v1/embeddings` only for LLM models.

Exit criteria:

- llama.cpp and Cactus completion paths work through `/v2/models/{model}/infer`.
- Context limits and generation parameters are enforced.
- Cancellation/deadline behavior is defined and tested.
- LLM scheduling does not reuse tensor batch merge/split logic.

## Validation Strategy

Local checks first:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Protocol checks:

```bash
curl -s http://127.0.0.1:8080/v2
curl -s http://127.0.0.1:8080/v2/health/live
curl -s http://127.0.0.1:8080/v2/health/ready
curl -s http://127.0.0.1:8080/v2/models/demo
```

Integration checks:

```text
direct neuriplo backend output
  vs
neuriplo-kserve-runtime output
```

Downstream checks:

```text
vision-inference -> KServe client -> runtime -> neuriplo
```

Performance checks:

- p50/p95/p99 latency.
- queue latency.
- throughput under concurrency.
- batching throughput improvement.
- memory stability under repeated requests.

## Key Risks

- Backend thread-safety is unknown per backend.
- Tensor dtype mapping must be explicit and tested.
- Dynamic batching can silently change output splitting if not constrained.
- LLM scheduling is not the same as tensor batching; llama.cpp and Cactus need
  explicit token, context, cancellation, and memory-pressure policies.
- KServe V2 string/BYTES conventions for LLMs need documentation.
- Pulling server dependencies into `neuriplo` core would create dependency
  creep if the boundary is not enforced.
- `vision-inference` removing direct `neuriplo` linkage is a larger migration
  than adding a remote backend adapter.

## First Engineering Decisions

1. Keep this as a separate repository.
2. Keep `neuriplo` server-side only.
3. Keep `vision-inference` as a KServe client in the target architecture.
4. Start with raw tensor inference.
5. Package as a KServe custom `ServingRuntime` early, before backend-specific
   optimization.
6. Add task-aware serving only after tensor serving is stable.
7. Add llama.cpp and Cactus LLM support as a dedicated scheduling policy, not
   as ordinary dynamic batching.
8. Prefer correctness and stable protocol behavior before optimizing transport.
9. Evolve architecture using `DESIGN_PATTERNS.md` for current patterns and the
   "Architecture And Design Pattern Evolution" section for target patterns.

## Production Readiness Gap Analysis

All eight roadmap steps have snapshot documents (STEP1–STEP8.md), and the
runtime passes its full test suite. This section records what remains before
this is a production-facing serving runtime. Items are grouped by severity.

### Gaps Blocking Tensor / CV Production Use

These reflect real-backend roughness or missing non-functional requirements on
the tensor path:

- **Dense FP32-only neuriplo adapter**: `NeuriploExecutor` only converts dense
  `FP32` tensors. Other datatypes (INT8, FP16, INT32, etc.) are rejected.
  Multi-input models that do not match the metadata input order exactly need
  explicit ordering tests.
- **Thread-per-client HTTP server**: The current `HttpServer` spawns one
  detached thread per client connection. Under production concurrency this
  creates unbounded thread pressure. An async reactor (epoll/io_uring) is
  deferred to the long-term list (see Design Pattern Evolution) and is not a
  first-launch blocker, but it must be addressed before sustained production
  load.
- **gRPC V2 surface**: Not implemented. KServe clients that require gRPC will
  not work until this adapter layer is added (also deferred to long-term).
- **No container readiness-gate health checks for model load**: The readiness
  endpoint flips based on model state, but the startup load-and-ready window
  does not yet integrate with Kubernetes `startupProbe` / `initialDelaySeconds`
  documentation.
- **No formal model state machine**: `ModelRegistry` readiness is still inside
  scattered checks rather than explicit `UNLOADED → LOADING → READY` transition
  helpers (planned in Step 3–7 design patterns).
- **Request pipeline not composable**: The infer handler is monolithic rather
  than a chain of `decode → validate → admit → schedule → encode` stages
  (planned in Step 7 middleware). Adding new cross-cutting behavior requires
  editing route handlers directly.
- **Request ID generation is weak**: The runtime currently accepts a
  client-supplied `id` or generates a counter-based ID. No UUID or
  collision-resistant generation.

### Gaps Blocking LLM Production Use

These are the items remaining from Step 8 (LLM Backends) that prevent any real
LLM workload:

- **No real LLM backend execution**: `LlmScheduler` dispatches only to the
  `StubExecutor`. Real llama.cpp and Cactus decode is not wired through
  `NeuriploExecutor`. Token-by-token decode loops, KV cache management, and
  generation parameter passthrough are not validated against real hardware.
- **No streaming responses**: Only non-streaming (full-response) infer is
  implemented. SSE or chunked transfer encoding for token-by-token output is
  missing.
- **No OpenAI-compatible endpoints**: `/v1/completions`,
  `/v1/chat/completions`, and `/v1/embeddings` are not implemented (Step 8.6).
- **Token-accurate context enforcement**: The current `LlmScheduler` uses a
  character-count proxy for context-length checks. Real tokenization is not
  performed, so context limits are approximate and will break for multi-byte or
  non-English text.
- **No cancellation or deadline propagation into backend decode loops**:
  Cancellation is only at the client-timeout level (`PendingRequest::cancelled`).
  A slow decode loop cannot be abandoned when the client deadline expires
  (Step 8.7). This is a correctness requirement for LLM production.
- **No KV-cache memory pressure policy**: Context length × KV cache slots can
  silently exceed available memory. No admission control ties memory to slot
  occupancy.

### Gaps Common To Both Paths

- **No `InferenceGraph` routing validation**: The example YAML exists, but no
  integration test proves the runtime response shape survives sequence, switch,
  or splitter routing.
- **No canary rollout validation**: The canary YAML exists, but no test proves
  the runtime behaves correctly under traffic-split semantics.
- **No autoscaling integration tests**: HPA/KEDA/custom autoscaler behavior has
  not been validated with the exposed metrics.
- **No structured error documentation for clients**: Error taxonomy is
  implemented in code but not documented for external consumers.
- **No latency SLO / performance baseline**: No p50/p95/p99 benchmarks exist
  in CI or documentation.

### Explicitly Deferred (Long-Term / Scale-Only)

These are recorded in the Design Pattern Evolution section and are NOT blockers
for a first production launch:

| Item | Trigger |
|------|---------|
| Async HTTP reactor (epoll/io_uring) | High connection concurrency |
| Zero-copy / borrowed tensor buffers | Copy overhead in hot path |
| Arena / PMR allocators | Allocation churn under load |
| gRPC V2 surface | Client demand |
| Multi-model hot reload | Deployment need |
| Backend plugin registry (no central switch) | Backend count growth |
| Control plane / data plane split | Multi-model + drain complexity |

### Recommended Next Actions

1. **For tensor CV serving**: Harden the neuriplo adapter to cover the datatypes
   used by the first target model, then validate end-to-end with a real ONNX
   model (beyond the identity smoke test). Add a latency baseline.
2. **For LLM serving**: Wire real llama.cpp decode through `NeuriploExecutor`,
   implement token-accurate context enforcement, add cancellation propagation,
   then add streaming. OpenAI-compatible endpoints should follow after the KServe
   V2 LLM path is stable.
3. **For Kubernetes deployment**: Add startup probe documentation, validate
   liveness/readiness under load, and smoke-test the ClusterServingRuntime
   manifest in a real cluster.
