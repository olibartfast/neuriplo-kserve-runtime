# Runtime Target Design - plan

> Retrospective packet. This is the original milestone target design for
> `neuriplo-kserve-runtime` (`plan/ROADMAP.md`, later
> `specs/history/target-design.md`), ported on 2026-10-05. It framed Steps 0-11;
> each step's scope and exit criteria now live in that step's packet. The live
> plan is [`../roadmap.md`](../roadmap.md); current patterns are in
> [`../architecture.md`](../architecture.md).

## Step index

| Step | Title | Packet |
| --- | --- | --- |
| Step 0 | Scaffold | [`2026-05-23-scaffold`](../2026-05-23-scaffold/requirements.md) |
| Step 1 | KServe Packaging | [`2026-05-24-kserve-packaging`](../2026-05-24-kserve-packaging/requirements.md) |
| Step 2 | Protocol Correctness | [`2026-05-24-protocol-correctness`](../2026-05-24-protocol-correctness/requirements.md) |
| Step 3 | Model Registry And Executor Abstraction | [`2026-05-26-model-registry-executor`](../2026-05-26-model-registry-executor/requirements.md) |
| Step 4 | Atomic neuriplo Integration | [`2026-05-27-neuriplo-integration`](../2026-05-27-neuriplo-integration/requirements.md) |
| Step 5 | Scheduler And Autoscaling Behavior | [`2026-05-29-scheduler-autoscaling`](../2026-05-29-scheduler-autoscaling/requirements.md) |
| Step 6 | Dynamic Batching | [`2026-05-30-dynamic-batching`](../2026-05-30-dynamic-batching/requirements.md) |
| Step 7 | Observability And KServe Deployment Examples | [`2026-05-31-observability-deployment-examples`](../2026-05-31-observability-deployment-examples/requirements.md) |
| Step 8 | LLM Backends (llama.cpp and Cactus) | [`2026-06-02-llm-backends`](../2026-06-02-llm-backends/requirements.md) |
| Step 9 | Production Hardening - Tensor Path | [`2026-06-04-tensor-production-hardening`](../2026-06-04-tensor-production-hardening/requirements.md) |
| Step 10 | LLM Path Completion | [`2026-06-04-llm-path-completion`](../2026-06-04-llm-path-completion/requirements.md) |
| Step 11 | Deployment Validation | [`2026-06-05-deployment-validation`](../2026-06-05-deployment-validation/requirements.md) |
| Step 12 | Multi-Component YOLO Integration | [`2026-06-05-yolo-platform-e2e`](../2026-06-05-yolo-platform-e2e/requirements.md) |
| Step 13 | Control Plane / Data Plane Split | [`2026-06-10-control-data-plane-split`](../2026-06-10-control-data-plane-split/requirements.md) |
| Step 14 | Multi-Model Hot Reload | [`2026-06-11-multi-model-hot-reload`](../2026-06-11-multi-model-hot-reload/requirements.md) |

The per-step scope and exit criteria that this design listed under "Step
Roadmap" are each step packet's `requirements.md`.

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
| gRPC V2 surface | Client demand | ✅ Implemented — parallel adapter reusing scheduler and executor boundaries |

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
