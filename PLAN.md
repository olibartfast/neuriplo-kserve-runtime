# neuriplo-kserve-runtime Implementation Plan

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
unless a later task-aware serving layer is explicitly designed.

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
- LLM/token scheduler for GGUF models when llama.cpp support is enabled.
- Metrics, health, readiness, and structured errors.
- Container image and KServe `ServingRuntime` manifests.

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
- No OpenAI-compatible API in the first CV milestone.
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

## High-Level Architecture

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
append compatible requests until max_batch_size
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

## LLM / llama.cpp / GGUF Support

GGUF serving is a separate policy from tensor model batching.

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
compliance. They are useful for ecosystem compatibility.

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
deploy/kserve/servingruntime.yaml
deploy/kserve/inferenceservice.yaml
```

The `ServingRuntime` should declare:

```yaml
protocolVersions:
  - v2
```

Supported model formats should start conservative:

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

## Milestones

### M0: Scaffold

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

### M1: Protocol Correctness

- Add KServe V2 request/response codec.
- Add structured errors.
- Add route tests.
- Add content-length/body-size enforcement.
- Add request id handling.

Exit criteria:

- Invalid model returns stable 404.
- Invalid JSON returns stable 400.
- Stub inference echoes deterministic tensor response.

### M2: Model Registry And Executor Abstraction

- Add `Executor` interface.
- Add `StubExecutor`.
- Add model states.
- Add startup model loading.
- Add metadata conversion.

Exit criteria:

- Stub executor can be replaced without route changes.
- Readiness tracks model state.

### M3: neuriplo Integration

- Link `neuriplo`.
- Add `NeuriploExecutor`.
- Load a configured backend/model once at startup.
- Convert KServe tensors to neuriplo input buffers.
- Convert neuriplo outputs to KServe tensors.

Exit criteria:

- ONNX Runtime or OpenCV DNN smoke model runs through `/infer`.
- Metadata comes from neuriplo.
- Single request output matches direct backend output.

### M4: Scheduler

- Add bounded request queue.
- Add worker thread.
- Add timeouts.
- Add overload behavior.
- Add least-inflight instance selection.

Exit criteria:

- Queue depth is bounded.
- Overload produces `429` or `503`.
- Multiple concurrent requests complete without data races.

### M5: Dynamic Batching

- Add compatible-request grouping.
- Add max batch size and max queue delay.
- Split batched outputs.
- Add tests comparing batched vs single outputs.

Exit criteria:

- Batched outputs are shape/schema equivalent to single-request outputs.
- Batch size metrics are exposed.

### M6: Observability And Containers

- Add `/metrics`.
- Add structured logs.
- Add Dockerfile.
- Add KServe `ServingRuntime` YAML.
- Add example `InferenceService`.

Exit criteria:

- Runtime deploys as a KServe custom serving runtime.
- Health/readiness work in Kubernetes.
- Metrics scrape succeeds.

### M7: llama.cpp / GGUF

- Add GGUF model config.
- Add llama.cpp-backed neuriplo executor path.
- Add token scheduler.
- Add KServe BYTES prompt convention.
- Optionally add OpenAI-compatible `/v1/*` endpoints.

Exit criteria:

- GGUF completion works through `/v2/models/{model}/infer`.
- Context limits and generation parameters are enforced.
- Cancellation/deadline behavior is defined.

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
- LLM scheduling is not the same as tensor batching.
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
5. Add task-aware serving only after tensor serving is stable.
6. Add llama.cpp/GGUF as a dedicated scheduling policy, not as ordinary dynamic
   batching.
7. Prefer correctness and stable protocol behavior before optimizing transport.

