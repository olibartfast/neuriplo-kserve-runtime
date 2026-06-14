# neuriplo-kserve-runtime

KServe-compatible C++ inference runtime for [neuriplo] backends.

This repository is the serving data plane for [neuriplo]. It owns the KServe
V2 protocol surface, request admission, scheduling, batching, model lifecycle,
and operational endpoints. The actual backend execution remains owned by
[neuriplo].

## Runtime Scope

- HTTP and optional gRPC KServe V2 health, metadata, and inference endpoints.
- Multi-model registry with admin load, unload, reload, and version activation endpoints.
- Bounded request handling with tensor and LLM scheduler paths.
- Stub execution by default, with optional real `neuriplo` adapter wiring.
- CMake-based C++ build.

## Request Flow

1. KServe client
2. `neuriplo-kserve-runtime`
3. model registry and lifecycle snapshot
4. scheduler and executor
5. [neuriplo] backend instance
6. KServe V2 response

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```


## Development

### Configure and build

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

The default build does not require `neuriplo`. To compile the real adapter
(neurip is auto-fetched from GitHub via CMake FetchContent; pin its tag in
`versions.env`):

```bash
cmake --preset real-onnx
cmake --build --preset real-onnx
ctest --preset real-onnx
```

If you need to iterate on neuriplo itself, point CMake at a local checkout:

```bash
cmake --preset real-onnx -DNEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR=/path/to/neuriplo
cmake --build --preset real-onnx
```

### Unit tests

Unit tests are built as `neuriplo-kserve-runtime-tests` and registered with CTest.

```bash
cmake --build --preset debug
ctest --preset debug
```

### gRPC support (optional)

gRPC V2 support is disabled by default. Enable it with the CMake option:

```bash
cmake -S . -B build/grpc -DNEURIPLO_RUNTIME_ENABLE_GRPC=ON
cmake --build build/grpc
```

Requires `libgrpc++-dev` and `protobuf-compiler-grpc` system packages:

```bash
sudo apt-get install -y libgrpc++-dev protobuf-compiler-grpc
```

When compiled with gRPC support, run both HTTP and gRPC on separate ports:

```bash
./build/grpc/neuriplo-kserve-runtime --port 8080 --grpc-port 9000
```

If `--grpc-port` is set but gRPC was not compiled in, the runtime prints a
warning and ignores it.

### Lint and format

```bash
scripts/check-format.sh
cmake --preset lint
cmake --build --preset lint
```

### Debugger

VS Code launch configurations are provided for the runtime and unit tests. Build the
`debug` preset before launching, or let the configured pre-launch task run it.

### Sanitizers and memory checks

```bash
cmake --preset asan && cmake --build --preset asan && ctest --preset asan
cmake --preset ubsan && cmake --build --preset ubsan && ctest --preset ubsan
cmake --preset tsan && cmake --build --preset tsan && ctest --preset tsan
valgrind --leak-check=full --error-exitcode=1 ./build/debug/neuriplo-kserve-runtime-tests
```

## Run

```bash
./build/neuriplo-kserve-runtime \
  --model-name demo \
  --model-path /models/model.onnx \
  --backend onnx_runtime \
  --max-request-bytes 67108864 \
  --max-queue-size 64 \
  --request-timeout-ms 30000 \
  --instances 1 \
  --port 8080 \
  --grpc-port 9000
```

`--backend` accepts any neuriplo tensor backend compiled into the binary:
`onnx_runtime`, `opencv_dnn`, `openvino`, `tensorrt`, `libtorch`,
`libtensorflow`, `migraphx`, `executorch`, `litert` (TFLite), plus the LLM
backends `llamacpp` / `cactus` / `ggml`. A backend is only servable if neuriplo
was built with it (see `DEFAULT_BACKEND` / `NEURIPLO_BACKENDS`); the runtime
reports its model `platform` as `neuriplo_<backend>`.

The runtime also reads KServe-friendly environment defaults. CLI flags override
environment values.

```text
MODEL_NAME   default model name
MODEL_PATH   default model path
BACKEND      default backend name
STORAGE_URI  retained for diagnostics; the server does not download it
MAX_REQUEST_BYTES  maximum accepted HTTP request size
```

Scheduler flags:

```text
--max-queue-size <count>       bounded per-model request queue capacity (default 64)
--request-timeout-ms <ms>      request deadline in milliseconds (default 30000)
--instances <count>            worker threads and executor instances per model (default 1)
--dynamic-batching-enabled true|false  enable compatible-request batching (default false)
--max-batch-size <count>       maximum requests per batch (default 1)
--max-queue-delay-us <us>      max wait for additional compatible requests (default 0)
--preferred-batch-sizes <csv>  optional preferred batch sizes, e.g. 2,4,8
```

LLM scheduler flags (llama.cpp, Cactus, GGUF models):

```text
--scheduler-strategy <tensor|llm>  scheduler policy (default tensor; LLM backends auto-detect)
--context-length <tokens>          max prompt context window (default 4096)
--kv-cache-slots <count>           concurrent decode slots (default 1)
--max-tokens <count>               default generation limit (default 256)
--temperature <float>              sampling default (default 0.8)
--top-p <float>                    nucleus sampling default (default 0.95)
--top-k <count>                    top-k sampling default (default 40)
--streaming-enabled true|false     SSE/chunked streaming toggle (default false)
```

For GGUF/local LLM artifacts in KServe, mount the model directory at `/mnt/models`
(for example `/mnt/models/model.gguf` or `/mnt/models/` containing the GGUF file).
Use `--backend llamacpp` or `--backend cactus` with the LLM scheduler path.

When `MODEL_PATH` is unset and `/mnt/models` exists, the runtime uses
`/mnt/models` as the model path. This matches KServe storage-initializer and PVC
mount conventions.

## Endpoints

### HTTP

```text
GET  /v2
GET  /v2/health/live
GET  /v2/health/ready
GET  /v2/models/{model_name}
GET  /v2/models/{model_name}/ready
POST /v2/models/{model_name}/infer
GET  /v2/models/{model_name}/versions/{version}
GET  /v2/models/{model_name}/versions/{version}/ready
POST /v2/models/{model_name}/versions/{version}/infer
GET  /metrics
```

### gRPC

When compiled with `NEURIPLO_RUNTIME_ENABLE_GRPC=ON`, the `GRPCInferenceService`
(KServe V2 gRPC protocol) is available on the configured `--grpc-port`:

```
service GRPCInferenceService {
  rpc ServerLive(ServerLiveRequest) returns (ServerLiveResponse);
  rpc ServerReady(ServerReadyRequest) returns (ServerReadyResponse);
  rpc ModelReady(ModelReadyRequest) returns (ModelReadyResponse);
  rpc ServerMetadata(ServerMetadataRequest) returns (ServerMetadataResponse);
  rpc ModelMetadata(ModelMetadataRequest) returns (ModelMetadataResponse);
  rpc ModelInfer(ModelInferRequest) returns (ModelInferResponse);
}
```

The gRPC server reuses the same `ModelRegistry`, scheduler, and executor
pipeline as the HTTP path. Request admission, batching, timeouts, and error
mapping behave identically across both transports.

The stub model exposes static version `1`. Stub inference validates the V2 JSON
request shape and returns a deterministic placeholder output:

```bash
curl -X POST http://127.0.0.1:8080/v2/models/demo/infer \
  -H 'Content-Type: application/json' \
  -d '{"id":"example","inputs":[{"name":"input","shape":[1,3,224,224],"datatype":"FP32","data":[]}]}'
```

## Observability

### Metrics

`GET /metrics` returns Prometheus text-format metrics:

```text
neuriplo_scheduler_queue_depth       (gauge)
neuriplo_scheduler_in_flight_requests (gauge)
neuriplo_scheduler_requests_accepted_total   (counter)
neuriplo_scheduler_requests_rejected_total   (counter)
neuriplo_scheduler_requests_timed_out_total  (counter)
neuriplo_scheduler_queue_wait_seconds        (counter)
neuriplo_scheduler_execution_seconds         (counter)
neuriplo_scheduler_request_total_seconds     (counter)
neuriplo_scheduler_batches_formed_total      (counter)
neuriplo_scheduler_batched_requests_total    (counter)
neuriplo_scheduler_batch_formation_seconds   (counter)
neuriplo_scheduler_batch_execution_seconds   (counter)
neuriplo_http_infer_requests_total           (counter)
neuriplo_http_infer_requests_success_total   (counter)
neuriplo_http_infer_requests_failure_total   (counter)
neuriplo_model_load_success_total            (counter)
neuriplo_model_load_failure_total            (counter)
neuriplo_process_resident_memory_bytes       (gauge)
```

All scheduler metrics include a `model` label. Process memory reports
`VmRSS` from `/proc/self/status`.

Sample Prometheus scrape config:

```yaml
scrape_configs:
  - job_name: neuriplo-kserve
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: /metrics
```

### Structured Logging

Logs are emitted as JSON lines to stderr. Each event includes:

```json
{"timestamp":"2026-01-01T00:00:00.000Z","severity":"info","message":"infer request completed",
 "model":"demo","model_version":"1","request_id":"abc","route":"/v2/models/demo/infer",
 "status":200,"queue_latency_ns":50000,"infer_latency_ns":30000,"total_latency_ns":80000,
 "batch_size":1,"request_bytes":100}
```

### Payload Logging

By default only byte counts are logged. Enable full payload logging with
`--log-payloads true`. This logs raw JSON request and response bodies for
diagnostic purposes. Do not enable in production without redaction controls.

### Error Taxonomy

Stable error categories mapped to HTTP status codes:

| Code                | HTTP | Description              |
|---------------------|------|--------------------------|
| INVALID_ARGUMENT    | 400  | Malformed request        |
| MODEL_NOT_FOUND     | 404  | Unknown model            |
| MODEL_NOT_READY     | 409  | Model not loaded yet     |
| QUEUE_FULL          | 429  | Request queue at capacity|
| DEADLINE_EXCEEDED   | 504  | Request timed out        |
| BACKEND_ERROR       | 500  | Executor failure         |
| UNAVAILABLE         | 503  | Model draining/failed    |
| INTERNAL            | 500  | Unexpected error         |

## KServe Packaging

Build a local image:

```bash
docker build -f docker/Dockerfile -t neuriplo-kserve-runtime:latest .
```

For a local minikube-style cluster, make the image available to the cluster
before applying the runtime. Then install the custom runtime and example service:

```bash
kubectl apply -f deploy/kserve/cluster-serving-runtime.yaml
kubectl apply -f deploy/kserve/inferenceservice.yaml
```

The example `InferenceService` uses `modelFormat.name: neuriplo`,
`protocolVersion: v2`, and explicitly selects `runtime:
neuriplo-kserve-runtime`. Replace the placeholder `storageUri` with the location
of your model artifact.

Check readiness and the current V2 metadata surface:

```bash
kubectl get inferenceservice neuriplo-demo
curl http://<host>/v2
curl http://<host>/v2/health/ready
curl http://<host>/v2/models/neuriplo-demo
```

### Canary Rollout

The canary example demonstrates changing the runtime image or model artifact
with traffic splitting:

```bash
kubectl apply -f deploy/kserve/canary-inferenceservice.yaml
```

This deploys `neuriplo-demo-canary` with `canaryTrafficPercent: 30`, routing
30% of traffic to the canary predictor. Adjust the percentage and verify both
predictors return valid responses before promoting.

### InferenceGraph

The InferenceGraph example routes through the runtime without custom adapters:

```bash
kubectl apply -f deploy/kserve/inferencegraph.yaml
```

Response shape compatibility ensures sequence, switch, splitter, and ensemble
routing work without changes to runtime output formats.

The default image still uses the stub execution path unless it is built with
`NEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO=ON` and the corresponding `neuriplo`
backend dependencies.

[neuriplo]: https://github.com/olibartfast/neuriplo
