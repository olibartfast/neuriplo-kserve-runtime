# neuriplo-kserve-runtime

KServe-compatible C++ inference runtime for [neuriplo] backends.

This repository is the serving data plane for [neuriplo]. It owns the KServe
V2 protocol surface, request admission, scheduling, batching, model lifecycle,
and operational endpoints. The actual backend execution remains owned by
[neuriplo].

## Initial Scope

- HTTP KServe V2 health and metadata endpoints.
- Single-model runtime process.
- Bounded request handling.
- Raw tensor inference through the stub executor by default, with optional
  `neuriplo` executor wiring.
- CMake-based C++ build.

## Planned Flow

1. KServe / client
2. `neuriplo-kserve-runtime`
3. model registry
4. scheduler
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
against a sibling checkout:

```bash
cmake -S . -B build/real-neuriplo \
  -DNEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO=ON \
  -DNEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR=/path/to/neuriplo
cmake --build build/real-neuriplo
```

For the local sibling checkout and ONNX Runtime smoke test:

```bash
cmake --preset real-onnx
cmake --build --preset real-onnx
ctest --preset real-onnx
```

### Unit tests

Unit tests are built as `neuriplo-kserve-runtime-tests` and registered with CTest.

```bash
cmake --build --preset debug
ctest --preset debug
```

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
  --port 8080
```

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
```

When `MODEL_PATH` is unset and `/mnt/models` exists, the runtime uses
`/mnt/models` as the model path. This matches KServe storage-initializer and PVC
mount conventions.

## Endpoints

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
```

The stub model exposes static version `1`. Stub inference validates the V2 JSON
request shape and returns a deterministic placeholder output:

```bash
curl -X POST http://127.0.0.1:8080/v2/models/demo/infer \
  -H 'Content-Type: application/json' \
  -d '{"id":"example","inputs":[{"name":"input","shape":[1,3,224,224],"datatype":"FP32","data":[]}]}'
```

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

The default image still uses the stub execution path unless it is built with
`NEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO=ON` and the corresponding `neuriplo`
backend dependencies.

[neuriplo]: https://github.com/olibartfast/neuriplo
