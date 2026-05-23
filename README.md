# neuriplo-kserve-runtime

KServe-compatible C++ inference runtime for neuriplo backends.

This repository is the serving data plane for `neuriplo`. It owns the KServe
V2 protocol surface, request admission, scheduling, batching, model lifecycle,
and operational endpoints. The actual backend execution remains owned by
`neuriplo`.

## Initial Scope

- HTTP KServe V2 health and metadata endpoints.
- Single-model runtime process.
- Bounded request handling.
- Raw tensor inference API placeholder.
- CMake-based C++ build.

## Planned Flow

```text
KServe / client
  -> neuriplo-kserve-runtime
    -> model registry
    -> scheduler
    -> neuriplo backend instance
    -> KServe V2 response
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Run

```bash
./build/neuriplo-kserve-runtime \
  --model-name demo \
  --model-path /models/model.onnx \
  --backend onnx_runtime \
  --port 8080
```

## Endpoints

```text
GET  /v2
GET  /v2/health/live
GET  /v2/health/ready
GET  /v2/models/{model_name}
GET  /v2/models/{model_name}/ready
POST /v2/models/{model_name}/infer
```
