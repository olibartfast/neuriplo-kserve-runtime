# Step 0: Current Implementation State

This document records what is implemented in this repository today. It is a
snapshot of the scaffold, not the target architecture.

## Current Scope

The repository currently builds a small C++17 HTTP server with a minimal
KServe/Open Inference Protocol V2-compatible surface.

Implemented endpoints:

```text
GET  /v2
GET  /v2/health/live
GET  /v2/health/ready
GET  /v2/models/{model_name}
GET  /v2/models/{model_name}/ready
POST /v2/models/{model_name}/infer
```

The runtime can be launched locally with CLI configuration for host, port,
model name, model path, and backend name.

## Implemented Components

### HTTP Server

Implemented in `src/HttpServer.cpp` and `src/HttpServer.hpp`.

- Opens an IPv4 TCP socket.
- Accepts one HTTP request per connection.
- Parses method, path, headers, and body.
- Supports `Content-Length`.
- Dispatches requests to a handler callback.
- Serializes JSON HTTP responses.
- Handles each client connection on a detached thread.

### Runtime Routing

Implemented in `src/KServeRuntime.cpp` and `src/KServeRuntime.hpp`.

- Routes basic KServe V2 server health and metadata requests.
- Routes model metadata, model readiness, and inference requests.
- Returns structured JSON error responses for unknown routes, missing models,
  malformed requests, and unavailable models.

### Runtime Configuration

Implemented in `src/RuntimeConfig.cpp` and `src/RuntimeConfig.hpp`.

Supported CLI flags:

```text
--host <host>
--port <port>
--model-name <name>
--model-path <path>
--backend <backend>
--help
--version
```

Default values:

```text
host:       0.0.0.0
port:       8080
model-name: demo
backend:    stub
```

### Model Registry

Implemented in `src/ModelRegistry.cpp` and `src/ModelRegistry.hpp`.

- Supports a single configured model.
- Marks the model ready immediately at startup.
- Returns hardcoded tensor metadata:

```text
input:  FP32 [1, 3, 224, 224]
output: FP32 [1, 1000]
```

- Sets the platform string to `neuriplo_<backend>`.

### Tests

Implemented in `tests/`.

Current test coverage checks:

- Runtime health and readiness endpoints.
- Model metadata response.
- Unknown model rejection.
- Placeholder inference route.
- Runtime configuration parsing.
- `--version` executable behavior through CTest.

## Current Build And Validation

The current scaffold builds and tests with:

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

At the time this file was added, the debug build and CTest suite passed.

## What Is Stubbed Or Missing

The repository is not yet ready for real `neuriplo`, `vision-core`, or
`vision-inference` integration.

Missing or stubbed items:

- No `neuriplo` dependency is linked.
- No executor abstraction exists yet.
- No `NeuriploExecutor` exists yet.
- `/infer` ignores the request body.
- `/infer` returns a fixed dummy output tensor.
- Model metadata does not come from the backend.
- Model readiness is not tied to real model loading.
- No scheduler, queue, timeout, batching, or backpressure implementation exists.
- No KServe `ServingRuntime` or `InferenceService` manifests exist.
- No container image definition exists.
- No `/mnt/models`, `STORAGE_URI`, or KServe storage-initializer behavior exists.
- No gRPC V2 support exists.
- No Prometheus `/metrics` endpoint exists.
- No LLM/token scheduler exists for llama.cpp or Cactus backends.
- No OpenAI-compatible `/v1/*` endpoints exist.

## Practical Status

This repository is currently at the scaffold stage:

```text
M0: Scaffold implemented
M1+: Not implemented
```

It is useful for validating basic process startup, routing, HTTP parsing, and
the shape of the initial KServe V2 surface. It is not yet useful for real model
execution or sibling repository integration.
