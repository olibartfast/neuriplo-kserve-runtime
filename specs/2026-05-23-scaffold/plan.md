# Step 0: Scaffold - plan

> Retrospective packet. Step 0 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 0
> snapshot (`plan/STEP0.md`, later `specs/history/steps/STEP0.md`) and the
> Step 0 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Tasks are the step's substeps. Each `T-` section is the implementation
record from the snapshot.

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

> This step was delivered as one unit; the sections above are its single
> task group.
