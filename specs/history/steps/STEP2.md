# Step 2: Protocol Correctness

This document records what was implemented for Step 2. The runtime still uses
stub inference, but the KServe/Open Inference Protocol V2 request and response
surface is now parsed and validated instead of ignoring inference request
bodies.

## Current Scope

Step 2 adds:

- KServe V2 inference request parsing.
- Deterministic stub inference response serialization.
- Request `id` preservation.
- Model metadata `versions`.
- Versioned model metadata, readiness, and inference routes.
- Structured `400`, `404`, `405`, `409`, and `413` error behavior.
- Runtime `max_request_bytes` configuration.
- HTTP-layer oversized request and invalid `Content-Length` handling.
- Unit tests for codec, routing, config, and validation behavior.
- HTTP integration tests that exercise the server over a real socket.

Step 2 does not add real `neuriplo` execution, executor abstraction, scheduler,
batching, queueing, metrics, gRPC, Kubernetes end-to-end tests, or LLM/OpenAI
endpoints.

## KServe V2 Codec

Implemented in:

```text
src/KServeV2Codec.hpp
src/KServeV2Codec.cpp
tests/KServeV2CodecTest.cpp
```

The codec uses `nlohmann/json` through CMake `FetchContent`.

Implemented request handling:

- Parses JSON inference request bodies.
- Requires a top-level JSON object.
- Preserves optional string `id`.
- Requires non-empty `inputs`.
- Validates input `name`, `shape`, `datatype`, and `data`.
- Validates requested `outputs` when present.
- Returns parse/validation failures as structured errors through the runtime
  route handler.

The first codec intentionally validates only the static stub model metadata:

```text
input:  FP32 [1, 3, 224, 224]
output: FP32 [1, 1000]
```

It does not validate tensor data length yet because real buffer conversion is
planned for the executor/backend steps.

## Inference Response

Stub inference now returns a protocol-shaped deterministic response:

```json
{
  "model_name": "demo",
  "model_version": "1",
  "id": "request-id-if-provided",
  "outputs": [
    {
      "name": "output",
      "shape": [1, 1],
      "datatype": "FP32",
      "data": [0.0]
    }
  ]
}
```

If the request has no `id`, the response omits `id`.

## Model Versions

Implemented static model version support for the single configured model:

```text
version: 1
```

Supported routes now include:

```text
GET  /v2/models/{model_name}/versions/{version}
GET  /v2/models/{model_name}/versions/{version}/ready
POST /v2/models/{model_name}/versions/{version}/infer
```

Unknown versions return `404 MODEL_NOT_FOUND`.

Model metadata includes:

```json
"versions": ["1"]
```

## HTTP Body Limits

`RuntimeConfig` now supports:

```text
--max-request-bytes <bytes>
MAX_REQUEST_BYTES=<bytes>
```

Default:

```text
67108864
```

The HTTP server enforces this limit while reading the request. Oversized
requests return:

```text
413 PAYLOAD_TOO_LARGE
```

Invalid `Content-Length` returns:

```text
400 INVALID_ARGUMENT
```

## Route And Error Behavior

Implemented stable behavior:

- Unknown routes return `404 NOT_FOUND`.
- Known model routes with wrong methods return `405 METHOD_NOT_ALLOWED`.
- Empty or malformed methods return `400 INVALID_ARGUMENT`.
- Unknown models and versions return `404 MODEL_NOT_FOUND`.
- Not-ready models return `409 MODEL_NOT_READY`.
- Malformed or unsupported inference requests return `400 INVALID_ARGUMENT`.

## Tests

Updated and added tests:

```text
tests/KServeRuntimeTest.cpp
tests/KServeV2CodecTest.cpp
tests/RuntimeConfigTest.cpp
tests/HttpIntegrationTest.cpp
```

Coverage includes:

- Metadata `versions`.
- Versioned metadata, readiness, and inference routes.
- Inference request `id` echo.
- Malformed JSON rejection.
- Unsupported shape, datatype, input, and output validation.
- Wrong-method `405` behavior.
- `MAX_REQUEST_BYTES` and `--max-request-bytes` parsing.
- Real-socket HTTP integration checks for server metadata, model metadata,
  wrong methods, inference, malformed JSON, unknown model, and oversized
  request handling.

## Validation

Validation commands run for this step:

```bash
cmake --preset debug
scripts/check-format.sh
cmake --build --preset debug
ctest --preset debug
docker build -f docker/Dockerfile -t neuriplo-kserve-runtime:step2 .
```

## Remaining Gaps

Deferred to later steps:

- Real tensor buffer conversion.
- Tensor data length validation.
- Executor abstraction and backend errors.
- Scheduler, queue limits, timeouts, and concurrency controls.
- Full Kubernetes end-to-end deployment validation.
