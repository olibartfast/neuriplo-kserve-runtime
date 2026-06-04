# neuriplo-kserve-runtime Error Reference

This document describes the complete error taxonomy used by neuriplo-kserve-runtime.
All errors follow a consistent JSON shape with `code` and `message` fields.

## Error Response Shape

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable description of what went wrong"
  }
}
```

## Error Codes and HTTP Status Mapping

| Error Code | HTTP Status | Description |
|---|---|---|
| `INVALID_ARGUMENT` | 400 | Request body is malformed, contains invalid JSON, or violates protocol constraints |
| `MODEL_NOT_FOUND` | 404 | The requested model name does not exist in the registry |
| `NOT_FOUND` | 404 | The requested route does not exist |
| `METHOD_NOT_ALLOWED` | 405 | The HTTP method is not supported for the requested route |
| `PAYLOAD_TOO_LARGE` | 413 | The request body exceeds `--max-request-bytes` |
| `MODEL_NOT_READY` | 409 | The model exists but is not in the READY state |
| `QUEUE_FULL` | 429 | The scheduler queue is at capacity; the request was rejected immediately |
| `UNAVAILABLE` | 503 | The service or model is unavailable (draining, loading, or failed state) |
| `DEADLINE_EXCEEDED` | 504 | The request exceeded its timeout budget while waiting in the queue |
| `BACKEND_ERROR` | 500 | The backend executor failed during inference |
| `INTERNAL` | 500 | An unexpected internal error occurred |

Any unrecognized error code falls back to HTTP 500.

## Detailed Error Scenarios

### INVALID_ARGUMENT (400)

Triggered when:
- Request body is not valid JSON
- Required fields are missing from the request (e.g., no `inputs` array)
- Tensor shape is incompatible with the model metadata (e.g., wrong dimensions)
- Tensor datatype is not supported (e.g., `INT32` when only `FP32` is accepted)
- Requested output name does not match any model output
- OpenAI endpoint requests are missing required fields (`model`, `prompt`, or `messages`)
- Context length or token budget is exceeded for LLM models

Example response:
```json
{
  "error": {
    "code": "INVALID_ARGUMENT",
    "message": "invalid shape for input 'input': expected [1,3,224,224] got [1,3]"
  }
}
```

Recovery: Fix the request payload and retry.

### MODEL_NOT_FOUND (404)

Triggered when:
- The model name in the URL path does not match any registered model
- The model version does not exist

Example response:
```json
{
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "model not found: yolo-v5"
  }
}
```

Recovery: Verify the model name in the URL path matches a loaded model. Check
`GET /v2/models/{name}` and `GET /v2` for registered models.

### NOT_FOUND (404)

Triggered when:
- The URL path does not match any known route

Recovery: Check the URL path. Supported routes are documented in the KServe V2
protocol and the OpenAI-compatible endpoint paths.

### METHOD_NOT_ALLOWED (405)

Triggered when:
- Using `POST` on a read-only endpoint (e.g., `POST /v2/models/demo`)
- Using `GET` on a write-only endpoint (e.g., `GET /v2/models/demo/infer`)

Recovery: Use the correct HTTP method for the endpoint.

### PAYLOAD_TOO_LARGE (413)

Triggered when:
- The request `Content-Length` exceeds `--max-request-bytes` (default: 64 MiB)
- The actual request body is larger than the declared content length

Recovery: Reduce the request payload size or increase `--max-request-bytes`.

### MODEL_NOT_READY (409)

Triggered when:
- The model is registered but not in the `READY` state
- This typically means the model is in a transient state that is not `LOADING`,
  `FAILED`, `UNLOADED`, or `UNAVAILABLE` (those map to `UNAVAILABLE`)

Example response:
```json
{
  "error": {
    "code": "MODEL_NOT_READY",
    "message": "model is not ready: demo"
  }
}
```

Recovery: Wait for the model to finish loading and retry. Use
`GET /v2/models/{name}/ready` to poll readiness.

### QUEUE_FULL (429)

Triggered when:
- The scheduler queue has reached `--max-queue-size` and cannot accept more
  requests
- This indicates the runtime is overloaded and new requests should be rejected
  rather than queued

Example response:
```json
{
  "error": {
    "code": "QUEUE_FULL",
    "message": "request queue is full"
  }
}
```

Recovery: Implement exponential backoff and retry. If persistent, scale up
the runtime instances or increase `--max-queue-size`.

### UNAVAILABLE (503)

Triggered when:
- The runtime is not ready (`GET /v2/health/ready` returns 503)
- A model is in `LOADING`, `FAILED`, `UNLOADED`, or `UNAVAILABLE` state
- The model is draining (e.g., during graceful shutdown)
- The scheduler is unavailable for the requested model
- Memory pressure is preventing new LLM decode requests

Example responses:
```json
{"error": {"code": "UNAVAILABLE", "message": "runtime is not ready"}}
{"error": {"code": "UNAVAILABLE", "message": "model is still loading"}}
{"error": {"code": "UNAVAILABLE", "message": "model has been unloaded"}}
{"error": {"code": "UNAVAILABLE", "message": "model is draining"}}
```

Recovery: Wait and retry. For draining, the service is intentionally shutting
down and new requests should be routed to other instances.

### DEADLINE_EXCEEDED (504)

Triggered when:
- A request waited in the scheduler queue longer than `--request-timeout-ms`
  and was abandoned before reaching an executor

Example response:
```json
{
  "error": {
    "code": "DEADLINE_EXCEEDED",
    "message": "request exceeded deadline"
  }
}
```

Recovery: Reduce server load, increase `--request-timeout-ms`, or increase
`--instances` to handle more concurrent requests.

### BACKEND_ERROR (500)

Triggered when:
- The backend executor returned an error during inference
- This could be caused by invalid input data, model corruption, or backend
  library failures

Example response:
```json
{
  "error": {
    "code": "BACKEND_ERROR",
    "message": "backend inference failed"
  }
}
```

Recovery: Check backend logs. Verify the model artifact is valid and the
backend is correctly configured.

### INTERNAL (500)

Triggered when:
- An unexpected internal error occurred
- Used as a catch-all for errors that do not map to a more specific code

Recovery: Check runtime logs for stack traces or unexpected conditions. This
typically indicates a bug in the runtime.

## KServe V2 Route Reference

| Method | Path | Success | Error Codes |
|---|---|---|---|
| GET | `/v2` | 200 | |
| GET | `/v2/health/live` | 200 | |
| GET | `/v2/health/ready` | 200/503 | `UNAVAILABLE` |
| GET | `/v2/models/{name}` | 200/404 | `MODEL_NOT_FOUND` |
| GET | `/v2/models/{name}/ready` | 200/404/409/503 | `MODEL_NOT_FOUND`, `MODEL_NOT_READY`, `UNAVAILABLE` |
| POST | `/v2/models/{name}/infer` | 200/400/404/409/429/500/503/504 | All error codes |
| GET | `/v2/models/{name}/versions/{v}` | 200/404 | `MODEL_NOT_FOUND` |
| GET | `/v2/models/{name}/versions/{v}/ready` | 200/404/409/503 | `MODEL_NOT_FOUND`, `MODEL_NOT_READY`, `UNAVAILABLE` |
| POST | `/v2/models/{name}/versions/{v}/infer` | 200/400/404/409/429/500/503/504 | All error codes |
| POST | `/v1/completions` | 200/400/404/500/503 | `INVALID_ARGUMENT`, `MODEL_NOT_FOUND`, `BACKEND_ERROR`, `UNAVAILABLE` |
| POST | `/v1/chat/completions` | 200/400/404/500/503 | `INVALID_ARGUMENT`, `MODEL_NOT_FOUND`, `BACKEND_ERROR`, `UNAVAILABLE` |
| POST | `/v1/embeddings` | 200/400/404/500/503 | `INVALID_ARGUMENT`, `MODEL_NOT_FOUND`, `BACKEND_ERROR`, `UNAVAILABLE` |
| GET | `/metrics` | 200 | |

## Prometheus Metrics for Error Monitoring

The following metrics can be used to monitor error rates:

- `neuriplo_http_requests_total{model, version, status}` — total requests by
  HTTP status code. Use `status="429"` for queue full rates and
  `status="503"` for availability issues.
- `neuriplo_http_infer_requests_failure_total{model, version}` — total failed
  infer requests regardless of status code.
- `neuriplo_backend_errors_total{model, version}` — total backend execution
  failures.
- `neuriplo_scheduler_requests_rejected_total{model, version}` — total requests
  rejected at admission (queue full).
- `neuriplo_scheduler_requests_timed_out_total{model, version}` — total requests
  that exceeded their deadline.
- `neuriplo_scheduler_requests_memory_pressure_rejected_total{model, version}` —
  total LLM requests rejected due to KV-cache memory pressure.

## Canary and Deployment Labels

Metrics include `version` and `deployment` labels for canary rollout validation:

- `version`: model version string (default: `"1"`, configurable via
  `--model-version` or `MODEL_VERSION` env var)
- `deployment`: deployment identifier (optional, configurable via `--deployment`
  or `DEPLOYMENT` env var; typically `"stable"` or `"canary"`)

Use these labels in Prometheus queries to distinguish canary vs stable traffic:

```promql
# Canary error rate
sum(rate(neuriplo_http_infer_requests_failure_total{deployment="canary"}[5m]))
/
sum(rate(neuriplo_http_infer_requests_total{deployment="canary"}[5m]))

# Stable vs canary latency
histogram_quantile(0.95, sum(rate(neuriplo_scheduler_total_latency_seconds_bucket{deployment="stable"}[5m])) by (le))
```

## KServe InferenceGraph Compatibility

The runtime's error response shape is compatible with KServe InferenceGraph
routing. All error responses conform to the `{"error": {"code": ..., "message": ...}}`
shape, which allows graph routers to inspect error codes for retry or fallback
decisions.

Key compatibility notes:
- `MODEL_NOT_FOUND` (404) indicates the model is not available for routing
- `UNAVAILABLE` (503) signals that the graph should route to alternative nodes
- `QUEUE_FULL` (429) indicates temporary overload; the graph may retry
- `DEADLINE_EXCEEDED` (504) indicates the node could not process in time; the
  graph may route to other nodes or return a timeout to the client
- All versioned routes (`/v2/models/{name}/versions/{v}/...`) return the same
  error codes, enabling Switch and Splitter routing based on model version