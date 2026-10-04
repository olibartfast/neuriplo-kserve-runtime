# Step 7: Observability And KServe Deployment Examples - plan

> Retrospective packet. Step 7 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 7
> snapshot (`plan/STEP7.md`, later `specs/history/steps/STEP7.md`) and the
> Step 7 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Tasks are the step's substeps. Each `T-` section is the implementation
record from the snapshot.

### [T-1] Step 7.1: Prometheus `/metrics` Export

- Registered `GET /metrics` route alongside existing KServe V2 routes.
- Implemented a thread-safe telemetry registry providing Prometheus text-format export.
- Scheduler metrics (queue depth, in-flight, accepted, rejected, timed-out, latency totals, batch formation totals) are dynamically queried from the scheduler and populated on scrape.
- Dynamically label all model metrics with the actual `model` name.
- Implemented cumulative latency histograms (`queue`, `infer`, `total`) and batch size histograms using custom Prometheus buckets.
- Added detailed request counts with `model`, `method`, and `status` labels.
- Added process resident memory (`VmRSS`) reading from `/proc/self/status`.
- Added dynamic model load success/failure counters and backend execution error counters.

### [T-2] Step 7.2: Structured Logging

- Implemented structured logging helper emitting JSON lines to `stderr`.
- Concurrency Hardening: Replaced thread-unsafe `std::gmtime` with POSIX-compliant thread-safe `gmtime_r` to prevent data races and crashes in concurrent environments.

### [T-3] Step 7.3: Optional Payload Logging Controls

- CLI flag `--log-payloads` and `RuntimeConfig` option integrated with `KServeRuntime`.
- Raw JSON request and response body logging is safely gated behind `--log-payloads true`.
- By default, only byte counts and high-level request parameters are logged, preventing secret leakages.

### [T-4] Step 7.4: Request-Path Trace Spans

- Implemented lightweight trace spans following request ID across stages:
  - `admit`: request parsed and validated (includes payload if enabled)
  - `queue`: request submitted to scheduler
  - `batch form`: dynamic batch formed in scheduler
  - `infer`: batch dispatched to backend executor
  - `split`: response sliced back to requests
  - `respond`: final HTTP completion response returned (includes payload if enabled)
- Request ID is fully propagated across all scheduler/batching phases, enabling end-to-end trace tracking.

### [T-5] Step 7.5: Error Taxonomy Stabilization

- Aligned error categories in `src/KServeErrors.hpp`: `INVALID_ARGUMENT`, `MODEL_NOT_FOUND`, `MODEL_NOT_READY`, `QUEUE_FULL`, `DEADLINE_EXCEEDED`, `BACKEND_ERROR`, `UNAVAILABLE`, `INTERNAL`, `METHOD_NOT_ALLOWED`, `PAYLOAD_TOO_LARGE`.
- Built robust HTTP status code mapper (400, 404, 409, 429, 504, 500, 503) applied across routing and scheduling boundaries.

### [T-6] Step 7.7: Canary Rollout Example

- Added `deploy/kserve/canary-inferenceservice.yaml`.
- Configured 30% traffic split (`canaryTrafficPercent: 30`) targeting canary serving runtime, documenting image and model artifact rollout validation.

### [T-7] Step 7.8: InferenceGraph Example

- Added `deploy/kserve/inferencegraph.yaml`.
- Configured a `Sequence` router directing weights between multiple predictor models without custom response adapters.

### [T-8] Step 7.9: Deployment Documentation

- Added comprehensive sections to `README.md` covering image build, Kubernetes service apply, liveness/readiness expectations, curl examples, and Prometheus scrape configuration.

## Request Flow

```text
KServeRuntime (http request admit)
  -> ModelRegistry
    -> ModelHandle
      -> Scheduler (+ trace spans, metrics)
        -> DynamicBatcher (when enabled)
          -> Executor
  -> http response encode + structured logs (respond span)
GET /metrics -> MetricsRegistry -> Prometheus text (scheduler stats queried on-scrape)
```
