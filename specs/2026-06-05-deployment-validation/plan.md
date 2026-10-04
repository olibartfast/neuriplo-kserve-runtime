# Step 11: Deployment Validation - plan

> Retrospective packet. Step 11 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 11
> snapshot (`plan/STEP11.md`, later `specs/history/steps/STEP11.md`) and the
> Step 11 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Tasks are the step's substeps. Each `T-` section is the implementation
record from the snapshot.

## Summary

Completed deployment validation: InferenceGraph routing compatibility tests, canary rollout
metrics labels, autoscaling manifests and load testing, and comprehensive error documentation.

## [T-1] 11.1 InferenceGraph Routing Validation

### Changes

**`tests/DeploymentValidationTest.cpp`** — new test file covering:
- InferenceGraph-compatible response shapes: all KServe V2 responses include `model_name`,
  `model_version`, and `outputs` fields required for graph routing.
- Error response stability: all error codes produce `{"error": {"code": "...", "message": "..."}}`
  shapes that survive graph routing (Sequence, Switch, Splitter).
- Versioned route compatibility: `/v2/models/{name}/versions/{v}/infer` produces responses
  with the correct `model_version` for Switch routing.
- Health endpoint response shapes for Splitter health checks.
- Server metadata for service discovery in graph contexts.
- Queue-full (429) error behavior under load for graph retry decisions.
- Complete HTTP status code mapping verification for all error codes.

## [T-2] 11.2 Canary Rollout Validation

### Changes

**`src/MetricsRegistry.hpp`** — added `setModelVersion()` and `setDeployment()` methods,
  `model_version_` and `deployment_` fields.

**`src/MetricsRegistry.cpp`** — all Prometheus metrics now include `version` label alongside
  `model`. `deployment` label is included when set (non-empty). `neuriplo_http_requests_total`
  includes `model`, `version`, `status`, and optional `deployment`.

**`src/RuntimeConfig.hpp`** — added `model_version` (default `"1"`) and `deployment` fields.

**`src/RuntimeConfig.cpp`** — added `--model-version` and `--deployment` CLI flags,
  `MODEL_VERSION` and `DEPLOYMENT` environment variable defaults.

**`src/main.cpp`** — calls `metrics.setModelVersion()` and `metrics.setDeployment()` at startup.

**`deploy/kserve/cluster-serving-runtime.yaml`** — added readiness, liveness, and startup probes
  for Kubernetes probe integration, and `DEPLOYMENT` env from pod labels.

**`deploy/kserve/canary-inferenceservice.yaml`** — added `DEPLOYMENT=canary` environment variable
  to distinguish canary traffic in metrics.

**`tests/DeploymentValidationTest.cpp`** — added canary label tests:
- `canary_metrics_include_version_label` — verifies `version="2"` appears in metrics output.
- `canary_metrics_include_deployment_label` — verifies `deployment="canary"` appears in metrics.
- `canary_metrics_default_deployment_label_absent` — verifies deployment label is absent when
  not configured.
- `autoscaling_metrics_include_queue_and_inflight` — verifies HPA-relevant metrics are present.

**`tests/RuntimeConfigTest.cpp`** — added tests for `--model-version`, `--deployment`,
  `MODEL_VERSION`, and `DEPLOYMENT` environment variables.

## [T-3] 11.3 Autoscaling Integration Tests

### Changes

**`deploy/kserve/hpa.yaml`** — new HorizontalPodAutoscaler manifest targeting:
  - `neuriplo_scheduler_queue_depth` average value of 5
  - `neuriplo_scheduler_in_flight_requests` average value of 8
  - CPU utilization 70%
  - Scale-down stabilization 300s, scale-up stabilization 60s
  - Min replicas 1, max replicas 10

**`deploy/kserve/keda-scaledobject.yaml`** — new KEDA ScaledObject targeting:
  - `neuriplo_scheduler_queue_depth` Prometheus trigger
  - `neuriplo_scheduler_in_flight_requests` Prometheus trigger
  - `neuriplo_scheduler_total_latency_seconds` average latency trigger
  - Polling interval 10s, cooldown period 300s

**`scripts/load-test.sh`** — new load test script that:
  - Validates liveness and readiness endpoints
  - Validates model metadata endpoint
  - Sends concurrent infer requests for a configurable duration
  - Reports total/success/fail request counts and success rate
  - Validates all required autoscaling metrics are present in `/metrics`
  - Checks `version` and `deployment` labels in metrics output

### Recommended Autoscaling Thresholds

| Metric | Target | Notes |
|---|---|---|
| `neuriplo_scheduler_queue_depth` | average ≤ 5 | Scale up when queue builds |
| `neuriplo_scheduler_in_flight_requests` | average ≤ 8 | Scale up when concurrency saturates |
| `neuriplo_scheduler_total_latency_seconds` | p95 ≤ 2s | Scale up when latency degrades |
| CPU utilization | ≤ 70% | General compute pressure |
| `neuriplo_process_resident_memory_bytes` | monitoring only | Alert on memory growth trends |

## [T-4] 11.4 Structured Error Documentation

### Changes

**`docs/errors.md`** — comprehensive error reference covering:
- Standard error response shape: `{"error": {"code": "...", "message": "..."}}`
- Complete error code table with HTTP status mapping
- Detailed error scenarios for each code with example responses and recovery guidance
- KServe V2 route reference with success/error status codes per endpoint
- OpenAI endpoint route reference
- Prometheus metrics for error monitoring (with version/deployment label queries)
- Canary and deployment label query examples
- InferenceGraph compatibility notes for all error codes

## Files Changed

| File | Change |
|---|---|
| `src/MetricsRegistry.hpp` | Added `setModelVersion()`, `setDeployment()`, `model_version_`, `deployment_` fields |
| `src/MetricsRegistry.cpp` | `version` and `deployment` labels in all metrics; implementations of new methods |
| `src/RuntimeConfig.hpp` | Added `model_version` and `deployment` fields |
| `src/RuntimeConfig.cpp` | Added `--model-version`, `--deployment` CLI flags; `MODEL_VERSION`, `DEPLOYMENT` env vars |
| `src/main.cpp` | Calls `metrics.setModelVersion()` and `metrics.setDeployment()` |
| `tests/DeploymentValidationTest.cpp` | New file — InferenceGraph, canary, autoscaling, error taxonomy tests |
| `tests/KServeRuntimeTest.cpp` | Updated metrics assertions for `version` label |
| `tests/RuntimeConfigTest.cpp` | Added tests for `--model-version`, `--deployment`, `MODEL_VERSION`, `DEPLOYMENT` |
| `deploy/kserve/cluster-serving-runtime.yaml` | Added readiness, liveness, startup probes; `DEPLOYMENT` env |
| `deploy/kserve/canary-inferenceservice.yaml` | Added `DEPLOYMENT=canary` environment variable |
| `deploy/kserve/inferenceservice.yaml` | Unchanged (kept clean) |
| `deploy/kserve/inferencegraph.yaml` | Unchanged (kept clean) |
| `deploy/kserve/hpa.yaml` | New file — HorizontalPodAutoscaler manifest |
| `deploy/kserve/keda-scaledobject.yaml` | New file — KEDA ScaledObject manifest |
| `scripts/load-test.sh` | New file — load test and metrics validation script |
| `docs/errors.md` | New file — structured error documentation |
| `CMakeLists.txt` | Added `tests/DeploymentValidationTest.cpp` |
