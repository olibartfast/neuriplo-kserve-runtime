# Step 11: Deployment Validation - validation

> Retrospective packet. Step 11 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 11
> snapshot (`plan/STEP11.md`, later `specs/history/steps/STEP11.md`) and the
> Step 11 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Validation was recorded at completion rather than written before the code;
the checks below are the ones the step snapshot reports as run.

## Checks

- [x] [V-1] -> [R-2], [R-3], [R-7], [R-8]: the step's build, format and test commands (Recorded checks below) pass

## Traceability

| Requirement | Status | Evidence |
| --- | --- | --- |
| R-1 | Partial | substep 11.1 tests and exit criteria below |
| R-2 | Met | substep 11.1 tests and exit criteria below |
| R-3 | Met | substep 11.2 tests and exit criteria below |
| R-4 | Not met | substep 11.2 tests and exit criteria below |
| R-5 | Partial | substep 11.3 tests and exit criteria below |
| R-6 | Partial | substep 11.3 tests and exit criteria below |
| R-7 | Met | substep 11.4 tests and exit criteria below |
| R-8 | Met | substep 11.4 tests and exit criteria below |

## Deviations

Where the snapshot shows the step delivered less than the planned
exit criterion. These are recorded from the snapshot text, not re-tested.

- [R-1] Partial: proven by in-process response-shape tests; no InferenceGraph ran in a
   cluster.
- [R-4] Not met: no rollout run is recorded.
- [R-5] Partial: HPA and KEDA manifests and `scripts/load-test.sh` exist; no cluster
   scaling run is recorded.
- [R-6] Partial: same as R-5.

## Recorded checks

### 11.1 InferenceGraph Routing Validation - Exit Criteria Met

- Integration tests prove the runtime response shape is compatible with InferenceGraph
  Sequence, Switch, and Splitter routing.
- Error responses include `code` and `message` fields that do not break graph routing.
- Metadata and ready endpoints produce discoverable JSON shapes.

### 11.2 Canary Rollout Validation - Exit Criteria Met

- Canary traffic produces distinguishable `deployment` labels in metrics.
- Stable traffic has `version` label; canary traffic can be configured with separate version.
- Prometheus queries can differentiate canary vs stable:
  `neuriplo_http_infer_requests_total{deployment="canary"}` vs
  `neuriplo_http_infer_requests_total{deployment="stable"}`.

### 11.3 Autoscaling Integration Tests - Exit Criteria Met

- HPA and KEDA manifests exist and reference runtime metrics.
- Load test script validates metrics exposure and request flow.
- Autoscaling thresholds are documented.

### 11.4 Structured Error Documentation - Exit Criteria Met

- Client-facing error docs exist and cover every `KServeErrors` category.
- HTTP status codes, error body shapes, and recovery guidance are documented.
- Canary label queries for error monitoring are documented.

### Validation

```bash
cmake --build --preset debug && ctest --preset debug
# 100% tests passed, 0 tests failed
scripts/check-format.sh  # passes
```

### Exit Criteria

#### 11.1 InferenceGraph Routing Validation

- ✅ Integration tests prove response shapes survive Sequence routing (model_name,
  model_version, outputs present).
- ✅ Error responses use stable `code` and `message` fields compatible with graph routing.
- ✅ Metadata and ready endpoints produce discoverable JSON shapes.

#### 11.2 Canary Rollout Validation

- ✅ `version` and `deployment` labels are present in all Prometheus metrics.
- ✅ `--model-version` and `--deployment` CLI flags and environment variables work.
- ✅ Canary traffic produces `deployment="canary"` labels distinguishable from stable.
- ✅ `deployment` label is absent from metrics when not configured.

#### 11.3 Autoscaling Integration Tests

- ✅ HPA manifest references `neuriplo_scheduler_queue_depth` and
  `neuriplo_scheduler_in_flight_requests`.
- ✅ KEDA manifest references runtime metrics with appropriate thresholds.
- ✅ Load test script validates metrics exposure and request flow.
- ✅ Recommended autoscaling thresholds are documented.

#### 11.4 Structured Error Documentation

- ✅ `docs/errors.md` documents every `KServeErrors` category.
- ✅ HTTP status codes, error body shapes, and recovery guidance are included.
- ✅ Prometheus queries for error monitoring by canary/stable deployment are documented.
- ✅ InferenceGraph compatibility notes per error code are documented.

## Evidence

| Date | Check | Result |
| --- | --- | --- |
| 2026-06-04 | Recorded checks above | Pass, as reported by the Step 11 snapshot |
