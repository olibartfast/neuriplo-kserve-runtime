# Step 5: Scheduler And Autoscaling Behavior

This snapshot records the Step 5 implementation state. The runtime now routes
inference through a per-model scheduler with bounded admission, worker
execution, deadline handling, overload responses, drain-aware readiness, and
internal autoscaling metrics.

## Implemented Scope

### Step 5.1: Runtime Configuration

- Added CLI flags:
  - `--max-queue-size <count>`
  - `--request-timeout-ms <ms>`
  - `--instances <count>`
- Defaults preserve local stub behavior:
  - queue size `64`
  - timeout `30000` ms
  - instances `1`
- Config parser rejects zero/negative values for the new settings.
- README documents the scheduler flags.

### Step 5.2: Scheduler Abstraction

- Added `Scheduler` interface and `ModelScheduler` implementation.
- Executors remain behind the existing `Executor` interface.
- Scheduler returns structured `SchedulerResult` values mapped to HTTP errors.
- Direct executor unit tests remain unchanged.

### Step 5.3: Bounded FIFO Queue

- Per-model queue capacity is enforced by `max_queue_size`.
- Full queue returns `OVERLOADED`.
- FIFO order is preserved for accepted work.
- Drain rejects queued work with `UNAVAILABLE`.

### Step 5.4: Worker Execution

- Scheduler owns worker threads configured by `--instances`.
- Model registry creates one executor per instance.
- Dispatch selects the least-inflight executor instance.
- Scheduler destruction joins worker threads.

### Step 5.5: Timeout And Deadline Handling

- Request deadline is derived from `--request-timeout-ms`.
- Queued work expires before execution when the deadline passes.
- Slow execution times out when multiple instances require async execution.
- Submit waits on a per-request future instead of blocking HTTP threads forever.

### Step 5.6: HTTP Error Mapping And Readiness

- Overload maps to HTTP `429` with code `OVERLOADED`.
- Timeout maps to HTTP `504` with code `DEADLINE_EXCEEDED`.
- Drain maps to HTTP `503` with code `UNAVAILABLE`.
- `/v2/health/ready` and `/v2/models/{model}/ready` return not-ready while draining.
- Liveness remains independent from queue readiness.

### Step 5.7: Autoscaling Metrics Hooks

- Added `SchedulerMetricsSnapshot` and Prometheus-ready metric names in
  `SchedulerMetrics.hpp`:
  - queue depth
  - in-flight requests
  - accepted/rejected/timed-out request counts
  - queue, execution, and total latency totals
- Metrics are updated by scheduler tests and ready for Step 7 export.

## Request Flow

```text
KServeRuntime
  -> ModelRegistry
    -> ModelHandle
      -> Scheduler
        -> Executor
```

## Validation

```bash
cmake --preset debug
cmake --build --preset debug
scripts/check-format.sh
ctest --preset debug
cmake --preset tsan
cmake --build --preset tsan
ctest --preset tsan
```

## Remaining Work

- Dynamic batching remains Step 6 work.
- Prometheus `/metrics` export remains Step 7 work.
