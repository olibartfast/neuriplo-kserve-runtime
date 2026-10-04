# Step 6: Dynamic Batching - plan

> Retrospective packet. Step 6 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 6
> snapshot (`plan/STEP6.md`, later `specs/history/steps/STEP6.md`) and the
> Step 6 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Tasks are the step's substeps. Each `T-` section is the implementation
record from the snapshot.

### [T-1] Step 6.1: Runtime Configuration

- Added CLI flags:
  - `--dynamic-batching-enabled true|false`
  - `--max-batch-size <count>`
  - `--max-queue-delay-us <microseconds>`
  - `--preferred-batch-sizes <csv>` (optional, e.g. `2,4,8`)
- Defaults preserve Step 5 single-request behavior:
  - dynamic batching disabled
  - max batch size `1`
  - max queue delay `0`
- Config parser rejects zero `max-batch-size`, negative queue delay, and
  `max-batch-size < 2` when dynamic batching is enabled.
- README documents the dynamic batching flags.

### [T-2] Step 6.2: Batch Compatibility Model

- Added `BatchCompatibility.hpp/cpp` with deterministic compatibility checks:
  - same input count
  - same requested outputs
  - same input names, datatypes, and rank
  - same shape except batch dimension (dimension 0)
- Incompatible queued neighbors are skipped; they never enter the same batch.
- Focused unit tests cover accept and reject paths.

### [T-3] Step 6.3: Dynamic Batcher Abstraction

- Added `DynamicBatcher.hpp/cpp` for merge/split logic.
- Executor details remain behind the existing `Executor` interface.
- Compatible requests merge into one batched `ExecutionRequest`.
- Per-request deadlines and cancellation behavior from Step 5 are preserved.
- Batching disabled uses the existing single-request scheduler path.

### [T-4] Step 6.4: Batch Formation And Dispatch

- `ModelScheduler::formBatch()` waits up to `max_queue_delay_us` for compatible
  work after dequeuing the first request.
- Batch growth stops at `max_batch_size`.
- `preferred_batch_sizes` is honored when configured.
- One batched executor call dispatches through least-inflight instance selection.

### [T-5] Step 6.5: Output Splitting

- `splitExecutionResponse()` slices batched executor outputs back into
  per-request responses.
- Batched outputs preserve KServe tensor name, datatype, and per-request shape.
- Backend failures replicate to all source requests in the batch.

### [T-6] Step 6.6: Timeout, Overload, And Drain Interaction

- Batch formation stops when a queued neighbor exceeds its deadline.
- Step 5 overload (`429`), timeout (`504`), and drain (`503`) semantics remain
  unchanged.
- Expired batch candidates are fulfilled with timeout results before dispatch.

### [T-7] Step 6.7: Batch Metrics Hooks

- Extended `SchedulerMetricsSnapshot` and Prometheus-ready metric names in
  `SchedulerMetrics.hpp`:
  - batches formed
  - batched request count
  - batch formation latency total
  - batch execution latency total
- Scheduler tests verify batch metric updates; Step 7 owns `/metrics` export.

## Request Flow

```text
KServeRuntime
  -> ModelRegistry
    -> ModelHandle
      -> Scheduler
        -> DynamicBatcher
          -> Executor
```

Dynamic batching applies only to compatible tensor models. LLM/token scheduling
remains Step 8 work and must not share the tensor batching path.
