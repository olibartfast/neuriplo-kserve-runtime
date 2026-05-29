# Step 6 WIP: Dynamic Batching

This work-in-progress plan tracks Step 6 implementation. Step 5 established
per-model scheduling with bounded admission, worker execution, deadlines,
overload behavior, drain-aware readiness, and internal autoscaling metrics;
Step 6 adds compatible-request grouping, bounded batch formation, batched
execution, and output splitting for tensor models.

## Goal

Extend the scheduler so compatible queued requests can be combined into a single
executor call:

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

## Step 6.1: Runtime Configuration

- Add CLI flags:
  - `--dynamic-batching-enabled true|false`
  - `--max-batch-size <count>`
  - `--max-queue-delay-us <microseconds>`
  - `--preferred-batch-sizes <csv>` (optional, e.g. `2,4,8`)
- Define conservative defaults that preserve current single-request behavior:
  - dynamic batching disabled
  - max batch size `1`
  - max queue delay `0`
- Validate invalid values during config parsing.
- Document the flags in `README.md`.

Exit criteria:

- Config parser accepts valid values.
- Config parser rejects zero/negative values where inappropriate.
- Default launch behavior remains compatible with Step 5.

## Step 6.2: Batch Compatibility Model

- Define compatibility rules for requests targeting the same model/version:
  - same input count
  - same tensor names
  - same datatype
  - same shape except batch dimension
  - same relevant request parameters / requested outputs policy
- Add a small compatibility helper used by the batcher and unit tests.
- Reject or bypass batching for incompatible queued neighbors.

Exit criteria:

- Compatibility checks are deterministic and covered by focused unit tests.
- Incompatible requests never enter the same batch.

## Step 6.3: Dynamic Batcher Abstraction

- Add a batching layer inside the scheduler path.
- Keep executor details behind the existing `Executor` interface.
- Merge compatible requests into one batched `ExecutionRequest`.
- Preserve per-request deadlines and cancellation behavior from Step 5.

Exit criteria:

- Scheduler can operate with batching disabled using the existing single-request
  path.
- Batching logic is testable without HTTP or registry dependencies.

## Step 6.4: Batch Formation And Dispatch

- After dequeuing the first compatible request, wait up to
  `max_queue_delay_us` for additional compatible work.
- Stop growing the batch at `max_batch_size`.
- Honor `preferred_batch_sizes` when configured.
- Dispatch one batched executor call through least-inflight instance selection.

Exit criteria:

- Multiple compatible requests can complete in one executor invocation.
- Batch size never exceeds configured limits.
- Queue delay budget is enforced.

## Step 6.5: Output Splitting

- Split batched executor outputs back into per-request responses.
- Preserve KServe response shape equivalence with single-request execution.
- Map backend failures to stable per-request errors.

Exit criteria:

- Batched outputs match single-request schema/shape for the same inputs.
- Each request receives its own response payload and request id.

## Step 6.6: Timeout, Overload, And Drain Interaction

- Do not grow batches beyond a request's remaining deadline.
- Keep overload (`429`) and drain (`503`) semantics from Step 5.
- Expire queued batch candidates that cannot be served in time.

Exit criteria:

- Batching does not weaken Step 5 admission or deadline guarantees.
- Drain still rejects new work and flips readiness false.

## Step 6.7: Batch Metrics Hooks

- Extend scheduler metrics state for Step 7 export:
  - formed batch size
  - batch count
  - batch formation latency
  - batch execution latency
- Keep metrics lightweight if `/metrics` export remains Step 7 work.

Exit criteria:

- Scheduler updates batch metric state in tests.
- Metric names and labels are ready to expose in Step 7.

## Step 6.8: Validation

Expected validation commands:

```bash
cmake --preset debug
cmake --build --preset debug
scripts/check-format.sh
ctest --preset debug
```

Additional recommended validation for batching/threading changes:

```bash
cmake --preset tsan
cmake --build --preset tsan
ctest --preset tsan
```

Golden comparisons:

- batched vs single-request stub output
- real backend smoke path when batch size > 1 is enabled

## Completion Criteria

Step 6 is complete when:

- Compatible requests batch deterministically under configured limits.
- Batched outputs are shape/schema equivalent to single-request outputs.
- Batch metrics state is available for observability integration.
- Step 5 overload, timeout, and drain behavior remain intact.
- A completed `STEP6.md` snapshot replaces this WIP document.

## Out Of Scope

- LLM/token-aware scheduling (`llama.cpp`, Cactus, GGUF) — Step 8.
- Prometheus `/metrics` export — Step 7.
- Multi-model hot reload or version-specific batch policies beyond current
  single-model runtime.
