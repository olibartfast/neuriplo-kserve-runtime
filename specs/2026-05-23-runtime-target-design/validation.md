# Runtime Target Design - validation

> Retrospective packet. This is the original milestone target design for
> `neuriplo-kserve-runtime` (`plan/ROADMAP.md`, later
> `specs/history/target-design.md`), ported on 2026-10-05. It framed Steps 0-11;
> each step's scope and exit criteria now live in that step's packet. The live
> plan is [`../roadmap.md`](../roadmap.md); current patterns are in
> [`../architecture.md`](../architecture.md).

## Checks

- [x] [V-1] -> [R-1], [R-2]: Steps 0-2 and 7 packets (protocol routes, KServe
  manifests, probes).
- [x] [V-2] -> [R-3]: Steps 3, 4 and 9 packets (executor boundary, real
  `neuriplo`, multi-datatype adapter), Step 12 YOLO e2e.
- [x] [V-3] -> [R-4]: Steps 8 and 10 packets (LLM scheduler, decode path).
- [x] [V-4] -> [R-5]: Steps 5, 6, 7, 9, 11 packets.
- [x] [V-5] -> [R-6]: one packet per step, linked from [`plan.md`](plan.md).

## Validation Strategy

Local checks first:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Protocol checks:

```bash
curl -s http://127.0.0.1:8080/v2
curl -s http://127.0.0.1:8080/v2/health/live
curl -s http://127.0.0.1:8080/v2/health/ready
curl -s http://127.0.0.1:8080/v2/models/demo
```

Integration checks:

```text
direct neuriplo backend output
  vs
neuriplo-kserve-runtime output
```

Downstream checks:

```text
vision-inference -> KServe client -> runtime -> neuriplo
```

Performance checks:

- p50/p95/p99 latency.
- queue latency.
- throughput under concurrency.
- batching throughput improvement.
- memory stability under repeated requests.

## Production Readiness Gap Analysis

Steps 0–8 are completed and have snapshot documents (now the Step 0-8 packets, listed in [`plan.md`](plan.md#step-index)). Steps
9–11 are planned to close remaining gaps. This section records what remains
before this is a production-facing serving runtime, annotated with the step
that addresses each gap.

### Gaps Blocking Tensor / CV Production Use → Addressed by Step 9

- **Dense FP32-only neuriplo adapter** → Step 9.1
- **No formal model state machine** → Step 9.2
- **Request pipeline not composable** → Step 9.3
- **Request ID generation is weak** → Step 9.4
- **No container readiness-gate health checks** → Step 9.5
- **No latency SLO / performance baseline** → Step 9.6
- **Thread-per-client HTTP server** → Explicitly deferred (scale trigger)
- **gRPC V2 surface** → ✅ Implemented (see proto/kserve_grpc.proto, src/GrpcServer.*)

### Gaps Blocking LLM Production Use → Addressed by Step 10

- **No real LLM backend execution** → Step 10.1
- **Token-accurate context enforcement** → Step 10.2
- **No cancellation/deadline in backend decode** → Step 10.3
- **No KV-cache memory pressure policy** → Step 10.4
- **No streaming responses** → Step 10.5
- **No OpenAI-compatible endpoints** → Step 10.6

### Gaps Common To Both Paths → Addressed by Step 11

- **No InferenceGraph routing validation** → Step 11.1
- **No canary rollout validation** → Step 11.2
- **No autoscaling integration tests** → Step 11.3
- **No structured error documentation** → Step 11.4

### Explicitly Deferred (Long-Term / Scale-Only)

These are recorded in the Design Pattern Evolution section and are NOT blockers
for a first production launch. They remain deferred even after Step 11 completion.

| Item | Trigger |
|------|---------|
| Async HTTP reactor (epoll/io_uring) | High connection concurrency |
| Zero-copy / borrowed tensor buffers | Copy overhead in hot path |
| Arena / PMR allocators | Allocation churn under load |
| gRPC V2 surface | ✅ Implemented |
| Multi-model hot reload | Deployment need |
| Backend plugin registry (no central switch) | Backend count growth |
| Control plane / data plane split | Multi-model + drain complexity |

## Post-Step 11 Launch Readiness

After Steps 9, 10, and 11 are completed, the runtime is production-ready for a
**first launch** under these constraints:

- Single model per runtime instance (tensor or LLM).
- KServe V2 HTTP protocol surface.
- Moderate connection concurrency (thread-per-client is sufficient).
- For LLM: streaming, cancellation, and token-accurate context limits are in
  place.
- Kubernetes probes, autoscaling, canary, and InferenceGraph routing are
  validated.

The explicitly deferred items (async HTTP reactor, gRPC, multi-model, zero-copy
buffers) become relevant only when the deployment outgrows single-model or
moderate-concurrency workloads. They are not correctness blockers for launch.

### Recommended Next Actions

1. **Step 9 (Tensor hardening)**: Start with 9.1 (multi-datatype adapter) and
   9.2 (model state machine) as they are the highest-impact correctness items.
2. **Step 10 (LLM completion)**: Start with 10.1 (real backend execution) and
   10.3 (cancellation propagation) as they gate all other LLM substeps.
3. **Step 11 (Deployment validation)**: Can run in parallel with Step 9 and 10;
   requires a test Kubernetes cluster.

## Evidence

| Date | Check | Result |
| --- | --- | --- |
| 2026-06-05 | Steps 0-11 complete | Pass; see each step packet's evidence |
