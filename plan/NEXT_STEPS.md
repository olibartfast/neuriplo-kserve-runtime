# Next Steps Roadmap

## Completed (Steps 0–11 + Extras)

- [x] Step 0–8: Core scaffold, protocol, scheduler, batching, LLM, observability
- [x] Step 9: Production hardening (multi-datatype, state machine, pipeline, benchmarks)
- [x] Step 10: LLM path completion (llama.cpp/Cactus, context, streaming, OpenAI endpoints)
- [x] Step 11: Deployment validation (InferenceGraph, canary, autoscaling, docs)
- [x] gRPC V2 surface (`GrpcServer`, `GrpcV2Codec`, 19 tests, CI preset)
- [x] Backend plugin registry (factory-per-backend, no central switch)
- [x] Docs reorganized into `plan/` directory

## E2E Validation Track (Priority: High)

### Step 12: Multi-Component YOLO Integration

**Goal**: Smoke-test the full neuriplo platform: `vision-inference` → KServe V2 →
`neuriplo-kserve-runtime` → `neuriplo` (ONNX Runtime) → YOLO model.

See `plan/E2E_YOLO.md` for the detailed plan.

| # | Item | Depends On |
|---|------|-----------|
| 12.1 | Verify YOLO task contract in vision-core (input shape, output format) | vision-core dev branch |
| 12.2 | Build runtime with `NEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO=ON` | neuriplo dev branch |
| 12.3 | Load real ONNX model, validate metadata from neuriplo | 12.2 |
| 12.4 | Run inference, compare output shapes against direct neuriplo | 12.3 |
| 12.5 | vision-inference KServe client (HTTP) → runtime → response | vision-inference dev branch |
| 12.6 | gRPC path parity test | 12.4 + gRPC build |
| 12.7 | `scripts/e2e-yolo.sh` automated smoke script | 12.5 |

## Production Track (Priority: Medium)

### Step 13: Control Plane / Data Plane Split

**Goal**: Separate model load/drain/reload (control) from infer hot path (data).
Required before multi-model hot reload.

| # | Item |
|---|------|
| 13.1 | Extract `ModelLifecycle` from `ModelRegistry` (load, unload, reload) |
| 13.2 | Make infer path independent of load path (no mutex contention) |
| 13.3 | Drain + reload without blocking in-flight inference |

### Step 14: Multi-Model Hot Reload

**Goal**: Load/unload/reload models without restarting the runtime.

| # | Item |
|---|------|
| 14.1 | Multi-model registry (map of name→handle) |
| 14.2 | Model add/remove at runtime via admin endpoint |
| 14.3 | Version-switch with zero-downtime drain |

## Optimization Track (Priority: Low / Scale-Triggered)

| Item | Trigger |
|------|---------|
| Async HTTP reactor (epoll/io_uring) | Connection concurrency > O(100) |
| Zero-copy tensor buffers | Copy overhead visible in profiling |
| Arena/PMR allocators | Allocation churn under load |
| Multi-datatype gRPC codec (INT8, FP16 in protobuf) | Client demand |

## Infrastructure Track

| Item |
|------|
| CI: build all sanitizers in parallel (currently sequential) |
| CI: gRPC + real-onnx combined preset |
| CI: E2E smoke test with stub model via curl |
| `.gitignore` entry for `.antigravitycli/` |
