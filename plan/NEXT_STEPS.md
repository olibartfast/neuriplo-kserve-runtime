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

**Goal**: Smoke-test the full neuriplo platform: `neuriplo-infer` → KServe V2 →
`neuriplo-kserve-runtime` → `neuriplo` (ONNX Runtime) → YOLO model.

See `plan/E2E_YOLO.md` for the detailed plan and `plan/STEP12.md` for the snapshot.

| # | Item | Status |
|---|------|--------|
| 12.2 | Build runtime with `NEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO=ON` | ✅ Done |
| 12.3 | Load real ONNX model, validate metadata from neuriplo | ✅ Done |
| 12.4 | Run inference, compare output shapes | ✅ Done |
| 12.7 | `scripts/e2e-yolo.sh` automated smoke script | ✅ Done |
| 12.1 | Verify YOLO task contract in neuriplo-tasks | ✅ Done: yolo26 → YOLO_NMS_FREE contract ([batch,dets,6], no NMS) matches yolo26s.onnx metadata exactly |
| 12.5 | neuriplo-infer KServe client (HTTP) → runtime → response | ✅ Done: real neuriplo-infer→KServe HTTP→runtime→neuriplo→ONNX Runtime→YOLO E2E verified |
| 12.6 | gRPC path parity test | ✅ Done: `real-onnx-grpc` preset, `grpc_real_neuriplo_yolo_infer`, `e2e-yolo.sh` check 8 |

Step 12 is complete. Next work starts at Step 13.

## Production Track (Priority: Medium)

### Step 13: Control Plane / Data Plane Split

**Goal**: Separate model load/drain/reload (control) from infer hot path (data).
Required before multi-model hot reload.

| # | Item | Status |
|---|------|--------|
| 13.1 | Extract `ModelLifecycle` from `ModelRegistry` (load, unload, reload) | ✅ Done |
| 13.2 | Make infer path independent of load path (no mutex contention) | ✅ Done |
| 13.3 | Drain + reload without blocking in-flight inference | ✅ Done: `SchedulerRetireQueue` background drain |

Step 13 is complete. See `plan/STEP13.md`.

### Step 14: Multi-Model Hot Reload

**Goal**: Load/unload/reload models without restarting the runtime.

| # | Item | Status |
|---|------|--------|
| 14.1 | Multi-model registry (map of name→handle) | ✅ Done: `ModelSlot` map, per-version snapshots |
| 14.2 | Model add/remove at runtime via admin endpoint | ✅ Done: `/v2/admin/models` list/load/unload/reload |
| 14.3 | Version-switch with zero-downtime drain | ✅ Done: `versions/{v}/activate`, old scheduler retired in background |

Step 14 is complete. See `plan/STEP14.md`.

## Multi-Backend Track (Complete)

Phases 0–2 (multi-backend registry, plugin ABI/loader, runtime integration) and Phase 3.1
(typed byte-buffer input hot path, ~2.5× latency improvement) are merged to `develop` in
both `neuriplo` and `neuriplo-kserve-runtime`. Nothing from the approved multi-backend
plan remains in scope.

### In-flight merge gates

| PR | Repo | Purpose | Status |
|----|------|---------|--------|
| [#7](https://github.com/olibartfast/neuriplo-kserve-runtime/pull/7) | runtime | CI: real-* jobs check out `neuriplo@develop` after PR #13 merge | ✅ Merged |
| [#14](https://github.com/olibartfast/neuriplo/pull/14) | neuriplo | `get_infer_results_raw()` + `RawOutputTensor` on built-in backends | Open; merge before runtime Step 15 CI goes green |

**Merge order:** PR #14 (neuriplo raw API) → runtime Step 15 adapter PR.

## Follow-ups (Priority: High)

Natural completions and deferred items, roughly in order of value.

### Step 15: Raw output hot path (next)

**Goal:** Complete Phase 3.1 perf work on the output side. Plugins already bypass
`TensorElement` via the C ABI; built-in backends still pay ~16 B/scalar through
`get_infer_results()`.

**Blocked on:** neuriplo PR #14 merge to `develop` for CI; adapter work in
`feature/step-15-raw-output`.

| # | Item | Status |
|---|------|--------|
| 15.1 | `RealNeuriploAdapter::infer()` calls `get_infer_results_raw()` | ✅ Done |
| 15.2 | Map `RawOutputTensor.dtype` → KServe datatype; move `bytes`/`shape` directly into `OutputTensor` | ✅ Done |
| 15.3 | Drop `tensorElementToDouble` / `withTensorElementType` scalar loop on tensor path | ✅ Done |
| 15.4 | Keep `llmInfer()` on `get_infer_results()` until neuriplo exposes raw LLM output | Deferred |

**Scope note:** existing real-neuriplo CI presets (`real-multi`, `real-plugin`,
`real-onnx-grpc`) should cover regression; optional byte-identical comparison test vs old
path.

### CI hardening (small)

| Item | Status |
|------|--------|
| Pin neuriplo SHA in runtime real-* jobs (cross-repo API drift guard) | In PR #7 / follow-up |
| Free-disk-space step before ROCm/MIGraphX Docker builds | neuriplo-side (PR #14 era) |

### Feature follow-ups (deferred)

| Item | Notes |
|------|-------|
| `--model-repository` Triton-layout scan (`config.pbtxt` → backend id) | Auto-load `/home/oli/model_repository` without admin POSTs |
| `device_id` / multi-GPU placement | Per-backend device selection |
| Per-backend OBJECT-lib isolation (built-in mode) | Match plugin isolation for static backends |

### Release

Cut **neuriplo v0.6.0** (last release v0.5.0) and tag runtime after develop CI is stable
post PR #7 + #14. Multi-backend + plugin ABI is a solid minor bump.

### Parallel track (out of scope unless requested)

`feature/step-13-2-infer-data-plane` — Step 13.2 data-plane work; separate from Step 15.

## Optimization Track (Priority: Low / Scale-Triggered)

| Item | Trigger |
|------|---------|
| Async HTTP reactor (epoll/io_uring) | Connection concurrency > O(100) |
| Zero-copy tensor buffers | Copy overhead visible in profiling |
| Arena/PMR allocators | Allocation churn under load |
| Multi-datatype gRPC codec (INT8, FP16 in protobuf) | Client demand |

## Infrastructure Track

| Item | Status |
|------|--------|
| CI: build all sanitizers in parallel (currently sequential) | ✅ Done: `fail-fast: false` matrix (asan/ubsan/tsan run as parallel jobs) |
| CI: `real-onnx-grpc` preset job | ✅ Done: real-neuriplo presets build + ctest; PR #7 flips checkout to `neuriplo@develop` |
| CI: E2E smoke test with stub model via curl | ✅ Done: `scripts/e2e-stub.sh` (11 checks incl. admin lifecycle) + `e2e-smoke` job |
| CI triggers only covered `master`; GitFlow PRs target `develop` and ran no CI | ✅ Fixed: push/PR triggers cover `master` and `develop` |
| `.gitignore` entry for `.antigravitycli/` | ✅ Done (plus `logs/`) |
| `enable_testing()` missing → `ctest` ran zero tests in CI | ✅ Fixed |

## Cross-Repo Progress Notes

- `neuriplo-infer` `feature/neuriplo-kserve-runtime`: KServe HTTP client path is
  wired through the existing `InferenceInterface` boundary. Remote mode accepts
  `--kserve_endpoint`, defaults `--kserve_model_name` to `--type`, supports
  `--kserve_transport=http|grpc`, and no longer requires local `--weights` for
  remote execution.
- Validation: `cmake -S . -B build-kserve-codex -DDEFAULT_BACKEND=OPENCV_DNN
  -DENABLE_APP_TESTS=ON -DCMAKE_BUILD_TYPE=Release`,
  `cmake --build build-kserve-codex --parallel`, and
  `ctest --test-dir build-kserve-codex --output-on-failure` pass in
  `neuriplo-infer` (22/22 tests).
- Real HTTP E2E proof completed: launched `neuriplo-kserve-runtime` with
  `--backend onnx_runtime`, `--model-name yolo`, and `yolo26s.onnx`, then ran
  `neuriplo-infer --type=yolo26 --source=data/dog.jpg --labels=labels/coco.names
  --kserve_endpoint=http://127.0.0.1:19090 --kserve_model_name=yolo
  --kserve_transport=http --min_confidence=0.25`. The CLI completed successfully,
  runtime metrics reported `neuriplo_http_infer_requests_success_total{model="yolo",version="1"} 1`,
  and `data/output/processed.png` was written.
