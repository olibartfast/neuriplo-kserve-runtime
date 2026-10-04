# Neuriplo KServe Runtime Technical Stack

> Status: working brownfield constitution, reconstructed on 2026-10-04. The
> implementation and executable checks take precedence if this file drifts;
> update both in the same branch when a durable technical decision changes.

## Core Stack

| Area | Current choice | Boundary |
| --- | --- | --- |
| Language | C++17 | `CMAKE_CXX_STANDARD 17`, required. Do not raise without an explicit compatibility decision. |
| Build | CMake 3.20 or newer, Ninja, `CMakePresets.json` | Presets are the supported build entry; sources are listed in `CMakeLists.txt`. |
| Transports | HTTP (always), gRPC V2 (`NEURIPLO_RUNTIME_ENABLE_GRPC`) | Both expose the same V2 surface; gRPC needs `libgrpc++-dev` and `protobuf-compiler-grpc`. |
| Execution | Stub executor by default; real adapter via `NEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO` | The default build must not require neuriplo. |
| Tests | In-repo harness `tests/Test.hpp` through CTest | Name files after the module; cover routes, config parsing, and failure paths. |
| Quality | clang-format (`scripts/check-format.sh`), clang-tidy (`lint`), ASan/UBSan/TSan presets, Valgrind in CI | Use the checked-in scripts and presets. |
| CI | GitHub Actions, `.github/workflows/ci.yml` | Triggers cover `master` and `develop`. |

Presets: `debug`, `release`, `lint`, `asan`, `ubsan`, `tsan`, `grpc`, and the
real-neuriplo set `real-onnx`, `real-onnx-grpc`, `real-multi` (OpenCV DNN +
ONNX Runtime built in), `real-plugin` (ONNX Runtime as a `dlopen` plugin),
`real-openvino`, `real-openvino-grpc`, `real-executorch`,
`real-executorch-grpc`.

## Dependencies and Pinning

- `versions.env` pins [neuriplo](https://github.com/olibartfast/neuriplo)
  (`NEURIPLO_VERSION`) and
  [neuriplo-tasks](https://github.com/olibartfast/neuriplo-tasks)
  (`NEURIPLO_TASKS_VERSION`) to release tags. Bumps are compatibility work, not
  routine.
- Real-neuriplo presets auto-clone neuriplo into
  `build/<preset>/_deps/neurip-src/` at the pinned tag. Override with
  `-DNEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR=<path>` (and
  `NEURIPLO_RUNTIME_TASKS_SOURCE_DIR` for tasks) to iterate on a sibling
  checkout; a local checkout bypasses the pins, so validate pins separately.
- `NEURIPLO_RUNTIME_ENABLE_TASKS` (default `OFF`) links `neuriplo-tasks` and
  `neuriplo-tasks::vision-stb`, never `vision-opencv`. Built `OFF`, the
  dependency graph is unchanged and task pipeline steps refuse to load.
- Release metadata: `VERSION` (read by CMake) and `CHANGELOG.md` (Keep a
  Changelog). `scripts/release-patch.sh` runs a GitFlow patch release.

## Architectural Boundaries

- Hide concrete scheduler, executor, and backend types behind interfaces and
  factory functions (`Scheduler`, `Executor`, `NeuriploAdapter`). Prefer
  extending existing Strategy, factory, and adapter boundaries over new
  frameworks; patterns in use are recorded in [architecture.md](architecture.md).
- `NeuriploAdapter` is the only seam to neuriplo. The real adapter uses the
  raw-output path (`get_infer_results_raw`) on tensors; `llmInfer()` stays on
  `get_infer_results()`.
- Control plane (`ModelLifecycle`: load, unload, reload, activate) is separate
  from the infer data plane; retired schedulers drain in the background via
  `SchedulerRetireQueue`.
- `RuntimeConfig` owns CLI and config parsing; `KServeV2Codec` and
  `GrpcV2Codec` own wire translation; `AdminCodec` owns admin bodies.
- Pipeline graphs (`PipelineConfig`, `PipelineExecutor`) are validated at load,
  not at first request.
- Backends are built in or loaded as plugins by neuriplo; the runtime reports
  model `platform` as `neuriplo_<backend>`.

## Explicit Non-Choices

- No task-aware output schema in the core serving path outside pipeline steps.
- No OpenCV in the runtime binary.
- No new dependency without explicit approval and a compatibility review.
- No auth or TLS in the runtime; deploy behind a reverse proxy.
- No change to tensor shapes, dtypes, or output semantics in the server.
- No feature work committed directly to `master`.
- Non-trivial public-behavior or architecture work records scope and validation
  in a dated packet before implementation.

## Validation Entrypoints

```bash
cmake --preset debug && cmake --build --preset debug && ctest --preset debug
scripts/check-format.sh
cmake --preset lint && cmake --build --preset lint --parallel
cmake --preset asan && cmake --build --preset asan && ctest --preset asan
cmake --preset grpc && cmake --build --preset grpc && ctest --preset grpc
cmake --preset real-onnx && cmake --build --preset real-onnx && ctest --preset real-onnx
scripts/e2e-stub.sh
```

Run format and lint before every push; add `ctest --preset debug` when C++
behavior changes. `scripts/e2e-yolo.sh` and `scripts/e2e-multi-backend.sh`
need a real model or a local GPU host and are manual checks. Documentation
edits must keep relative links resolving.

## Assumptions to Confirm

- [A-6] CI exercises `debug`, `lint`, sanitizers, Valgrind, `grpc`, the stub
  smoke test, the repository preparer, and the real-onnx jobs; the full job
  list in `ci.yml` was read only in part, so confirm no job is missing here.
- [A-7] The `real-openvino` and `real-executorch` presets are supported, not
  experimental, and have a declared validation path.

Resolved:

- (was A-8) The `versions.env` pins are not current. The 2026-10-04
  pre-release audit found neuriplo-tasks v0.8.0 lacks `decodeImage`, so a build
  with `NEURIPLO_RUNTIME_ENABLE_TASKS=ON` does not compile. The pins move to the
  ecosystem's current releases in the v0.4.0 release batch.

_Revision: 2026-10-04 - initial brownfield constitution._
