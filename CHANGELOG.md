# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **Pipeline (ensemble) models**: a `backend: ensemble` model kind that serves an
  ordered graph of steps as one KServe V2 model. `model` steps execute another
  model in the registry; `preprocess` and `postprocess` steps run the task layer
  in process. The composed model reports `platform: ensemble`, exposes the first
  step's inputs and the last step's outputs, and emits the platform ensemble
  contract's decoded envelopes (detection, packed mask, polygon). Graphs are
  validated at load: unknown step kinds, duplicate names, model steps without a
  model, references to unloaded models, and tensors no earlier step produces all
  fail the load rather than the first request. See `deploy/ensemble/README.md`.
- `NEURIPLO_RUNTIME_ENABLE_TASKS` (default `OFF`) builds the task-layer
  preprocess/postprocess steps, linking `neuriplo-tasks` and
  `neuriplo-tasks::vision-stb` -- never `vision-opencv`, so the runtime stays
  OpenCV-free. Built `OFF`, the dependency graph is unchanged and those steps
  refuse to load with an explicit message.
- `pipeline_graph` in the admin load body, for supplying a graph inline instead
  of by path.
- `deploy/ensemble/yolo-seg-dali-tensorrt.json`: worked DALI -> TensorRT ->
  postprocess graph, and launcher scripts (`scripts/serve-dali-trt.sh`,
  `scripts/serve-onnx-gpu.sh`) recording each configuration's runtime library
  requirements -- NVIDIA runtimes come from NVIDIA's own distribution in a
  dedicated directory, never from another vendor's package tree.
- `dali` backend id: a DALI "model" is a serialized GPU preprocessing pipeline
  (nvJPEG decode, letterbox, normalize) served like any other model, so an
  ensemble can chain it ahead of a TensorRT engine. Validated live: DALI ->
  TensorRT FP16 -> task postprocess on YOLO26-seg agrees with the CPU
  preprocessing path on 91% of detections over 50 frames (96% at
  confidence >= 0.5), disagreements concentrated at the 0.30 threshold margin.
- `--use-gpu` (also `use_gpu` in the admin load body). The runtime built
  `EngineOptions` without ever setting `use_gpu`, so it could only serve on the
  CPU no matter what the backend was capable of. On an ONNX Runtime GPU build,
  a YOLO26-seg ensemble goes from ~1650 ms to ~160 ms per request.

### Fixed
- Executor failures were reported to clients as a generic
  `INTERNAL / internal error`: the scheduler set `ok = false` without copying
  the executor's error code and message onto the `SchedulerResult`, and the
  transports read only those fields on failure. Both fulfil paths now carry the
  real error through -- the DALI ensemble failure was undiagnosable from the
  outside until they did.
- `NeuriploExecutor` validated request shapes against metadata with exact
  equality, rejecting every model that declares a dynamic (negative) dimension.
  The same dynamic-axis rule as the KServe codec now applies.
- Pipeline model steps reconcile batch-dimension conventions between producers
  and consumers: backends disagree about whether metadata shapes include the
  batch dimension (ONNX Runtime `[1,3,640,640]`, TensorRT `[3,640,640]`), so a
  tensor produced by one step could fail the next model's validation with the
  bytes exactly right. When element counts agree, the seam restates the shape
  in the target model's convention.
- Packed-mask envelopes shipped no mask bytes for YOLO segmentation. The encoder
  read `InstanceSegmentation::mask_data`, which the YOLO postprocessor leaves
  empty -- it populates the mask image instead, and does so at box size on one
  code path and frame size on another. All three cases are now normalized to the
  box-sized bytes the contract specifies.
- Model metadata declaring a negative (dynamic) dimension rejected every
  inference request, because request shapes were compared for exact equality
  against the metadata shape. A negative metadata dimension is KServe's
  dynamic-axis marker and now accepts any concrete extent. Encoded-image inputs
  are the first models here to need it: their byte length varies per request.

## [0.3.2] - 2026-06-15

## [0.3.1] - 2026-06-14

### Changed
- **Build system**: neuriplo is now auto-cloned from GitHub into
  `build/<preset>/_deps/neurip-src/` at configure time (tag pinned in
  `versions.env`). The mandatory `../neuriplo` sibling checkout is no longer
  required. Local checkout still available as override via
  `-DNEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR=/path/to/neuriplo`.
- CI: real-* jobs no longer need a second `actions/checkout` for neuriplo.
- Presets: `NEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR` removed from all real-*
  preset cache variables.

### Added
- `versions.env` -- single source of truth for third-party dependency versions
  (`NEURIPLO_VERSION`).

## [0.3.0] - 2026-06-14

### Added
- `litert` registered as a neuriplo tensor backend, so a litert-enabled build
  (`-DDEFAULT_BACKEND=LITERT -DLITERT_DIR=...`) serves TFLite models over KServe
  V2. Validated locally 2026-06-13 with a `.tflite` round-trip
  (`platform: neuriplo_litert`). Availability for non-litert binaries is still
  gated by `realNeuriploAvailableBackends`.

## [0.2.0] - 2026-06-13

### Added
- HTTP binary tensor framing for lower-copy inference payloads.
- gRPC `raw_output_contents` emission for tensor outputs (KServe V2
  conformance).
- Real tensor datatype propagation in model metadata via the
  `RealNeuriploAdapter` (requires neuriplo v0.7.0 plugin metadata ABI v2).

### Changed
- `RealNeuriploSmokeTest` guards non-FP32 metadata datatypes (e.g. INT64
  identity ONNX models).

## [0.1.0] - 2026-06-12

### Added
- KServe V2 HTTP and optional gRPC serving runtime with stub and real-neuriplo
  backends.
- Multi-model registry, version switching with drain, and admin lifecycle
  endpoints.
- Real-neuriplo adapter using `get_infer_results_raw` typed byte outputs.
- Multi-backend plugin mode (built-in plus dlopen backends in one process).
- Dynamic batching, LLM scheduler path, metrics, and e2e smoke scripts.

[Unreleased]: https://github.com/olibartfast/neuriplo-kserve-runtime/compare/v0.3.2...HEAD
[0.3.0]: https://github.com/olibartfast/neuriplo-kserve-runtime/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/olibartfast/neuriplo-kserve-runtime/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/olibartfast/neuriplo-kserve-runtime/releases/tag/v0.1.0
[0.3.2]: https://github.com/olibartfast/neuriplo-kserve-runtime/compare/v0.3.1...v0.3.2
