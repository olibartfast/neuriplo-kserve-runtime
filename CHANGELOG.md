# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.4.0] - 2026-10-05

### Added
- **Model repository serving**: `--models` (alias `--model-repository`, env
  `MODEL_REPOSITORY`) scans a Triton-style tree,
  `<root>/<model>/<version>/<file>`, infers the backend from the file
  extension and serves every model found. The backend choice among several
  candidates follows a priority list, overridable with
  `NEURIPLO_REPOSITORY_BACKEND_PRIORITY`. Versions compare as digit strings, so
  an oversized version directory sorts correctly instead of aborting the scan
  with `fatal: stoull`. Ensembles are ordinary repository entries. See
  `docs/model-repository.md`.
- `--model-control-mode none|explicit`. `none` (default) loads everything at
  startup; `explicit` loads nothing and leaves loading to the client, so a model
  whose backend crashes on load cannot stop the server from starting. A
  bare-name load resolves against the repository catalog.
- **KServe model repository extension**: `POST /v2/repository/index` and
  `POST /v2/repository/models/{name}/load|unload`, driving the same registry as
  `/v2/admin/models`, which is unchanged. `neuriplo-kserve-client` already spoke
  this extension, but the server only exposed it at the non-standard admin path.
- `--flag=value` argument parsing. Unknown arguments throw, so the equals form
  used by most manifests used to abort startup.
- **Init-container deployment**: `deploy/prepare` (the repository preparer
  image and `prepare-repository.sh`), `docker/Dockerfile.tensorrt` (converts
  ONNX to a TensorRT engine at serve time, on the node that will serve it,
  because an engine is tied to its GPU, driver and TensorRT version),
  `deploy/k3d/` and `deploy/compose/` (the same flow with and without
  Kubernetes), and `deploy/models/build-model-image.sh`. A new CI job
  (`prepare-repository`) runs the preparer tests under dash and busybox sh and
  shellchecks the scripts.
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
- GPU-accelerated ensembles: chaining a DALI preprocessing model, a TensorRT
  engine, and a DALI GPU postprocessing model runs the compute of every step on
  the GPU. Results are still staged through host memory between steps -- the
  runtime copies each step's output out before handing it to the next step --
  so the pipeline is not device-resident end to end. On YOLO26m-seg the
  server-side pipeline goes from 144.5 ms (CPU pre and post) to 69.9 ms, 2.07x.
  The source-dimensions tensor is INT64 (height, width) so one preprocessing
  output feeds either postprocess path.
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

### Changed
- **MODEL_VERSION / `--model-version` now override the version a backend
  reports**, but only when actually requested (the flag, the env var, or a
  repository version directory). `ModelLifecycle::load` used to fall back to
  `"1"` whenever no override was passed, so `--model-version` was silently
  ignored. With no version requested, the backend-reported version still wins,
  so a deployment that relied on the old fallback may now see a different
  version string.
- **Readiness semantics**: in explicit model-control mode an empty registry is
  ready; in repository mode a failed model is reported through
  `/v2/repository/index` as `UNAVAILABLE` with its reason and no longer makes
  `/v2/health/ready` fail, unless no model is ready at all. A first load in
  progress does not make the server unready. Single-model mode keeps strict
  readiness.
- Combining a repository with the single-model flags (`--model-path`,
  `--backend`, `--model-name`, `--model-version`) now fails fast at startup.
  `MODEL_VERSION` from the environment stays compatible with `MODEL_REPOSITORY`.
- Dependency pins: neuriplo v0.10.0 and neuriplo-tasks v0.8.2.
- CMake pin precedence is now `-D` flag, then `versions.env`, then the built-in
  default, so `-DNEURIPLO_TASKS_VERSION=...` can override the file.
- A new CI job, `debug-tasks`, builds and tests with
  `NEURIPLO_RUNTIME_ENABLE_TASKS=ON`; no job built that configuration before.
- `prepare-repository.sh` publishes atomically (the old version is moved aside,
  the new one renamed in, and the old one restored if the rename fails) and
  writes a `.prepared` stamp per version (source sha256; for engines also
  TensorRT version, GPU, driver, precision, shapes and extra args). A version is
  skipped only when its artifact exists and the stamp matches. **Upgrade note**:
  an existing volume has no stamp, so every engine and copied model is rebuilt
  once on the first start after upgrading.

### Fixed
- The pinned neuriplo-tasks v0.8.0 lacked `decodeImage` (added in v0.8.1), so
  the `NEURIPLO_RUNTIME_ENABLE_TASKS=ON` build, the one that serves
  encoded-image ensembles, did not compile. Fixed by the v0.8.2 pin.
- Executor failures were reported to clients as a generic
  `INTERNAL / internal error`: the scheduler set `ok = false` without copying
  the executor's error code and message onto the `SchedulerResult`, and the
  transports read only those fields on failure. All paths (including the LLM
  executor and the dynamic-batch merge) now go through one `adopt()` that
  carries the real error through -- the DALI ensemble failure was
  undiagnosable from the outside until they did.
- Model metadata declaring a negative (dynamic) dimension rejected every
  inference request, because request shapes were compared for exact equality
  against the metadata shape, in `NeuriploExecutor` and in the codec. A negative
  metadata dimension is KServe's dynamic-axis marker and now accepts any
  concrete extent. Encoded-image inputs are the first models here to need it:
  their byte length varies per request.
- Pipeline model steps reconcile batch-dimension conventions between producers
  and consumers: backends disagree about whether metadata shapes include the
  batch dimension (ONNX Runtime `[1,3,640,640]`, TensorRT `[3,640,640]`), so a
  tensor produced by one step could fail the next model's validation with the
  bytes exactly right. The seam restates the shape in the target model's
  convention only when the two shapes differ solely by leading dimensions of
  extent 1 and every other dimension matches. Equal element counts are not
  enough: NHWC `[640,640,3]` must not be relabelled NCHW `[3,640,640]` and fed
  to a model transposed. Declared dynamic axes are left alone.
- Packed-mask envelopes shipped no mask bytes for YOLO segmentation. The encoder
  read `InstanceSegmentation::mask_data`, which the YOLO postprocessor leaves
  empty -- it populates the mask image instead, and does so at box size on one
  code path and frame size on another. All three cases are now normalized to the
  box-sized bytes the contract specifies.
- Pipeline executor: a tensor a later step still needs is no longer moved out
  from under it; a step that references itself and nested ensemble steps are
  rejected at load.
- Encoded-image preprocess caps decoded pixels (64 Mi) before decoding, using a
  header walk that matches the vendored stb_image for PNG (including CgBI),
  JPEG (fill and padding bytes) and BMP. A hostile header can no longer force a
  huge allocation. `FRAME_SIZE` must be INT64 with at least 2 elements.
- Postprocess top-score ordering is stable and puts NaN scores last. Wrong-typed
  float fields in a pipeline config are errors instead of being coerced.
- KServe V2 codec element counts are overflow-checked, and the duplicated
  `shapeMatches` is gone.
- Model lifecycle: a slow load, reload or version switch no longer blocks
  inference on other models, and a first load no longer fails
  `/v2/health/ready`. Concurrent loads of one name join the in-flight build.
  `ModelLifecycle::load` converts exceptions into a Failed state, and a failed
  reload of a Failed model stays Failed.
- Admin and repository status codes: unload during a load or reload, admin load
  of an already-loaded model, and a joined load that failed (with its
  `load_error`) return 409. The repository route returns 404 only after the body
  parses. Admin load fields are type-checked (400) with the CLI's ranges (empty
  backend, instance count 1..64, queue, batch and timeout limits).
- Repository preparation no longer serves stale engines after a model,
  TensorRT, GPU or precision change; two staged artifacts with the same model
  name fail the run, naming both; `MODEL_VERSION` and tree version directories
  must be integers; a failed rebuild leaves the served version untouched.
- Deploy scripts: `build-model-image.sh` rejects non-ONNX inputs and duplicate
  basenames, the serve scripts require their dependency paths instead of
  defaulting to a developer's home, compose staging is cleared before each copy,
  and busybox is pinned by version and digest.
- Repository scan: the highest version directory that actually holds a model
  file is served, so an empty top-numbered directory no longer drops the whole
  model. Empty, non-numeric and duplicate (leading-zero) versions are skipped
  with a warning, the reported version is canonical (`1`, never `01`), and the
  scan iterates without throwing.
- Activating a version of a repository-mode model resolves the requested
  version through the scanner (numeric only, never a joined path) and returns
  404 when it has no model, so one version's weights are never served under
  another version's label.
- `/v2/repository/index` takes an optional body (invalid JSON is 400, and
  `{"ready":true}` filters), no longer throws on non-UTF-8 names, and answers
  405 with `METHOD_NOT_ALLOWED`; unknown repository actions are 404 for any
  method. The README now lists the repository endpoints.
- `--log-payloads` and `--tokens-per-char` are honored in explicit mode.
- Metrics carry per-model load success/failure and version labels, and
  repository mode no longer reports a phantom startup model.
- Ensembles no longer inherit the server-wide dynamic batching defaults, which
  made them fail to load; an explicit batching request on an ensemble is still
  rejected.
- An ensemble whose step model failed to load, or is still loading, is refused
  at load instead of coming up ready with empty metadata.
- A reload keeps the explicitly requested version (repository version
  directory or `--model-version`) instead of reverting to the backend-reported
  one.
- Version activate rejects a body `version` that differs from the URL version
  (400). `--help` lists `--models`, `--model-control-mode`, `--use-gpu` and
  `--plugin-dir`.

### Known limitations
- Admin and repository routes are unauthenticated and accept an arbitrary
  `model_path` and `plugin_dir`. Expose them only on trusted networks.
- Images and pods run as root, and `docker/Dockerfile.tensorrt` builds from the
  context checkout rather than a pinned source.
- The gRPC codec does no metadata validation; the executors validate.
- TGA images have no pre-decode size cap, because the format has no signature to
  detect it by.
- An ensemble stays ready when one of its step models is unloaded, and its
  cached step metadata goes stale.

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
