# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
