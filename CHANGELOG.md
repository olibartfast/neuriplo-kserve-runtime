# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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

[Unreleased]: https://github.com/olibartfast/neuriplo-kserve-runtime/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/olibartfast/neuriplo-kserve-runtime/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/olibartfast/neuriplo-kserve-runtime/releases/tag/v0.1.0
