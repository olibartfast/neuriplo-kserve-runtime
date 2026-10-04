# Neuriplo KServe Runtime Mission

> Status: working brownfield constitution, reconstructed on 2026-10-04 from
> the current code, documentation, repository metadata, and recent history.
> Product assumptions that still need maintainer confirmation are listed below.

## Why This Repository Exists

`neuriplo-kserve-runtime` is the serving data plane for
[neuriplo](https://github.com/olibartfast/neuriplo): a KServe-compatible
(Open Inference Protocol V2) C++ runtime that serves models through neuriplo
backends. It lets a model run behind a standard serving endpoint, on one host
or as a KServe `ServingRuntime` in a cluster, without each deployment writing
its own server around a backend.

## Who It Serves

- Clients of a KServe V2 endpoint, including the
  [neuriplo-infer](https://github.com/olibartfast/neuriplo-infer) CLI in remote
  mode (HTTP or gRPC) and any generic V2 client.
- Operators who deploy, scale, canary, and reload models and need readiness,
  metrics, and structured errors that are diagnosable.
- Maintainers who extend scheduling, batching, and pipeline behavior without
  destabilizing the serving path.

## What It Owns

- The KServe V2 HTTP surface and the optional gRPC surface, plus the admin
  load, unload, reload, and version-activation endpoints.
- Process lifecycle, model registry, model state, and the control/data plane
  split (`ModelLifecycle`, `SchedulerRetireQueue`).
- Request admission, backpressure, per-model scheduling, instance pools, and
  dynamic batching for compatible tensor models.
- The LLM token scheduler and the OpenAI-compatible endpoints for the llama.cpp
  and Cactus backends.
- Pipeline (ensemble) models: an ordered graph of preprocess, model, and
  postprocess steps served as one model.
- Metrics, health, readiness, container image, and KServe manifests.

## What It Must Never Do

- Own backend execution: model loading, backend adapters, runtime dependency
  compatibility, and thread-safety guarantees belong to
  [neuriplo](https://github.com/olibartfast/neuriplo).
- Own task semantics: model type strings, tensor expectations, and
  pre/postprocessing belong to
  [neuriplo-tasks](https://github.com/olibartfast/neuriplo-tasks). The runtime
  only calls them in process for pipeline steps, and only when
  `NEURIPLO_RUNTIME_ENABLE_TASKS` is on.
- Own the client or CLI flow: sources, visualization, and user configuration
  belong to [neuriplo-infer](https://github.com/olibartfast/neuriplo-infer) and
  [neuriplo-kserve-client](https://github.com/olibartfast/neuriplo-kserve-client).
- Silently change tensor shapes, dtypes, or output semantics, or silently
  substitute a backend, device, or model format.
- Link OpenCV into the runtime; the task steps use the stb image path only.

Cross-repository contracts, the version matrix, and joint packets live in
[neuriplo-platform](https://github.com/olibartfast/neuriplo-platform).

## What Good Means

- A stub-only build needs no neuriplo and still passes the full unit suite.
- The V2 protocol surface behaves the same over HTTP and gRPC for the same
  request, and errors carry the real code and message to the client.
- Reload, drain, and version switches never block or drop in-flight inference.
- Malformed requests and bad pipeline graphs fail at the boundary, with an
  actionable message, never as a crash or a generic internal error.
- Sanitizer presets pass for changes touching request handling, threading,
  parsing, or ownership.
- A new maintainer can find the boundaries and run the required checks from
  version-controlled files.

## Engineering Values and Tone

Protocol correctness, no silent behavior change, bounded resource use, and
clear failure take priority over surface features. Documentation is direct and
operational: exact commands, supported paths, known exclusions.

## Assumptions to Confirm

- [A-1] The standing non-goals above (no auth/TLS beyond
  reverse-proxy friendliness, no ModelMesh replacement, no distributed model
  cache) still hold. "No multi-model hot reload" no longer does:
  Step 14 shipped it.
- [A-2] The conservative custom `modelFormat: neuriplo` stays the only KServe
  format advertised; automatic selection for `onnx`, `openvino`, `tensorrt`, or
  `gguf` is still unwanted.
- [A-3] The OpenAI-compatible LLM endpoints remain in scope for this repo
  rather than moving to a dedicated LLM gateway.
- [A-4] No quantitative latency, throughput, or memory targets exist; each
  packet states its own.
- [A-5] The sibling-repo ownership split above matches the platform contracts.

_Revision: 2026-10-04 - initial brownfield constitution._
