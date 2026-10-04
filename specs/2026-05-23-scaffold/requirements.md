# Step 0: Scaffold - requirements

> Retrospective packet. Step 0 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 0
> snapshot (`plan/STEP0.md`, later `specs/history/steps/STEP0.md`) and the
> Step 0 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Roadmap phase: [Phase 0 - Serving Foundation](../roadmap.md#phase-0---serving-foundation-steps-0-11)
Status: **Complete**
Specified: 2026-05-23 (snapshot date)

## Goal

This document records what is implemented in this repository today. It is a
snapshot of the scaffold, not the target architecture.

The repository currently builds a small C++17 HTTP server with a minimal
KServe/Open Inference Protocol V2-compatible surface.

Implemented endpoints:

```text
GET  /v2
GET  /v2/health/live
GET  /v2/health/ready
GET  /v2/models/{model_name}
GET  /v2/models/{model_name}/ready
POST /v2/models/{model_name}/infer
```

The runtime can be launched locally with CLI configuration for host, port,
model name, model path, and backend name.

## In Scope

- Create repo.
- Add CMake project.
- Add basic HTTP server.
- Add health and metadata endpoints.
- Add placeholder `/infer`.
- Add plan and README.

## Requirements

Each requirement is an exit criterion of the original step plan; the
`(n.m)` tag names the substep that owned it.

- [R-1] Builds locally.
- [R-2] `curl /v2/health/live` returns 200.
- [R-3] `curl /v2/health/ready` returns 200 for stub model.
- [R-4] `curl /v2/models/demo` returns model metadata.

## Out of Scope

Recorded at the end of the step as remaining gaps; later steps picked them
up as noted in [`../roadmap.md`](../roadmap.md).

### What Is Stubbed Or Missing

The repository is not yet ready for real `neuriplo`, `vision-core`, or
`vision-inference` integration.

Missing or stubbed items:

- No `neuriplo` dependency is linked.
- No executor abstraction exists yet.
- No `NeuriploExecutor` exists yet.
- `/infer` ignores the request body.
- `/infer` returns a fixed dummy output tensor.
- Model metadata does not come from the backend.
- Model readiness is not tied to real model loading.
- No scheduler, queue, timeout, batching, or backpressure implementation exists.
- No KServe `ServingRuntime` or `InferenceService` manifests exist.
- No container image definition exists.
- No `/mnt/models`, `STORAGE_URI`, or KServe storage-initializer behavior exists.
- No gRPC V2 support exists.
- No Prometheus `/metrics` endpoint exists.
- No LLM/token scheduler exists for llama.cpp or Cactus backends.
- No OpenAI-compatible `/v1/*` endpoints exist.

### Practical Status

This repository is currently at the scaffold stage:

```text
M0: Scaffold implemented
M1+: Not implemented
```

It is useful for validating basic process startup, routing, HTTP parsing, and
the shape of the initial KServe V2 surface. It is not yet useful for real model
execution or sibling repository integration.
