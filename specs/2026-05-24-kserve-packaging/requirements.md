# Step 1: KServe Packaging - requirements

> Retrospective packet. Step 1 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 1
> snapshot (`plan/STEP1.md`, later `specs/history/steps/STEP1.md`) and the
> Step 1 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Roadmap phase: [Phase 0 - Serving Foundation](../roadmap.md#phase-0---serving-foundation-steps-0-11)
Status: **Complete**
Specified: 2026-05-24 (snapshot date)

## Goal

This document records what was implemented for Step 1. The runtime is still on
the stub model execution path, but it can now be packaged and configured in the
shape expected by a KServe custom runtime.

Step 1 adds:

- Container image definition.
- KServe `ClusterServingRuntime` manifest.
- Example KServe `InferenceService`.
- KServe/container configuration defaults.
- `/mnt/models` model path convention.
- Basic deployment documentation.
- Tests for new configuration behavior.

Step 1 does not add real `neuriplo` execution, full V2 infer parsing,
scheduling, batching, metrics, gRPC, or LLM endpoints.

## In Scope

- Add Dockerfile.
- Add `ClusterServingRuntime` manifest for `modelFormat: neuriplo`.
- Add example `InferenceService`.
- Add `/mnt/models` and KServe environment default handling.
- Document local KServe deployment flow.

## Requirements

Each requirement is an exit criterion of the original step plan; the
`(n.m)` tag names the substep that owned it.

- [R-1] Runtime image can be referenced by a KServe `InferenceService`.
- [R-2] KServe storage-initialized model path resolves to `/mnt/models`.
- [R-3] Health/readiness endpoints are compatible with Kubernetes probes.

Not every requirement was fully met; the delivered status of each is in
[`validation.md`](validation.md#deviations).

## Out of Scope

Nothing recorded beyond the later steps in the target design.
