# Step 12: Multi-Component YOLO Integration - requirements

> Retrospective packet. Step 12 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 12
> snapshot (`plan/STEP12.md`, later `specs/history/steps/STEP12.md`) and the
> Step 12 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Roadmap phase: [Phase 1 - Platform E2E and Production Track](../roadmap.md#phase-1---platform-e2e-and-production-track-steps-12-14)
Status: **Complete**
Specified: 2026-06-05 (snapshot date)

## Goal

Smoke-test the full neuriplo platform end-to-end: `neuriplo-infer` → KServe V2
(HTTP or gRPC) → `neuriplo-kserve-runtime` → `neuriplo` → ONNX Runtime backend →
YOLO model → response → `neuriplo-infer` displays result.

Outcome recorded in the snapshot: the platform-defined local serving
chain `neuriplo-infer` -> KServe V2 HTTP -> `neuriplo-kserve-runtime` ->
`neuriplo` -> ONNX Runtime -> YOLO was tested end-to-end.

## In Scope

### Components

| Component | Repo | Dev Branch | Role |
|-----------|------|-----------|------|
| `neuriplo` | `github.com/olibartfast/neuriplo` | `neuriplo-kserve-runtime` | ONNX Runtime backend, model load/infer |
| `neuriplo-tasks` | `github.com/olibartfast/neuriplo-tasks` | `develop` | YOLO task contract, pre/post tensor shapes |
| `neuriplo-infer` | `github.com/olibartfast/neuriplo-infer` | `develop` | CLI, reads image, calls KServe endpoint |
| `neuriplo-kserve-runtime` | this repo | `master` | KServe V2 HTTP+gRPC server |

## Requirements

Each requirement is an exit criterion of the original step plan; the
`(n.m)` tag names the substep that owned it.

- [R-1] (12.1) The YOLO task contract in neuriplo-tasks matches the exported model's input and output tensors.
- [R-2] (12.2) The runtime builds with `NEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO=ON` against a neuriplo checkout.
- [R-3] (12.3) A real YOLO ONNX model loads through neuriplo and its metadata comes from the backend.
- [R-4] (12.4) Inference through `/v2/models/yolo/infer` reaches ONNX Runtime and returns the expected output shape.
- [R-5] (12.5) The `neuriplo-infer` CLI drives the runtime over KServe HTTP and renders detections.
- [R-6] (12.6) gRPC returns the same inference result as HTTP.
- [R-7] (12.7) An automated smoke script checks health, metadata and inference over HTTP and gRPC.

## Out of Scope

Recorded at the end of the step as remaining gaps; later steps picked them
up as noted in [`../roadmap.md`](../roadmap.md).

### Known gaps when the procedure was written

1. **neuriplo-tasks YOLO task contract** — verify input shape (H×W), number of classes,
   output format (boxes + class scores) matches the ONNX model export.
2. **neuriplo-infer KServe client** — needs an HTTP/gRPC client abstraction that
   sends KServe V2 JSON or protobuf requests instead of linking neuriplo directly.
3. **Model artifact** — need a YOLOv8n or YOLOv5s exported to ONNX with the
   expected input/output names (`input` → `output`).
4. **Dynamic batching** — if enabled, batch dimension handling must match the
   ONNX model's NCHW layout.

All four were closed by the snapshot's 12.1-12.7 results.
