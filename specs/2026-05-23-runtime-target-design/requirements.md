# Runtime Target Design - requirements

> Retrospective packet. This is the original milestone target design for
> `neuriplo-kserve-runtime` (`plan/ROADMAP.md`, later
> `specs/history/target-design.md`), ported on 2026-10-05. It framed Steps 0-11;
> each step's scope and exit criteria now live in that step's packet. The live
> plan is [`../roadmap.md`](../roadmap.md); current patterns are in
> [`../architecture.md`](../architecture.md).

Roadmap phase: [Phase 0 - Serving Foundation](../roadmap.md#phase-0---serving-foundation-steps-0-11)
Status: **Complete**
Specified: 2026-05-23

## Goal

Build a production-oriented, KServe-compatible C++ inference runtime that serves
models through `neuriplo` backends while keeping application behavior and task
semantics outside the server unless explicitly added later.

The runtime should be usable in two deployment modes:

```text
same host:
  vision-inference -> http://127.0.0.1:8080 -> neuriplo-kserve-runtime -> neuriplo

cluster:
  vision-inference/client -> KServe endpoint -> neuriplo-kserve-runtime pod -> neuriplo
```

The first implementation target is raw tensor inference over KServe V2. Task
preprocessing and postprocessing stay in `vision-core` / `vision-inference`
unless a later task-aware serving layer is explicitly designed. LLM serving is a
separate first-class track because `neuriplo` already has llama.cpp and Cactus
backends, but it should use token-aware scheduling instead of the tensor-model
dynamic batching path.

## Repository Boundary

### This Repository Owns

- KServe/Open Inference Protocol V2 HTTP and gRPC surface.
- Runtime process lifecycle.
- Model registry and model state.
- Request admission and backpressure.
- Per-model scheduling.
- Execution instance pools.
- Dynamic batching for compatible tensor models.
- LLM/token scheduler for llama.cpp, Cactus, and GGUF-style local models.
- Metrics, health, readiness, and structured errors.
- Container image and KServe `ServingRuntime` manifests.
- KServe deployment examples for `InferenceService`, canary rollout, and
  InferenceGraph-friendly request/response behavior.

### `neuriplo` Owns

- Backend abstraction.
- Backend adapters.
- Runtime dependency compatibility.
- Model loading/execution for local backends.
- Backend-specific metadata extraction.
- Thread-safety guarantees or documented limitations.

### `vision-core` Owns

- Task contracts.
- Model type strings.
- Tensor shape/dtype expectations.
- Preprocessing and postprocessing semantics.
- Result variants and output schema.

### `vision-inference` Owns

- CLI/application flow.
- Source/video input.
- User configuration.
- Calling a remote KServe endpoint.
- Visualization and local output.

## Out of Scope

- No task-aware output schema in the server.
- No OpenAI-compatible API in the first CV milestone. OpenAI-compatible LLM
  endpoints are allowed only in the explicit LLM milestone for llama.cpp and
  Cactus backends.
- No multi-model hot reload.
- No distributed model cache.
- No ModelMesh replacement.
- No auth/TLS beyond being reverse-proxy friendly.
- No changes to tensor shapes, dtypes, task decoding, NMS, or output semantics.

## Compatibility Target

Implement the KServe V2 protocol surface required by a custom serving runtime:

```text
GET  /v2
GET  /v2/health/live
GET  /v2/health/ready
GET  /v2/models/{model_name}
GET  /v2/models/{model_name}/ready
POST /v2/models/{model_name}/infer
```

Initial response bodies should match the Open Inference Protocol shape closely
enough that KServe clients and simple HTTP tests can validate readiness,
metadata, and inference behavior.

The runtime should also be deployable through KServe's model spec using a
conservative custom model format:

```yaml
predictor:
  model:
    modelFormat:
      name: neuriplo
    protocolVersion: v2
    runtime: neuriplo-kserve-runtime
    storageUri: pvc://...
```

Do not advertise automatic selection for generic formats such as `onnx`,
`openvino`, `tensorrt`, or `gguf` until metadata conversion and backend behavior
are validated per format.

## Requirements

- [R-1] The runtime serves the KServe V2 HTTP surface listed under
  Compatibility Target, with Open Inference Protocol response shapes.
- [R-2] It deploys through KServe with `modelFormat: neuriplo`,
  `protocolVersion: v2`, and `autoSelect: false` for every format.
- [R-3] Raw tensor inference runs through `neuriplo` backends; task
  pre/postprocessing stays in the client.
- [R-4] LLM serving (llama.cpp, Cactus) uses token-aware scheduling, not
  tensor dynamic batching.
- [R-5] Model lifecycle, scheduling, observability and error categories
  follow the designs in [`plan.md`](plan.md).
- [R-6] Each step lands with a recorded snapshot of what was built and
  validated (now a dated packet).

## Decisions

- [D-1] Keep this as a separate repository.
- [D-2] Keep `neuriplo` server-side only.
- [D-3] Keep `vision-inference` as a KServe client in the target architecture.
- [D-4] Start with raw tensor inference.
- [D-5] Package as a KServe custom `ServingRuntime` early, before backend-specific optimization.
- [D-6] Add task-aware serving only after tensor serving is stable.
- [D-7] Add llama.cpp and Cactus LLM support as a dedicated scheduling policy, not as ordinary dynamic batching.
- [D-8] Prefer correctness and stable protocol behavior before optimizing transport.
- [D-9] Evolve architecture using `specs/architecture.md` for current patterns and the "Architecture And Design Pattern Evolution" section for target patterns.

## Risks

- Backend thread-safety is unknown per backend.
- Tensor dtype mapping must be explicit and tested.
- Dynamic batching can silently change output splitting if not constrained.
- LLM scheduling is not the same as tensor batching; llama.cpp and Cactus need
  explicit token, context, cancellation, and memory-pressure policies.
- KServe V2 string/BYTES conventions for LLMs need documentation.
- Pulling server dependencies into `neuriplo` core would create dependency
  creep if the boundary is not enforced.
- `vision-inference` removing direct `neuriplo` linkage is a larger migration
  than adding a remote backend adapter.
