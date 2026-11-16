# Feature Requirements — vision-GGUF serving on LLM-kind backends

Roadmap phase: [Phase 6 — Vision GGUF serving](../roadmap.md)
Branch: `feature/ggml-vision-serving-v2` (from `develop`, PR back to `develop`)

## Goal

Serve a vision GGUF model (RF-DETR detection or segmentation weights produced by
`convert_onnx_to_gguf.py`) through the neuriplo GGML backend with KServe V2 tensor
I/O. Before this branch the `ggml` backend id was registered as `BackendKind::Llm`,
so a vision model was forced down the LLM path: fake BYTES prompt/text metadata and
token-generation scheduling. A vision model needs the tensor scheduler and the
engine's real tensor contract.

## In Scope

- `--task-type` / `TASK_TYPE` (`RuntimeConfig::task_type`): a neuriplo-tasks
  model-type string (`rfdetr`, `rfdetrseg`, `yolo26`, ...). Non-empty selects the
  tensor serving path on any backend, including LLM-kind ones.
- Admin load/reload bodies accept `task_type`; the merged config is validated the
  same way the CLI validates it.
- CLI `--input-sizes` / `INPUT_SIZES` for backends that need static geometry
  (the GGML backend throws when input sizes are empty).
- `real-ggml` configure/build/test presets and `scripts/e2e-ggml-rfdetr.sh`.
- Documentation: README (dual-use `ggml`, flag list), `docs/model-repository.md`
  (GGUF is not discoverable by the repository scanner), this packet and the
  roadmap phase.

## Out of Scope

- Repository-scan discovery of `.gguf` files. `extensionBackends()` has no `.gguf`
  entry, so a version directory holding only a GGUF is skipped; vision GGUF is
  served with `--model-path` or an explicit admin load. Deferred to the
  repository-scan follow-up in [Phase 4](../roadmap.md).
- Pipeline model steps. `PipelineStepConfig::task_type` already exists for
  pre/postprocess steps; model steps inherit the model's own serving config.
- LLaMA / Cactus text models: unchanged, still LLM-kind and still on the LLM path.
- Making `ggml` a `BackendKind::Tensor` backend globally. That would break the
  existing GGUF LLM serving path (llama.cpp-derived GGUF chat models); the backend
  stays LLM-kind and the task selects the serving path per model.

## Decisions

- **Task-selected routing over re-classifying the backend.** `BackendKind`
  describes what the backend can do, not what a given model is. `ggml` genuinely
  serves both text models (LLM path) and vision models (tensor path), so the
  per-model `task_type` is the discriminator. Implemented as an overload of
  `usesLlmScheduler(strategy, backend, task_type)`; the two-argument form is
  unchanged so every existing call site and test keeps its behavior.
- **CLI rejects `--scheduler-strategy llm` + `--task-type`.** A vision task cannot
  be served as token generation; failing at startup is better than silently
  scheduling it on the wrong path. The admin codec applies the same rule to the
  merged config.
- **Metadata from the engine, not hardcoded names.** With a task set,
  `RealNeuriploAdapter` advertises `engine_->get_inference_metadata()` (the GGML
  loader's `input1`/`output0..N` contract) instead of the BYTES prompt/text
  convention. This follows the "Model Metadata From Model Or config.pbtxt — Never
  Backend Hardcoding" rule in `AGENTS.md`: no model-specific names are added to the
  runtime or the backend.

## Constraints and Context

- Ownership: the loader and its I/O contract stay in `neuriplo`
  (`backends/ggml/src/GGMLInfer.cpp`); the runtime only chooses the scheduling and
  metadata path.
- Contract surfaces touched: CLI flags and environment names (`--task-type`,
  `--input-sizes`, `TASK_TYPE`, `INPUT_SIZES`), the admin load/reload body
  (`task_type`), and which metadata shape `/v2/models/<name>` reports for a
  GGML model with a task set.
- Fixed batch-1 geometry is the backend's own rule (RF-DETR 560, RF-DETR-seg 432);
  the runtime passes shapes through unchanged.
- Cross-repo: the GGUF weights and the RF-DETR/seg slices come from the `neuriplo`
  `feature/ggml-rfdetr` work and the `neuriplo-tasks` export docs; no runtime
  change is needed in those repos.
