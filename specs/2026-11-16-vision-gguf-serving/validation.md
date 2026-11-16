# Feature Validation — vision-GGUF serving on LLM-kind backends

Executed on 2026-11-16/17 against `neuriplo` `feature/ggml-rfdetr` (GGML backend with
the RF-DETR and RF-DETR-seg slices) and this branch.

## Automated

- [x] [V-1] -> [R: task routing] `ctest --preset debug`: 399/399 pass, including
      the new `backend_registry_vision_task_selects_tensor_path`,
      `backend_registry_empty_task_keeps_legacy_routing`,
      `stub_executor_vision_task_selects_tensor_metadata`,
      `parse_runtime_config_task_type_env_and_cli`,
      `parse_runtime_config_task_type_defaults_empty`,
      `parse_runtime_config_rejects_llm_strategy_with_task_type`,
      `parse_runtime_config_input_sizes_flag_and_env`,
      `parse_runtime_config_rejects_malformed_input_sizes`,
      `admin_endpoint_load_accepts_task_type`, and
      `admin_endpoint_load_rejects_llm_strategy_with_task_type`.
- [x] `scripts/check-format.sh` clean; `cmake --preset lint` builds clean.
- [x] [V-2] -> [R: real GGUF serving] `cmake --preset real-ggml` with
      `-DNEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR=<neuriplo checkout>` builds, and
      `scripts/e2e-ggml-rfdetr.sh` passes 6/6:
  - model loads and `/v2/health/ready` is true;
  - `/v2/models/rfdetr` reports `"platform":"neuriplo_ggml"` with `input1` and
    `output0`, and carries no `prompt` tensor;
  - `/v2/models/rfdetr/infer` returns `output0 [1,300,4]` and `output1 [1,300,91]`;
  - RF-DETR-seg returns `output0 [1,200,4]` and `output1 [1,200,108,108]`.
- [x] Stub-backend regression: the existing 317 tests (scheduler, admin, pipeline,
      LLM path, repository, gRPC codecs) pass unchanged.

## Manual

- [x] [V-3] -> [R: CLI validation] `--scheduler-strategy llm --task-type rfdetr`
      fails at startup with
      `task-type selects tensor serving and cannot be combined with --scheduler-strategy llm`.
- [x] [V-4] -> [R: metadata contract] With `--task-type rfdetr`, `/v2/models/rfdetr`
      advertises the GGML engine contract (`input1`, 4-D `[1,3,560,560]`) rather
      than BYTES `prompt`; without a task the GGUF LLM behavior is unchanged.
- [x] Hyperlinks in the touched docs resolve (relative paths checked with `ls`).

## Deviations

- The first published branch (`feature/ggml-vision-tensor-routing`, PR #31) was cut
  from a stale local `develop` that had diverged 89 commits from `origin/develop`;
  its PR diff contained 164 files instead of 19. The change was re-applied onto
  `origin/develop` on `feature/ggml-vision-serving-v2`, and the packet was written
  in the repository's current `specs/` layout rather than the removed `plan/` one.
- The repository scanner still cannot discover `.gguf` files (no extension
  mapping). Explicitly deferred in [requirements.md](requirements.md#out-of-scope)
  and recorded in `docs/model-repository.md`.
- `real-ggml` is a local preset; CI does not build the GGML backend (no
  `real-ggml` job in `ci.yml`), so the E2E above is local evidence only.

## Traceability

| Requirement | Status | Evidence |
| --- | --- | --- |
| Tensor path for a vision `task_type` | Met | V-1, V-3 |
| Real GGUF detection/segmentation serving | Met | V-2 |
| Metadata from the engine, not hardcoded names | Met | V-4, E2E metadata checks |
| Static input geometry available from the CLI | Met | V-1 (`--input-sizes` cases) |
| Repository-scan GGUF discovery | Deferred | Deviation above |
