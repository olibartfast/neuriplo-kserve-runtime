# Feature Plan — vision-GGUF serving on LLM-kind backends

Derived from [requirements.md](requirements.md).

## Group 1 — Configuration

1. Add `RuntimeConfig::task_type`, `--task-type`, and `TASK_TYPE`.
2. Add `--input-sizes` / `INPUT_SIZES` with `;`-separated inputs and
   `,`-separated dimensions; reject empty groups, empty dimensions, and a trailing
   separator.

## Group 2 — Routing

3. Add the `task_type` overload of `usesLlmScheduler`; a non-empty task always
   selects the tensor path. Keep the two-argument form's behavior identical.
4. Use the overload in `ModelLifecycle::load` (scheduler choice) and
   `StubExecutor` (stub metadata), and skip the BYTES prompt/text metadata branch
   in `RealNeuriploAdapter::load` when a task is set, so metadata and scheduling
   cannot disagree.
5. Accept `task_type` in `AdminCodec::applyCommonFields` and reject the
   `scheduler_strategy == "llm"` + `task_type` combination there, mirroring the
   CLI check on the merged config.

## Group 3 — Contract and documentation

6. Register the `real-ggml` configure/build/test presets; add
   `scripts/e2e-ggml-rfdetr.sh` (detection always, segmentation when the weights
   are provided).
7. Update `README.md` (GGML dual use, `TASK_TYPE`/`INPUT_SIZES`, invocation
   example), `docs/model-repository.md` (scanner cannot discover GGUF), this
   packet, `specs/roadmap.md` Phase 6, `specs/README.md` packet index, and
   `CHANGELOG.md` under `[Unreleased]`.

## Group 4 — Verification

8. Unit tests: routing overload (task set, task empty, LLM-kind, LLM strategy
   rejection), config parsing (CLI, environment, defaults, malformed
   `--input-sizes`), admin body acceptance, stub metadata selection.
9. Run the checks in [validation.md](validation.md), including the real
   `real-ggml` preset against `neuriplo` `feature/ggml-rfdetr` weights.

> Note: this packet was written after the code was first validated on the branch,
> because the branch originally predated the `specs/` layout. The validation
> record below is the executed evidence, not a plan.
