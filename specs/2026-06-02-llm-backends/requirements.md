# Step 8: LLM Backends (llama.cpp and Cactus) - requirements

> Retrospective packet. Step 8 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 8
> snapshot (`plan/STEP8.md`, later `specs/history/steps/STEP8.md`) and the
> Step 8 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Roadmap phase: [Phase 0 - Serving Foundation](../roadmap.md#phase-0---serving-foundation-steps-0-11)
Status: **Complete**
Specified: 2026-06-02 (snapshot date)

## Goal

This snapshot records the Step 8 implementation state through substeps 8.1–8.5.
Steps 6–7 established dynamic tensor batching and observability; Step 8 adds LLM/token-aware
scheduling, BYTES-tensor prompt conventions, backend capability classification, and
non-streaming KServe V2 LLM inference scaffolding for llama.cpp and Cactus backends through
neuriplo.

## In Scope

- Add LLM model config for llama.cpp and Cactus backends.
- Add GGUF/local model artifact conventions where applicable.
- Add backend selection for llama.cpp vs Cactus through neuriplo.
- Add token-aware scheduler strategy separate from tensor dynamic batching.
- Add prefill, decode, cancellation, and deadline propagation into backend
  calls.
- Add KServe V2 `BYTES` prompt convention.
- Add streaming response design for generated tokens.
- Add OpenAI-compatible `/v1/completions`, `/v1/chat/completions`, and
  `/v1/embeddings` only for LLM models.

## Requirements

Each requirement is an exit criterion of the original step plan; the
`(n.m)` tag names the substep that owned it.

- [R-1] llama.cpp and Cactus completion paths work through `/v2/models/{model}/infer`.
- [R-2] Context limits and generation parameters are enforced.
- [R-3] Cancellation/deadline behavior is defined and tested.
- [R-4] LLM scheduling does not reuse tensor batch merge/split logic.

Not every requirement was fully met; the delivered status of each is in
[`validation.md`](validation.md#deviations).

## Out of Scope

Recorded at the end of the step as remaining gaps; later steps picked them
up as noted in [`../roadmap.md`](../roadmap.md).

### Remaining Work

- **8.6** OpenAI-compatible `/v1/completions`, `/v1/chat/completions`, `/v1/embeddings`
- **8.7** Cancellation and deadline propagation into backend decode loops
- **8.8** Streaming responses (SSE/chunked transfer)
- **8.9** Full docs refresh once OpenAI endpoints and streaming land
- Real llama.cpp/Cactus decode through `NeuriploExecutor` (currently scaffold + stub path)
- Token-accurate context enforcement (current scaffold uses character-count proxy)
- Memory pressure policy for context length × KV slots
