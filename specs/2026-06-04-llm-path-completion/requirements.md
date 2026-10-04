# Step 10: LLM Path Completion - requirements

> Retrospective packet. Step 10 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 10
> snapshot (`plan/STEP10.md`, later `specs/history/steps/STEP10.md`) and the
> Step 10 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Roadmap phase: [Phase 0 - Serving Foundation](../roadmap.md#phase-0---serving-foundation-steps-0-11)
Status: **Complete**
Specified: 2026-06-04 (snapshot date)

## Goal

**Completed:** 2026-06-04
**Snapshot:** All 6 substeps implemented and tested.

Completed the LLM production path: real backend execution scaffolding with generation parameters, token-accurate context enforcement, cancellation and deadline propagation, KV-cache memory pressure admission, SSE streaming responses, and OpenAI-compatible endpoints for chat completions and embeddings.

## In Scope

Step 10 addresses the remaining Step 8 gaps and closes the LLM production
path. Each substep builds on Step 8 scaffolding to reach a working
llama.cpp/Cactus decode with streaming, cancellation, and OpenAI endpoints.

### 10.1 Real LLM Backend Execution

- Wire real llama.cpp and Cactus decode through `NeuriploExecutor`.
- Pass generation parameters (`max_tokens`, `temperature`, `top_p`, `top_k`)
  into backend calls.
- Validate token-by-token decode loops and KV cache management against
  real hardware.

### 10.2 Token-Accurate Context Enforcement

- Replace the character-count proxy in `LlmScheduler` with real
  tokenization.
- Enforce context-length limits at admission time.
- Reject requests exceeding available context window.

### 10.3 Cancellation and Deadline Propagation

- Propagate cancel/deadline through `LlmScheduler` into backend decode
  loops.
- Abandon slow decode when the client deadline expires.
- Ensure cancelled work does not hold KV cache slots or executor threads.

### 10.4 KV-Cache Memory Pressure Policy

- Add admission control that ties KV cache slot occupancy to available
  memory.
- Reject or queue requests when context length × active slots exceeds
  a configurable threshold.
- Expose KV-cache slot utilization in metrics.

### 10.5 Streaming Responses

- Implement SSE or chunked transfer encoding for token-by-token output.
- Keep KServe V2 response shape stable for non-streaming path.
- Gate streaming per-request via `parameters.stream`.

### 10.6 OpenAI-Compatible Endpoints

- Add `/v1/completions`, `/v1/chat/completions`, and `/v1/embeddings`
  only for LLM models.
- Share model lifecycle, metrics, timeouts, and cancellation with KServe
  V2 path.
- Map OpenAI request/response shapes to internal LLM execution types.

## Requirements

Each requirement is an exit criterion of the original step plan; the
`(n.m)` tag names the substep that owned it.

- [R-1] (10.1) A real llama.cpp or Cactus model produces text through `/v2/models/{model}/infer`.
- [R-2] (10.1) Backend metadata (context length, tokenizer) comes from neuriplo.
- [R-3] (10.2) Context enforcement is accurate for multi-byte and non-English text.
- [R-4] (10.2) Admission rejects over-context requests with a stable error code.
- [R-5] (10.3) A deliberately slow decode loop is abandoned on client timeout.
- [R-6] (10.3) KV cache slots and inflight counters are released correctly on cancel.
- [R-7] (10.4) Memory pressure blocks new requests before OOM.
- [R-8] (10.4) KV-cache metrics are exported on `/metrics`.
- [R-9] (10.5) Streaming tokens are delivered progressively to the client.
- [R-10] (10.5) Non-streaming path is unchanged.
- [R-11] (10.6) A llama.cpp-backed model responds to `/v1/chat/completions` with a valid OpenAI-shaped response.
- [R-12] (10.6) Tensor-only models do not expose OpenAI endpoints.

Not every requirement was fully met; the delivered status of each is in
[`validation.md`](validation.md#deviations).

## Out of Scope

Nothing recorded beyond the later steps in the target design.
