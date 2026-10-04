# Step 10: LLM Path Completion - validation

> Retrospective packet. Step 10 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 10
> snapshot (`plan/STEP10.md`, later `specs/history/steps/STEP10.md`) and the
> Step 10 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Validation was recorded at completion rather than written before the code;
the checks below are the ones the step snapshot reports as run.

## Checks

- [x] [V-1] -> [R-4], [R-5], [R-6], [R-7], [R-8], [R-9], [R-10]: the step's build, format and test commands (Recorded checks below) pass

## Traceability

| Requirement | Status | Evidence |
| --- | --- | --- |
| R-1 | Partial | substep 10.1 tests and exit criteria below |
| R-2 | Partial | substep 10.1 tests and exit criteria below |
| R-3 | Not met | substep 10.2 tests and exit criteria below |
| R-4 | Met | substep 10.2 tests and exit criteria below |
| R-5 | Met | substep 10.3 tests and exit criteria below |
| R-6 | Met | substep 10.3 tests and exit criteria below |
| R-7 | Met | substep 10.4 tests and exit criteria below |
| R-8 | Met | substep 10.4 tests and exit criteria below |
| R-9 | Met | substep 10.5 tests and exit criteria below |
| R-10 | Met | substep 10.5 tests and exit criteria below |
| R-11 | Partial | substep 10.6 tests and exit criteria below |
| R-12 | Not met | substep 10.6 tests and exit criteria below |

## Deviations

Where the snapshot shows the step delivered less than the planned
exit criterion. These are recorded from the snapshot text, not re-tested.

- [R-1] Partial: `RealNeuriploAdapter::llmInfer()` delegates to the tensor `infer()` path;
   the LLM path is tested with the stub executor, not a real llama.cpp or Cactus model.
- [R-2] Partial: metadata flows through the adapter boundary; context length and tokenizer
   do not come from the backend.
- [R-3] Not met: `CharRatioTokenizer` and `WhitespaceTokenizer` are estimates, not real
   tokenization.
- [R-11] Partial: tested with the stub executor, not a llama.cpp-backed model.
- [R-12] Not met: the snapshot records that tensor-only models can be queried through
   `/v1/completions` and `/v1/embeddings`.

## Recorded checks

### 10.1 Real LLM Backend Execution - Tests Added

`tests/LlmPathTest.cpp` — LLm path integration tests verifying KServe V2 infer with metadata, completions, chat completions, and embeddings.

### 10.2 Token-Accurate Context Enforcement - Tests Added

`tests/TokenizerTest.cpp` — unit tests for `CharRatioTokenizer` and `WhitespaceTokenizer`.
Updated `tests/LlmSchedulerTest.cpp` — tests for context length enforcement, token-based validation, custom tokenizer integration.

### 10.3 Cancellation and Deadline Propagation - Exit Criteria Met

- Deliberately slow decode loops (`SlowLlmExecutor` tests) are abandoned on timeout.
- Cancel token is stored on requests and checked in `processSingle()`.
- KV cache slots and inflight counters are released on cancel path (`releaseDecodeSlot()`, `decrementInflight()`).

### 10.4 KV-Cache Memory Pressure Policy - Tests Added

Updated `tests/LlmSchedulerTest.cpp` — tests for KV cache metrics reporting and memory pressure rejection.

### 10.6 OpenAI-Compatible Endpoints - Tests Added

`tests/OpenAiCodecTest.cpp` — comprehensive codec tests for all 3 endpoint parsers and serializers.
`tests/LlmPathTest.cpp` — integration tests for completions, chat completions, embeddings, context-length rejection, and KServe V2 metadata.

### Validation

```bash
cmake --build --preset debug && ctest --preset debug
# 100% tests passed, 0 tests failed
scripts/check-format.sh  # passes
```

### Exit Criteria

#### 10.1 Real LLM Backend Execution

- ✅ `NeuriploExecutor::inferStreaming()` routes LLM requests to `adapter_->llmInfer()` with generation parameters (`max_tokens`, `temperature`, `top_p`, `top_k`).
- ✅ Stub executor returns deterministic text with `LlmResultMetadata` including prompt/completion tokens and finish reason.
- ✅ Backend metadata flows through adapter boundary for LLM path.

#### 10.2 Token-Accurate Context Enforcement

- ✅ `Tokenizer` interface replaces character-count proxy: `CharRatioTokenizer` (default) and `WhitespaceTokenizer` implementations.
- ✅ Context-length checked at admission using `tokenizer_->countTokens()`.
- ✅ Combined context check: `prompt_tokens + max_tokens > context_length` rejects over-context requests.
- ✅ Custom tokenizer can be injected via `makeLlmScheduler()` overload.

#### 10.3 Cancellation and Deadline Propagation

- ✅ Cancel token propagated through `ExecutionRequest.cancel_token` into executor and adapter layers.
- ✅ `LlmScheduler::processSingle()` checks cancellation before acquire, after deadline, and after inference.
- ✅ `SlowLlmExecutor` tests verify timeout-based cancellation.
- ✅ KV cache slots and inflight counters released on all cancel/timeout paths.

#### 10.4 KV-Cache Memory Pressure Policy

- ✅ `--memory-budget-bytes` CLI flag controls memory-based admission.
- ✅ Requests rejected when `estimated_context_bytes × active_decodes > memory_budget_bytes`.
- ✅ `neuriplo_kv_cache_slots_total`, `neuriplo_kv_cache_slots_active`, and `neuriplo_scheduler_requests_memory_pressure_rejected_total` exported on `/metrics`.

#### 10.5 Streaming Responses

- ✅ SSE response path in `HttpServer`: `Content-Type: text/event-stream`, chunked token output via `StreamWriter`.
- ✅ `StreamingTokenCallback` interface for executor token-by-token output.
- ✅ KServe V2 non-streaming path unchanged.
- ✅ OpenAI streaming gated per-request via `stream: true`.

#### 10.6 OpenAI-Compatible Endpoints

- ✅ `POST /v1/completions` returns OpenAI-shaped response with `object`, `choices`, `usage`.
- ✅ `POST /v1/chat/completions` returns `chat.completion` response with `message.role` and `message.content`.
- ✅ `POST /v1/embeddings` returns `list` response with embedding vectors and usage.
- ✅ All endpoints include real token counts from `LlmResultMetadata` when available.
- ✅ Tensor-only models can be queried through `/v1/completions` and `/v1/embeddings` (they execute through the scheduler regardless of backend kind).

## Evidence

| Date | Check | Result |
| --- | --- | --- |
| 2026-06-04 | Recorded checks above | Pass, as reported by the Step 10 snapshot |
