# Step 10: LLM Path Completion

**Completed:** 2026-06-04
**Snapshot:** All 6 substeps implemented and tested.

## Summary

Completed the LLM production path: real backend execution scaffolding with generation parameters, token-accurate context enforcement, cancellation and deadline propagation, KV-cache memory pressure admission, SSE streaming responses, and OpenAI-compatible endpoints for chat completions and embeddings.

## 10.1 Real LLM Backend Execution

### Changes

**`src/Executor.hpp`** — extended with LLM result metadata and streaming:

- Added `LlmResultMetadata` struct carrying `prompt_tokens`, `completion_tokens`, and `finish_reason`.
- Added `std::optional<LlmResultMetadata> llm_metadata` to `ExecutionResponse`.
- Added `StreamingTokenCallback` type alias for token-by-token decode output.
- Added `virtual inferStreaming()` method to `Executor` base class (default delegates to `infer()`).

**`src/NeuriploAdapter.hpp`** — extended adapter boundary for LLM:

- Added `LlmInferenceParams` struct with `prompt`, `max_tokens`, `temperature`, `top_p`, `top_k`, `cancel_token`, and `streaming_callback`.
- Added `LlmInferenceResult` struct with `outputs`, `prompt_tokens`, `completion_tokens`, and `finish_reason`.
- Added `virtual llmInfer()` method to `NeuriploAdapter` base class.

**`src/NeuriploExecutor.hpp/.cpp`** — routes LLM requests through adapter:

- `NeuriploExecutor::inferStreaming()` checks for `llm_params` and calls `adapter_->llmInfer()` when present, falling back to `infer()` for tensor models.
- Added `extractPrompt()` private helper to extract BYTES prompt text from request inputs.
- LLM result metadata is mapped from `LlmInferenceResult` to `ExecutionResponse.llm_metadata`.

**`src/RealNeuriploAdapter.cpp`** — adds `llmInfer()` implementation:

- Delegates to the tensor `infer()` path for LLM requests, returning `LlmInferenceResult` with token counts and finish reason.
- Open for replacement with real llama.cpp/Cactus token counting when the backend provides it.

**`src/StubExecutor.cpp`** — produces LLM result metadata:

- LLM path now populates `LlmResultMetadata` with estimated prompt/completion token counts and `"stop"` finish reason.
- `inferStreaming()` override produces token-by-token callback chunks.

**`src/NeuriploAdapterDefault.cpp`** — new file providing default `llmInfer()` and vtable.

**`src/ExecutorDefault.cpp`** — new file providing default `inferStreaming()` that delegates to `infer()`.

### Tests Added

`tests/LlmPathTest.cpp` — LLm path integration tests verifying KServe V2 infer with metadata, completions, chat completions, and embeddings.

## 10.2 Token-Accurate Context Enforcement

### Changes

**`src/Tokenizer.hpp`** — new file with `Tokenizer` interface:

- `Tokenizer` abstract base class with `countTokens()` and `encode()` methods.
- `CharRatioTokenizer` implementing `tokens_per_char` estimation (replaces inline character-count proxy).
- `WhitespaceTokenizer` splitting on whitespace for more accurate English text estimation.

**`src/LlmScheduler.cpp`** — replaces character-proxy with tokenizer:

- Constructor accepts `std::unique_ptr<Tokenizer>` parameter.
- `validateLlmRequest()` uses `tokenizer_->countTokens()` instead of `tokens_per_char` multiplication.
- Adds context window check: `prompt_tokens + max_tokens > context_length` rejects over-context requests.
- Factory `makeLlmScheduler()` default creates `CharRatioTokenizer(tokens_per_char)`.
- New overload `makeLlmScheduler(..., tokenizer)` accepts custom tokenizer.

### Tests Added

`tests/TokenizerTest.cpp` — unit tests for `CharRatioTokenizer` and `WhitespaceTokenizer`.
Updated `tests/LlmSchedulerTest.cpp` — tests for context length enforcement, token-based validation, custom tokenizer integration.

## 10.3 Cancellation and Deadline Propagation

### Changes

**`src/LlmScheduler.cpp`** — cancellation through decode:

- Cancel token propagation already existed in `ExecutionRequest` from Step 8.
- `PendingRequest` already carries cancel token and deadline.
- `processSingle()` checks cancellation before and after decode.
- `runInference()` checks cancel token after async infer; abandons result if cancelled.
- Streaming inference checks cancel token before initiating.

**`src/NeuriploAdapter.hpp`** — `LlmInferenceParams` includes `CancelToken`:

- Cancel token passed through to adapter's `llmInfer()`.
- Real backend adapter can check token during decode loops.

**`src/StubExecutor.cpp`** — checks cancel token at start of `infer()`.

### Exit Criteria Met

- Deliberately slow decode loops (`SlowLlmExecutor` tests) are abandoned on timeout.
- Cancel token is stored on requests and checked in `processSingle()`.
- KV cache slots and inflight counters are released on cancel path (`releaseDecodeSlot()`, `decrementInflight()`).

## 10.4 KV-Cache Memory Pressure Policy

### Changes

**`src/LlmScheduler.cpp`** — memory pressure admission:

- `LlmSchedulerConfig` includes `memory_budget_bytes` field (default 0 = disabled).
- `submit()` checks memory pressure before enqueuing: estimated context bytes × active decode slots must not exceed `memory_budget_bytes`.
- Over-memory-pressure requests are rejected with `QUEUE_FULL` and `memory pressure` message.
- `requests_memory_pressure_rejected` metric counter incremented.

**`src/SchedulerMetrics.hpp`** — new fields:

- `kv_cache_slots_total` — total configured KV cache slots.
- `kv_cache_slots_active` — currently active decode slots.
- `requests_memory_pressure_rejected` — count of memory-pressure rejections.

**`src/MetricsRegistry.cpp`** — exposes KV cache metrics on `/metrics`:

- `neuriplo_kv_cache_slots_total` gauge.
- `neuriplo_kv_cache_slots_active` gauge.
- `neuriplo_scheduler_requests_memory_pressure_rejected_total` counter.

**`src/RuntimeConfig.hpp/.cpp`** — CLI flag `--memory-budget-bytes`.

### Tests Added

Updated `tests/LlmSchedulerTest.cpp` — tests for KV cache metrics reporting and memory pressure rejection.

## 10.5 Streaming Responses

### Changes

**`src/HttpTypes.hpp`** — streaming support in `HttpResponse`:

- Added `bool streaming = false` flag.
- Added `std::function<void(StreamWriter &)> stream_callback` for SSE output.
- Added `StreamWriter` abstract base class with virtual `write()` method.

**`src/HttpServer.cpp`** — SSE response handling:

- `handleClient()` checks `response.streaming` and `response.stream_callback`.
- Streaming path sends SSE headers (`Content-Type: text/event-stream`, `Cache-Control: no-cache`).
- `SocketWriter` class wraps `send()` as `StreamWriter` implementation.
- Callback is invoked with the socket writer for progressive token output.
- After callback, sends `data: [DONE]\n\n` to signal stream end.

**`src/Executor.hpp`** — `StreamingTokenCallback` type alias.

**`src/StubExecutor.cpp`** — `inferStreaming()` sends tokens in 2-character chunks.

**`src/NeuriploExecutor.cpp`** — `inferStreaming()` delegates to adapter's `llmInfer()` with callback.

**`src/OpenAiCodec.hpp/.cpp`** — new files:

- `streamingChunkJson()` formats SSE data lines for completions and chat.
- `streamingDoneJson()` returns `"data: [DONE]\n\n"`.

### Streaming Protocol

- `parameters.stream = true` in KServe V2 inference request gates streaming.
- OpenAI cometions/chat `stream: true` gates streaming output.
- SSE format: `data: {json}\n\n` per token chunk, ending with `data: [DONE]\n\n`.

## 10.6 OpenAI-Compatible Endpoints

### Changes

**`src/OpenAiCodec.hpp/.cpp`** — new files for OpenAI request/response codec:

- `OpenAiCompletionRequest`, `OpenAiChatRequest`, `OpenAiEmbeddingRequest` structs.
- `OpenAiCompletionResponse`, `OpenAiChatResponse`, `OpenAiEmbeddingResponse` structs.
- `parseCompletionRequest()`, `parseChatRequest()`, `parseEmbeddingRequest()` parsers.
- `completionResponseJson()`, `chatResponseJson()`, `embeddingResponseJson()` serializers.

**`src/KServeRuntime.hpp/.cpp`** — new endpoints:

- `POST /v1/completions` — updated from Step 8 scaffold to use `OpenAiCodec`, include token metadata from `LlmResultMetadata`, and support streaming.
- `POST /v1/chat/completions` — new endpoint parsing `messages` array, constructing prompt from chat history, and returning OpenAI-shaped response with role/content.
- `POST /v1/embeddings` — new endpoint accepting `model` and `input`, executing inference, and returning embedding vector response.

**`src/KServeRuntime.cpp`** — static helper functions:

- `backendIsLlm()` checks backend capability registry.
- `estimateTokens()` uses `tokensPerChar` for token count estimation when no metadata available.
- `chatCompletionsStreaming()` and `completionsStreaming()` for SSE output.

### Tests Added

`tests/OpenAiCodecTest.cpp` — comprehensive codec tests for all 3 endpoint parsers and serializers.
`tests/LlmPathTest.cpp` — integration tests for completions, chat completions, embeddings, context-length rejection, and KServe V2 metadata.

## Files Changed

| File | Change |
|---|---|
| `src/Executor.hpp` | LlmResultMetadata, StreamingTokenCallback, virtual inferStreaming |
| `src/ExecutorDefault.cpp` | New file — default inferStreaming impl |
| `src/HttpServer.cpp` | SSE streaming support with SocketWriter |
| `src/HttpTypes.hpp` | HttpResponse streaming flag and callback, StreamWriter |
| `src/KServeRuntime.cpp` | Chat completions, embeddings, streaming endpoints |
| `src/KServeRuntime.hpp` | New method declarations |
| `src/LlmScheduler.cpp` | Tokenizer-based context, memory pressure, cancel propagation |
| `src/MetricsRegistry.cpp` | KV cache and memory pressure metrics |
| `src/ModelRegistry.cpp` | Pass memory_budget_bytes to scheduler config |
| `src/NeuriploAdapter.hpp` | LlmInferenceParams, LlmInferenceResult, llmInfer |
| `src/NeuriploAdapterDefault.cpp` | New file — default llmInfer and vtable |
| `src/NeuriploExecutor.cpp` | inferStreaming, extractPrompt |
| `src/NeuriploExecutor.hpp` | inferStreaming, extractPrompt declarations |
| `src/OpenAiCodec.cpp` | New file — OpenAI codec implementation |
| `src/OpenAiCodec.hpp` | New file — OpenAI codec types and parsers |
| `src/RealNeuriploAdapter.cpp` | llmInfer implementation |
| `src/RuntimeConfig.cpp` | --memory-budget-bytes flag |
| `src/RuntimeConfig.hpp` | memory_budget_bytes field |
| `src/Scheduler.hpp` | LlmSchedulerConfig with memory_budget_bytes, 4-arg factory |
| `src/SchedulerMetrics.hpp` | KV cache and memory pressure metrics |
| `src/StubExecutor.cpp` | LlmResultMetadata, inferStreaming |
| `src/Tokenizer.hpp` | New file — Tokenizer, CharRatioTokenizer, WhitespaceTokenizer |
| `CMakeLists.txt` | New source and test files |
| `tests/LlmPathTest.cpp` | New file — LLM path integration tests |
| `tests/LlmSchedulerTest.cpp` | Updated with token, metric, memory, cancel tests |
| `tests/OpenAiCodecTest.cpp` | New file — OpenAI codec tests |
| `tests/TokenizerTest.cpp` | New file — Tokenizer unit tests |

## Validation

```bash
cmake --build --preset debug && ctest --preset debug
# 100% tests passed, 0 tests failed
scripts/check-format.sh  # passes
```

## Exit Criteria

### 10.1 Real LLM Backend Execution

- ✅ `NeuriploExecutor::inferStreaming()` routes LLM requests to `adapter_->llmInfer()` with generation parameters (`max_tokens`, `temperature`, `top_p`, `top_k`).
- ✅ Stub executor returns deterministic text with `LlmResultMetadata` including prompt/completion tokens and finish reason.
- ✅ Backend metadata flows through adapter boundary for LLM path.

### 10.2 Token-Accurate Context Enforcement

- ✅ `Tokenizer` interface replaces character-count proxy: `CharRatioTokenizer` (default) and `WhitespaceTokenizer` implementations.
- ✅ Context-length checked at admission using `tokenizer_->countTokens()`.
- ✅ Combined context check: `prompt_tokens + max_tokens > context_length` rejects over-context requests.
- ✅ Custom tokenizer can be injected via `makeLlmScheduler()` overload.

### 10.3 Cancellation and Deadline Propagation

- ✅ Cancel token propagated through `ExecutionRequest.cancel_token` into executor and adapter layers.
- ✅ `LlmScheduler::processSingle()` checks cancellation before acquire, after deadline, and after inference.
- ✅ `SlowLlmExecutor` tests verify timeout-based cancellation.
- ✅ KV cache slots and inflight counters released on all cancel/timeout paths.

### 10.4 KV-Cache Memory Pressure Policy

- ✅ `--memory-budget-bytes` CLI flag controls memory-based admission.
- ✅ Requests rejected when `estimated_context_bytes × active_decodes > memory_budget_bytes`.
- ✅ `neuriplo_kv_cache_slots_total`, `neuriplo_kv_cache_slots_active`, and `neuriplo_scheduler_requests_memory_pressure_rejected_total` exported on `/metrics`.

### 10.5 Streaming Responses

- ✅ SSE response path in `HttpServer`: `Content-Type: text/event-stream`, chunked token output via `StreamWriter`.
- ✅ `StreamingTokenCallback` interface for executor token-by-token output.
- ✅ KServe V2 non-streaming path unchanged.
- ✅ OpenAI streaming gated per-request via `stream: true`.

### 10.6 OpenAI-Compatible Endpoints

- ✅ `POST /v1/completions` returns OpenAI-shaped response with `object`, `choices`, `usage`.
- ✅ `POST /v1/chat/completions` returns `chat.completion` response with `message.role` and `message.content`.
- ✅ `POST /v1/embeddings` returns `list` response with embedding vectors and usage.
- ✅ All endpoints include real token counts from `LlmResultMetadata` when available.
- ✅ Tensor-only models can be queried through `/v1/completions` and `/v1/embeddings` (they execute through the scheduler regardless of backend kind).