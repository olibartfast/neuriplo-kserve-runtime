# Step 8: LLM Backends (llama.cpp and Cactus)

This snapshot records the Step 8 implementation state through substeps 8.1–8.5.
Steps 6–7 established dynamic tensor batching and observability; Step 8 adds LLM/token-aware
scheduling, BYTES-tensor prompt conventions, backend capability classification, and
non-streaming KServe V2 LLM inference scaffolding for llama.cpp and Cactus backends through
neuriplo.

## Implemented Scope

### Step 8.1: LLM Configuration

- Added CLI flags and `RuntimeConfig` fields:
  - `--scheduler-strategy <tensor|llm>` (default `tensor`)
  - `--context-length <tokens>`
  - `--kv-cache-slots <count>`
  - `--max-tokens <count>`
  - `--temperature`, `--top-p`, `--top-k` sampling defaults
  - `--streaming-enabled true|false`
- Validation enforces allowed strategy values, positive context/slot/token limits,
  non-negative temperature, `top_p` in `(0, 1]`, and boolean streaming toggle.
- GGUF/local model artifacts follow the existing KServe `/mnt/models` convention documented
  in `README.md`.

### Step 8.2: BYTES Tensor Convention

- Extended `KServeV2Codec` to parse and serialize `"datatype": "BYTES"` tensors.
- Added `string_data` payloads to `InputTensor` and `OutputTensor`.
- Added `LlmGenerationParams` on `InferenceRequest` / `ExecutionRequest` parsed from the
  KServe `parameters` object.
- Numeric tensor behavior is unchanged; empty numeric `data` arrays remain valid.
- Dynamic batching rejects BYTES tensors via `BatchCompatibility`.

### Step 8.3: Backend Capability Registry

- Added `BackendRegistry` keyed by backend id with tensor vs LLM classification.
- Classified `llamacpp`, `cactus`, and `ggml` as LLM/token backends.
- Existing tensor backends and `stub` remain registered; unknown ids fail predictably.
- `ModelRegistry` factory selection flows through the registry instead of a hardcoded
  `isNeuriploBackend()` whitelist.

### Step 8.4: Token-Aware Scheduler Strategy

- Added `LlmScheduler` as a separate `Scheduler` implementation from `ModelScheduler`.
- Does not reuse `DynamicBatcher` merge/split logic.
- Enforces prompt presence, context length, generation parameter bounds, and KV cache slot
  admission.
- Scheduler selection uses `--scheduler-strategy llm` or auto-detection for LLM backends.

### Step 8.5: Non-Streaming LLM KServe V2 Inference

- Wired BYTES prompt requests through `LlmScheduler` when the LLM path is active.
- Stub executor returns deterministic BYTES `text` output for local LLM-path testing.
- KServe V2 infer route preserves request `parameters` through to the scheduler.

## Request Flow

```text
Tensor path (unchanged):
  KServeRuntime -> ModelRegistry -> ModelHandle -> ModelScheduler -> DynamicBatcher -> Executor

LLM path (new):
  KServeRuntime -> ModelRegistry -> ModelHandle -> LlmScheduler -> Executor
```

Tensor and LLM paths share model lifecycle, metrics, error taxonomy, structured logging, and
trace spans. They do not share batch formation or tensor batch compatibility checks.

## Validation

All local checks run successfully:

```bash
cmake --preset debug
cmake --build --preset debug
scripts/check-format.sh
ctest --preset debug
```

Test coverage added for:
- `RuntimeConfigTest`: LLM CLI flags and validation
- `KServeV2CodecTest`: BYTES parse/serialize round-trip
- `BackendRegistryTest`: registration, capability query, scheduler selection
- `LlmSchedulerTest`: admission, context limits, KV slot behavior
- `ModelRegistryTest`: LLM scheduler wiring with stub backend

## Remaining Work

- **8.6** OpenAI-compatible `/v1/completions`, `/v1/chat/completions`, `/v1/embeddings`
- **8.7** Cancellation and deadline propagation into backend decode loops
- **8.8** Streaming responses (SSE/chunked transfer)
- **8.9** Full docs refresh once OpenAI endpoints and streaming land
- Real llama.cpp/Cactus decode through `NeuriploExecutor` (currently scaffold + stub path)
- Token-accurate context enforcement (current scaffold uses character-count proxy)
- Memory pressure policy for context length × KV slots
