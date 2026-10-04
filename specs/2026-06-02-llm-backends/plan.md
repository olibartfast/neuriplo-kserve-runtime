# Step 8: LLM Backends (llama.cpp and Cactus) - plan

> Retrospective packet. Step 8 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 8
> snapshot (`plan/STEP8.md`, later `specs/history/steps/STEP8.md`) and the
> Step 8 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Tasks are the step's substeps. Each `T-` section is the implementation
record from the snapshot.

### [T-1] Step 8.1: LLM Configuration

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

### [T-2] Step 8.2: BYTES Tensor Convention

- Extended `KServeV2Codec` to parse and serialize `"datatype": "BYTES"` tensors.
- Added `string_data` payloads to `InputTensor` and `OutputTensor`.
- Added `LlmGenerationParams` on `InferenceRequest` / `ExecutionRequest` parsed from the
  KServe `parameters` object.
- Numeric tensor behavior is unchanged; empty numeric `data` arrays remain valid.
- Dynamic batching rejects BYTES tensors via `BatchCompatibility`.

### [T-3] Step 8.3: Backend Capability Registry

- Added `BackendRegistry` keyed by backend id with tensor vs LLM classification.
- Classified `llamacpp`, `cactus`, and `ggml` as LLM/token backends.
- Existing tensor backends and `stub` remain registered; unknown ids fail predictably.
- `ModelRegistry` factory selection flows through the registry instead of a hardcoded
  `isNeuriploBackend()` whitelist.

### [T-4] Step 8.4: Token-Aware Scheduler Strategy

- Added `LlmScheduler` as a separate `Scheduler` implementation from `ModelScheduler`.
- Does not reuse `DynamicBatcher` merge/split logic.
- Enforces prompt presence, context length, generation parameter bounds, and KV cache slot
  admission.
- Scheduler selection uses `--scheduler-strategy llm` or auto-detection for LLM backends.

### [T-5] Step 8.5: Non-Streaming LLM KServe V2 Inference

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
