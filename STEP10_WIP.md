# Step 10: LLM Path Completion (WIP)

This is a work-in-progress planning document. See `ROADMAP.md` Step 10 for
the full substep breakdown and exit criteria.

## Status

Not started. All substeps (10.1–10.6) are pending.

## Substeps

| # | Item | Status |
|---|------|--------|
| 10.1 | Real LLM backend execution (llama.cpp/Cactus) | Pending |
| 10.2 | Token-accurate context enforcement | Pending |
| 10.3 | Cancellation and deadline propagation into decode | Pending |
| 10.4 | KV-cache memory pressure policy | Pending |
| 10.5 | Streaming responses (SSE/chunked) | Pending |
| 10.6 | OpenAI-compatible endpoints | Pending |

## Dependencies

- 10.1 (real backend execution) gates 10.2–10.6; real decode must work first.
- 10.3 (cancellation) gates 10.5 (streaming) — streaming without cancellation
  is unsafe.
- 10.6 (OpenAI endpoints) can be built in parallel with 10.2–10.5 once 10.1
  lands.

## Notes

- Step 8 scaffolding (BYTES tensors, LlmScheduler, BackendRegistry) is in
  place.
- Stub executor path remains available for local dev without real hardware.
