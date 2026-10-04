# Step 8: LLM Backends (llama.cpp and Cactus) - validation

> Retrospective packet. Step 8 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 8
> snapshot (`plan/STEP8.md`, later `specs/history/steps/STEP8.md`) and the
> Step 8 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Validation was recorded at completion rather than written before the code;
the checks below are the ones the step snapshot reports as run.

## Checks

- [x] [V-1] -> [R-4]: the step's build, format and test commands (Recorded checks below) pass

## Traceability

| Requirement | Status | Evidence |
| --- | --- | --- |
| R-1 | Partial | Recorded checks below |
| R-2 | Partial | Recorded checks below |
| R-3 | Not met | Recorded checks below |
| R-4 | Met | Recorded checks below |

## Deviations

Where the snapshot shows the step delivered less than the planned
exit criterion. These are recorded from the snapshot text, not re-tested.

- [R-1] Partial: scaffold and stub-executor path only; real decode was deferred to Step 10.
- [R-2] Partial: context length used a character-count proxy until the Step 10.2 tokenizer.
- [R-3] Not met: deferred as 8.7 and delivered in Step 10.3.

## Recorded checks

### Validation

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

## Evidence

| Date | Check | Result |
| --- | --- | --- |
| 2026-06-02 | Recorded checks above | Pass, as reported by the Step 8 snapshot |
