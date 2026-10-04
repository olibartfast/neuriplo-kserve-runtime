# Step 9: Production Hardening - Tensor Path - validation

> Retrospective packet. Step 9 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 9
> snapshot (`plan/STEP9.md`, later `specs/history/steps/STEP9.md`) and the
> Step 9 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Validation was recorded at completion rather than written before the code;
the checks below are the ones the step snapshot reports as run.

## Checks

- [x] [V-1] -> [R-2], [R-3], [R-5], [R-6], [R-7], [R-8], [R-10], [R-11]: the step's build, format and test commands (Recorded checks below) pass

## Traceability

| Requirement | Status | Evidence |
| --- | --- | --- |
| R-1 | Partial | substep 9.1 tests and exit criteria below |
| R-2 | Met | substep 9.1 tests and exit criteria below |
| R-3 | Met | substep 9.2 tests and exit criteria below |
| R-4 | Partial | substep 9.2 tests and exit criteria below |
| R-5 | Met | substep 9.3 tests and exit criteria below |
| R-6 | Met | substep 9.3 tests and exit criteria below |
| R-7 | Met | substep 9.4 tests and exit criteria below |
| R-8 | Met | substep 9.4 tests and exit criteria below |
| R-9 | Partial | substep 9.5 tests and exit criteria below |
| R-10 | Met | substep 9.5 tests and exit criteria below |
| R-11 | Met | substep 9.6 tests and exit criteria below |
| R-12 | Partial | substep 9.6 tests and exit criteria below |

## Deviations

Where the snapshot shows the step delivered less than the planned
exit criterion. These are recorded from the snapshot text, not re-tested.

- [R-1] Partial: datatype preservation is tested through `NeuriploExecutor` with a fake
  adapter; `extractDatatypes()` still returned FP32 for real backends.
- [R-4] Partial: invalid transitions are rejected and tested; logging them is not recorded.
- [R-9] Partial: probes are documented in `deploy/kserve/README.md`; no run with a
  slow-loading model is recorded.
- [R-12] Partial: baselines are published in the snapshot; there is no automated
  regression gate.

## Recorded checks

### 9.1 Multi-Datatype Neuriplo Adapter - Tests Added

`tests/NeuriploExecutorTest.cpp`:
- `neuriplo_executor_preserves_int32_output_datatype` — verifies INT32 outputs propagate through NeuriploExecutor
- `neuriplo_executor_preserves_fp16_output_datatype` — verifies FP16 output preservation
- `neuriplo_executor_preserves_uint8_output_datatype` — verifies UINT8 output preservation

FakeNeuriploAdapter refactored to output metadata-driven tensors instead of hardcoded scores/labels.

### 9.2 Formal Model State Machine - Tests Added

`tests/ModelStateMachineTest.cpp` — 11 test cases:
- `state_machine_starts_unloaded` — default state is Unloaded
- `state_machine_normal_lifecycle` — full Unloaded→Loading→Ready→Unloading→Unloaded cycle
- `state_machine_load_failure` — Loading→Failed→reset→Unloaded
- `state_machine_unavailable_from_loading` — Loading→Unavailable
- `state_machine_unavailable_from_ready` — Ready→Unavailable→reset
- `state_machine_rejects_invalid_transitions` — no invalid jumps from Unloaded
- `state_machine_rejects_double_load` — no re-entrant Loading
- `state_machine_reset_noop_when_unloaded` — reset is no-op on Unloaded
- `state_machine_observes_transitions` — observer callbacks fire correctly
- `state_machine_observer_sees_failure` — observer captures Failed transition
- `state_machine_state_name` — stateName() returns correct strings
- `state_machine_multiple_observers` — multiple observers all fire

### 9.6 Latency SLO Benchmarks - Baselines (stub executor, debug build, 2 instances)

```
mean: ~247 us
p50:  ~223 us
p95:  ~357 us
p99:  ~409 us
Throughput: ~4280 req/s
```

### Validation

```bash
cmake --build --preset debug  &&  ctest --preset debug
# 100% tests passed, 0 tests failed
scripts/check-format.sh  # passes
```

## Evidence

| Date | Check | Result |
| --- | --- | --- |
| 2026-06-04 | Recorded checks above | Pass, as reported by the Step 9 snapshot |
