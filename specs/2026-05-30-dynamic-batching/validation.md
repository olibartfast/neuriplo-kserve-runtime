# Step 6: Dynamic Batching - validation

> Retrospective packet. Step 6 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 6
> snapshot (`plan/STEP6.md`, later `specs/history/steps/STEP6.md`) and the
> Step 6 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Validation was recorded at completion rather than written before the code;
the checks below are the ones the step snapshot reports as run.

## Checks

- [x] [V-1] -> [R-1]: the step's build, format and test commands (Recorded checks below) pass

## Traceability

| Requirement | Status | Evidence |
| --- | --- | --- |
| R-1 | Met | Recorded checks below |
| R-2 | Partial | Recorded checks below |

## Deviations

Where the snapshot shows the step delivered less than the planned
exit criterion. These are recorded from the snapshot text, not re-tested.

- [R-2] Partial: batch metrics were added as scheduler hooks; `/metrics` export landed
  in Step 7.

## Recorded checks

### Validation

```bash
cmake --preset debug
cmake --build --preset debug
scripts/check-format.sh
ctest --preset debug
cmake --preset tsan
cmake --build --preset tsan
ctest --preset tsan
```

Golden comparisons:

- `scheduler_batched_outputs_match_single_request_shape` verifies batched stub
  outputs match single-request shape and schema.
- Real backend smoke with batch size > 1 remains optional when the neuriplo
  adapter is available locally.

## Evidence

| Date | Check | Result |
| --- | --- | --- |
| 2026-05-30 | Recorded checks above | Pass, as reported by the Step 6 snapshot |
