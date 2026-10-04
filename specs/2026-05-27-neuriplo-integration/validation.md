# Step 4: Atomic neuriplo Integration - validation

> Retrospective packet. Step 4 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 4
> snapshot (`plan/STEP4.md`, later `specs/history/steps/STEP4.md`) and the
> Step 4 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Validation was recorded at completion rather than written before the code;
the checks below are the ones the step snapshot reports as run.

## Checks

- [x] [V-1] -> [R-1], [R-2], [R-3], [R-4], [R-5], [R-6], [R-7], [R-8], [R-9], [R-10], [R-11], [R-12], [R-13], [R-14], [R-15], [R-16], [R-17]: the step's build, format and test commands (Recorded checks below) pass

## Traceability

| Requirement | Status | Evidence |
| --- | --- | --- |
| R-1 | Met | substep 4.1 tests and exit criteria below |
| R-2 | Met | substep 4.1 tests and exit criteria below |
| R-3 | Met | substep 4.2 tests and exit criteria below |
| R-4 | Met | substep 4.2 tests and exit criteria below |
| R-5 | Met | substep 4.3 tests and exit criteria below |
| R-6 | Met | substep 4.3 tests and exit criteria below |
| R-7 | Met | substep 4.4 tests and exit criteria below |
| R-8 | Met | substep 4.4 tests and exit criteria below |
| R-9 | Met | substep 4.5 tests and exit criteria below |
| R-10 | Met | substep 4.5 tests and exit criteria below |
| R-11 | Met | substep 4.6 tests and exit criteria below |
| R-12 | Met | substep 4.6 tests and exit criteria below |
| R-13 | Met | substep 4.7 tests and exit criteria below |
| R-14 | Met | substep 4.7 tests and exit criteria below |
| R-15 | Met | substep 4.8 tests and exit criteria below |
| R-16 | Met | substep 4.8 tests and exit criteria below |
| R-17 | Met | substep 4.8 tests and exit criteria below |

## Recorded checks

### Validation

Validated commands:

```bash
cmake --preset debug
cmake --build --preset debug
scripts/check-format.sh
ctest --preset debug
cmake -S . -B build/real-neuriplo \
  -DNEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO=ON \
  -DNEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR=/home/oli/repos/neuriplo \
  -DBUILD_INFERENCE_ENGINE_TESTS=OFF
cmake --build build/real-neuriplo --target neuriplo-kserve-runtime
cmake --preset real-onnx
cmake --build --preset real-onnx
ctest --preset real-onnx
```

## Evidence

| Date | Check | Result |
| --- | --- | --- |
| 2026-05-27 | Recorded checks above | Pass, as reported by the Step 4 snapshot |
