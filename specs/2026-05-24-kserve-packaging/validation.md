# Step 1: KServe Packaging - validation

> Retrospective packet. Step 1 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 1
> snapshot (`plan/STEP1.md`, later `specs/history/steps/STEP1.md`) and the
> Step 1 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Validation was recorded at completion rather than written before the code;
the checks below are the ones the step snapshot reports as run.

## Checks

- [x] [V-1] -> [R-2], [R-3]: the step's build, format and test commands (Recorded checks below) pass

## Traceability

| Requirement | Status | Evidence |
| --- | --- | --- |
| R-1 | Partial | Recorded checks below |
| R-2 | Met | Recorded checks below |
| R-3 | Met | Recorded checks below |

## Deviations

Where the snapshot shows the step delivered less than the planned
exit criterion. These are recorded from the snapshot text, not re-tested.

- [R-1] Partial: the manifests reference the image and the image was built and run with
  `docker`; no run in a KServe cluster is recorded.

## Recorded checks

### Tests

New focused coverage in `tests/RuntimeConfigTest.cpp` checks:

- Environment defaults are read.
- CLI flags override environment defaults.
- `/mnt/models` is selected when present and `MODEL_PATH` is unset.
- `MODEL_PATH` overrides the `/mnt/models` convention.
- Existing defaults, CLI parsing, and invalid port validation still work.

### Validation

Validation commands run for this step:

```bash
cmake --build --preset debug
ctest --preset debug
docker build -f docker/Dockerfile -t neuriplo-kserve-runtime:step1 .
docker run --rm neuriplo-kserve-runtime:step1 --version
docker run --rm neuriplo-kserve-runtime:step1 --help
```

## Evidence

| Date | Check | Result |
| --- | --- | --- |
| 2026-05-24 | Recorded checks above | Pass, as reported by the Step 1 snapshot |
