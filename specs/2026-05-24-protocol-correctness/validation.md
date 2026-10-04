# Step 2: Protocol Correctness - validation

> Retrospective packet. Step 2 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 2
> snapshot (`plan/STEP2.md`, later `specs/history/steps/STEP2.md`) and the
> Step 2 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Validation was recorded at completion rather than written before the code;
the checks below are the ones the step snapshot reports as run.

## Checks

- [x] [V-1] -> [R-1], [R-2], [R-3], [R-4], [R-5]: the step's build, format and test commands (Recorded checks below) pass

## Traceability

| Requirement | Status | Evidence |
| --- | --- | --- |
| R-1 | Met | Recorded checks below |
| R-2 | Met | Recorded checks below |
| R-3 | Met | Recorded checks below |
| R-4 | Met | Recorded checks below |
| R-5 | Met | Recorded checks below |

## Recorded checks

### Tests

Updated and added tests:

```text
tests/KServeRuntimeTest.cpp
tests/KServeV2CodecTest.cpp
tests/RuntimeConfigTest.cpp
tests/HttpIntegrationTest.cpp
```

Coverage includes:

- Metadata `versions`.
- Versioned metadata, readiness, and inference routes.
- Inference request `id` echo.
- Malformed JSON rejection.
- Unsupported shape, datatype, input, and output validation.
- Wrong-method `405` behavior.
- `MAX_REQUEST_BYTES` and `--max-request-bytes` parsing.
- Real-socket HTTP integration checks for server metadata, model metadata,
  wrong methods, inference, malformed JSON, unknown model, and oversized
  request handling.

### Validation

Validation commands run for this step:

```bash
cmake --preset debug
scripts/check-format.sh
cmake --build --preset debug
ctest --preset debug
docker build -f docker/Dockerfile -t neuriplo-kserve-runtime:step2 .
```

## Evidence

| Date | Check | Result |
| --- | --- | --- |
| 2026-05-24 | Recorded checks above | Pass, as reported by the Step 2 snapshot |
