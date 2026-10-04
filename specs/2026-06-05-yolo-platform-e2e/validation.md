# Step 12: Multi-Component YOLO Integration - validation

> Retrospective packet. Step 12 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 12
> snapshot (`plan/STEP12.md`, later `specs/history/steps/STEP12.md`) and the
> Step 12 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Validation was recorded at completion rather than written before the code;
the checks below are the ones the step snapshot reports as run.

## Checks

- [x] [V-1] -> [R-1]: task contract compared with `yolo26s.onnx` metadata (Verified 12.1)
- [x] [V-2] -> [R-2]: `cmake --preset real-onnx` build (Verified 12.2)
- [x] [V-3] -> [R-3], [R-4]: `/v2/models/yolo` metadata and `/infer` output shape (Verified 12.3, 12.4)
- [x] [V-4] -> [R-6]: `grpc_real_neuriplo_yolo_infer` test, in-process and against a live runtime (gRPC Validation)
- [x] [V-5] -> [R-7]: `bash scripts/e2e-yolo.sh` passes all HTTP and gRPC checks (Verified 12.7)
- [x] [M-1] -> [R-5]: manual CLI run against a live runtime renders `data/output/processed.png` (Latest HTTP E2E Validation)

## Traceability

| Requirement | Status | Evidence |
| --- | --- | --- |
| R-1 | Met | substep 12.1 tests and exit criteria below |
| R-2 | Met | substep 12.2 tests and exit criteria below |
| R-3 | Met | substep 12.3 tests and exit criteria below |
| R-4 | Met | substep 12.4 tests and exit criteria below |
| R-5 | Met | substep 12.5 tests and exit criteria below |
| R-6 | Met | substep 12.6 tests and exit criteria below |
| R-7 | Met | substep 12.7 tests and exit criteria below |

## Recorded checks

### Verified

| Check | Result |
|-------|--------|
| 12.1 | YOLO task contract verified against neuriplo-tasks (`feature/neuriplo-kserve-runtime`): `yolo26` maps to `YOLO_NMS_FREE`, postprocessor consumes `tensors[0]` `[batch, detections, 6]` (x1,y1,x2,y2,score,class) with confidence filter and no NMS; `yolo26s.onnx` reports `images` [1,3,640,640] FP32 in, `output0` [1,300,6] FP32 out — exact match |
| 12.2 | Build with `NEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO=ON` against sibling neuriplo checkout |
| 12.3 | Real YOLOv6s ONNX model loads via neuriplo, metadata extracted: input `images` [1,3,640,640] FP32, output `output0` [1,300,6] FP32 |
| 12.4 | Inference through `/v2/models/yolo/infer` reaches ONNX Runtime, returns correct output shape |
| 12.5 | `neuriplo-infer` CLI calls KServe HTTP endpoint, runtime executes YOLO via neuriplo/ONNX Runtime, task postprocess/render writes `data/output/processed.png` |
| 12.7 | `scripts/e2e-yolo.sh` automated smoke script (HTTP + gRPC checks, all pass) |
| 12.6 | gRPC path parity: `grpc_real_neuriplo_yolo_infer` test + live-runtime gRPC infer returns `output0` [1,300,6] |

### gRPC Validation

Build preset: `real-onnx-grpc` (`NEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO=ON` +
`NEURIPLO_RUNTIME_ENABLE_GRPC=ON`).

```bash
cmake --preset real-onnx-grpc
cmake --build --preset real-onnx-grpc
NEURIPLO_TEST_FILTER=grpc_real_neuriplo_yolo_infer \
  NEURIPLO_E2E_YOLO_MODEL=/path/to/yolo26s.onnx \
  ./build/real-onnx-grpc/neuriplo-kserve-runtime-tests
```

Live-runtime check (started by `scripts/e2e-yolo.sh`):

```bash
NEURIPLO_TEST_FILTER=grpc_real_neuriplo_yolo_infer \
  NEURIPLO_GRPC_E2E_HOST=127.0.0.1 \
  NEURIPLO_GRPC_E2E_PORT=19091 \
  ./build/real-onnx-grpc/neuriplo-kserve-runtime-tests
```

### E2E Smoke Script

Run with: `bash scripts/e2e-yolo.sh`

### Latest HTTP E2E Validation

Runtime:

```bash
./build/real-onnx/neuriplo-kserve-runtime \
    --model-name yolo \
    --model-path /home/oli/repos/neuriplo-infer/yolo26s.onnx \
    --backend onnx_runtime \
    --port 19090 \
    --instances 1
```

CLI:

```bash
/home/oli/repos/neuriplo-infer/build-kserve-codex/app/neuriplo-infer \
    --type=yolo26 \
    --source=data/dog.jpg \
    --labels=labels/coco.names \
    --kserve_endpoint=http://127.0.0.1:19090 \
    --kserve_model_name=yolo \
    --kserve_transport=http \
    --min_confidence=0.25
```

Result:

- CLI completed successfully.
- Runtime handled `/v2/models/yolo/versions/1/infer` with status `200`.
- Runtime metrics reported one accepted/successful infer request for `model="yolo",version="1"`.
- Rendered output written to `/home/oli/repos/neuriplo-infer/data/output/processed.png`.

## Procedure: test calls

The manual procedure that drove this step.

## Procedure: validation checklist

| Step | Check | Expected |
|------|-------|----------|
| 1 | Runtime starts | `--version` prints version |
| 2 | Backend loads | `/v2/health/ready` returns 200 |
| 3 | Metadata from neuriplo | `/v2/models/yolo` shows correct I/O shapes |
| 4 | Single infer | Returns dense FP32 output tensor |
| 5 | neuriplo-infer e2e | CLI processes image, shows bounding boxes |
| 6 | gRPC parity | Same infer result via gRPC as HTTP |
| 7 | Metrics | `/metrics` shows `neuriplo_http_infer_requests_total{status="200"}` |

## Evidence

| Date | Check | Result |
| --- | --- | --- |
| 2026-06-09 | Recorded checks above | Pass, as reported by the Step 12 snapshot |
