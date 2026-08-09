# Step 12: Multi-Component YOLO Integration

## Status: Complete

The platform-defined local serving chain was tested end-to-end:
`neuriplo-infer` -> KServe V2 HTTP → `neuriplo-kserve-runtime` → `neuriplo` → ONNX Runtime → YOLO.

## Verified

| Check | Result |
|-------|--------|
| 12.1 | YOLO task contract verified against neuriplo-tasks (`feature/neuriplo-kserve-runtime`): `yolo26` maps to `YOLO_NMS_FREE`, postprocessor consumes `tensors[0]` `[batch, detections, 6]` (x1,y1,x2,y2,score,class) with confidence filter and no NMS; `yolo26s.onnx` reports `images` [1,3,640,640] FP32 in, `output0` [1,300,6] FP32 out — exact match |
| 12.2 | Build with `NEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO=ON` against sibling neuriplo checkout |
| 12.3 | Real YOLOv6s ONNX model loads via neuriplo, metadata extracted: input `images` [1,3,640,640] FP32, output `output0` [1,300,6] FP32 |
| 12.4 | Inference through `/v2/models/yolo/infer` reaches ONNX Runtime, returns correct output shape |
| 12.5 | `neuriplo-infer` CLI calls KServe HTTP endpoint, runtime executes YOLO via neuriplo/ONNX Runtime, task postprocess/render writes `data/output/processed.png` |
| 12.7 | `scripts/e2e-yolo.sh` automated smoke script (HTTP + gRPC checks, all pass) |
| 12.6 | gRPC path parity: `grpc_real_neuriplo_yolo_infer` test + live-runtime gRPC infer returns `output0` [1,300,6] |

## Runtime Configuration

```bash
./build/real-onnx/neuriplo-kserve-runtime \
    --model-name yolo \
    --model-path <path-to-yolo.onnx> \
    --backend onnx_runtime \
    --port 8080 \
    --instances 1
```

## gRPC Validation

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

## E2E Smoke Script

Run with: `bash scripts/e2e-yolo.sh`

## Latest HTTP E2E Validation

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
