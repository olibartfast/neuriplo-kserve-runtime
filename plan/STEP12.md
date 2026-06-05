# Step 12: Multi-Component YOLO Integration

## Status: Verification Complete

The full neuriplo platform chain was smoke-tested end-to-end:
`curl/Python` → KServe V2 HTTP → `neuriplo-kserve-runtime` → `neuriplo` → ONNX Runtime → YOLO.

## Verified

| Check | Result |
|-------|--------|
| 12.2 | Build with `NEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO=ON` against sibling neuriplo checkout |
| 12.3 | Real YOLOv6s ONNX model loads via neuriplo, metadata extracted: input `images` [1,3,640,640] FP32, output `output0` [1,300,6] FP32 |
| 12.4 | Inference through `/v2/models/yolo/infer` reaches ONNX Runtime, returns correct output shape |
| 12.7 | `scripts/e2e-yolo.sh` automated smoke script (7 checks, all pass) |

## Runtime Configuration

```bash
./build/real-onnx/neuriplo-kserve-runtime \
    --model-name yolo \
    --model-path <path-to-yolo.onnx> \
    --backend onnx_runtime \
    --port 8080 \
    --instances 1
```

## Remaining (12.1, 12.5)

- **12.1**: Verify YOLO task contract in vision-core (in/out shapes, class count, NMS settings
  match the exported ONNX model)
- **12.5**: vision-inference KServe client — replace direct neuriplo linking with HTTP/gRPC
  calls to the runtime; requires changes in the vision-inference repository on the
  `neuriplo-kserve-runtime` branch

## E2E Smoke Script

Run with: `bash scripts/e2e-yolo.sh`
