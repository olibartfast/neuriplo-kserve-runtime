# End-to-End YOLO Integration Plan

## Goal

Smoke-test the full neuriplo platform end-to-end: `neuriplo-infer` → KServe V2
(HTTP or gRPC) → `neuriplo-kserve-runtime` → `neuriplo` → ONNX Runtime backend →
YOLO model → response → `neuriplo-infer` displays result.

## Components And Branches

| Component | Repo | Dev Branch | Role |
|-----------|------|-----------|------|
| `neuriplo` | `github.com/olibartfast/neuriplo` | `neuriplo-kserve-runtime` | ONNX Runtime backend, model load/infer |
| `neuriplo-tasks` | `github.com/olibartfast/neuriplo-tasks` | `develop` | YOLO task contract, pre/post tensor shapes |
| `neuriplo-infer` | `github.com/olibartfast/neuriplo-infer` | `develop` | CLI, reads image, calls KServe endpoint |
| `neuriplo-kserve-runtime` | this repo | `master` | KServe V2 HTTP+gRPC server |

## Data Flow

```
neuriplo-infer (CLI)
  │
  ├─ Read dog.jpg
  ├─ Preprocess: neuriplo-tasks YOLO task → resize/normalize → FP32 [1,3,640,640]
  │
  ├─ POST /v2/models/yolo/infer (HTTP)  or  gRPC ModelInfer
  │     Inputs:  [{"name":"input","shape":[1,3,640,640],"datatype":"FP32","data":[...]}]
  │     Outputs: [{"name":"output","shape":[1,84,8400],"datatype":"FP32"}]
  │
  ▼
neuriplo-kserve-runtime
  │
  ├─ KServeRuntime / GrpcServer routes request
  ├─ ModelRegistry → BackendRegistry.createExecutorFor("onnx_runtime")
  ├─ Scheduler → batches/completes → ExecutionInstance
  │
  ▼
neuriplo (via NeuriploExecutor / RealNeuriploAdapter)
  │
  ├─ Load yolo.onnx from /mnt/models (or local path)
  ├─ Run ONNX Runtime inference
  ├─ Return dense FP32 output tensor
  │
  ▼
neuriplo-kserve-runtime
  │
  ├─ Scheduler splits batched response
  ├─ KServeV2Codec / GrpcV2Codec serializes JSON/protobuf
  │
  ▼
neuriplo-infer
  │
  ├─ Postprocess: neuriplo-tasks YOLO task → NMS, decode boxes
  ├─ Display image with bounding boxes
```

## Environment Setup

```bash
# 1. Clone all components on their dev branches
git clone -b neuriplo-kserve-runtime git@github.com:olibartfast/neuriplo.git
git clone -b develop git@github.com:olibartfast/neuriplo-tasks.git
git clone -b develop git@github.com:olibartfast/neuriplo-infer.git
git clone git@github.com:olibartfast/neuriplo-kserve-runtime.git

# 2. Build neuriplo first (shared lib)
cd neuriplo
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 3. Build runtime with real neuriplo
cd ../neuriplo-kserve-runtime
cmake --preset real-onnx -DNEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR=../neuriplo
cmake --build --preset real-onnx

# 4. Build neuriplo-tasks and neuriplo-infer (TBD — depends on their build system)
cd ../neuriplo-tasks && <build>
cd ../neuriplo-infer && <build>
```

## Runtime Launch

```bash
# Local mode (model on local filesystem)
./build/real-onnx/neuriplo-kserve-runtime \
    --model-name yolo \
    --model-path /models/yolo.onnx \
    --backend onnx_runtime \
    --port 8080 \
    --grpc-port 9000 \
    --instances 1

# KServe mode (model at /mnt/models from storage-initializer)
./build/real-onnx/neuriplo-kserve-runtime \
    --model-name yolo \
    --backend onnx_runtime \
    --port 8080
```

## Test Calls

### HTTP

```bash
# Health
curl http://127.0.0.1:8080/v2/health/live
curl http://127.0.0.1:8080/v2/health/ready

# Model metadata (should show YOLO input/output shapes from neuriplo)
curl http://127.0.0.1:8080/v2/models/yolo

# Inference
curl -X POST http://127.0.0.1:8080/v2/models/yolo/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "yolo-test-1",
    "inputs": [{
      "name": "input",
      "shape": [1, 3, 640, 640],
      "datatype": "FP32",
      "data": []  // neuriplo-infer populates real pixels
    }],
    "outputs": [{"name": "output"}]
  }'
```

### gRPC (when compiled with GRPC support)

```bash
# With grpcurl (install: go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest)
grpcurl -plaintext 127.0.0.1:9000 list
grpcurl -plaintext 127.0.0.1:9000 inference.GRPCInferenceService/ServerLive
grpcurl -plaintext 127.0.0.1:9000 inference.GRPCInferenceService/ModelMetadata \
  -d '{"name":"yolo"}'
```

### Full CLI (neuriplo-infer)

```bash
neuriplo-infer \
    --endpoint http://127.0.0.1:8080 \
    --model yolo \
    --type yolo \
    --source data/dog.jpg \
    --output result.jpg
```

## Validation Checklist

| Step | Check | Expected |
|------|-------|----------|
| 1 | Runtime starts | `--version` prints version |
| 2 | Backend loads | `/v2/health/ready` returns 200 |
| 3 | Metadata from neuriplo | `/v2/models/yolo` shows correct I/O shapes |
| 4 | Single infer | Returns dense FP32 output tensor |
| 5 | neuriplo-infer e2e | CLI processes image, shows bounding boxes |
| 6 | gRPC parity | Same infer result via gRPC as HTTP |
| 7 | Metrics | `/metrics` shows `neuriplo_http_infer_requests_total{status="200"}` |

## Known Gaps

1. **neuriplo-tasks YOLO task contract** — verify input shape (H×W), number of classes,
   output format (boxes + class scores) matches the ONNX model export.
2. **neuriplo-infer KServe client** — needs an HTTP/gRPC client abstraction that
   sends KServe V2 JSON or protobuf requests instead of linking neuriplo directly.
3. **Model artifact** — need a YOLOv8n or YOLOv5s exported to ONNX with the
   expected input/output names (`input` → `output`).
4. **Dynamic batching** — if enabled, batch dimension handling must match the
   ONNX model's NCHW layout.

## Next Step: Multi-Component Verification Script

Create `scripts/e2e-yolo.sh` that:
1. Checks all repos are checked out at correct branches
2. Builds each component
3. Downloads a test model (or uses a fixture)
4. Starts the runtime
5. Calls health/metadata/infer via curl
6. Validates JSON response shapes
7. Reports pass/fail for each checkpoint
