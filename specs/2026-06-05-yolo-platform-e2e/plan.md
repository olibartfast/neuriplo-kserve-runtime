# Step 12: Multi-Component YOLO Integration - plan

> Retrospective packet. Step 12 was delivered before this repository adopted
> spec-driven packets; this packet was ported on 2026-10-05 from the Step 12
> snapshot (`plan/STEP12.md`, later `specs/history/steps/STEP12.md`) and the
> Step 12 section of the original target design
> ([`2026-05-23-runtime-target-design`](../2026-05-23-runtime-target-design/plan.md)). The snapshot text is kept as
> written; file paths and line counts describe the tree at the time.

Tasks are the step's substeps. Each `T-` section is the implementation
record from the snapshot.

## Data flow

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

## Environment setup

```bash
# 1. Clone all components on their dev branches
git clone -b develop git@github.com:olibartfast/neuriplo-tasks.git
git clone -b develop git@github.com:olibartfast/neuriplo-infer.git
git clone git@github.com:olibartfast/neuriplo-kserve-runtime.git

# 2. Build runtime (neurip is auto-fetched via FetchContent; pin tag in versions.env)
cd neuriplo-kserve-runtime
cmake --preset real-onnx
cmake --build --preset real-onnx

# 3. Build neuriplo-tasks and neuriplo-infer (TBD — depends on their build system)
cd ../neuriplo-tasks && <build>
cd ../neuriplo-infer && <build>
```

## Runtime launch

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

## Follow-up that became 12.7

Create `scripts/e2e-yolo.sh` that:
1. Checks all repos are checked out at correct branches
2. Builds each component
3. Downloads a test model (or uses a fixture)
4. Starts the runtime
5. Calls health/metadata/infer via curl
6. Validates JSON response shapes
7. Reports pass/fail for each checkpoint

## Status: Complete

The platform-defined local serving chain was tested end-to-end:
`neuriplo-infer` -> KServe V2 HTTP → `neuriplo-kserve-runtime` → `neuriplo` → ONNX Runtime → YOLO.

## Runtime Configuration

```bash
./build/real-onnx/neuriplo-kserve-runtime \
    --model-name yolo \
    --model-path <path-to-yolo.onnx> \
    --backend onnx_runtime \
    --port 8080 \
    --instances 1
```

> This step was delivered as one unit; the sections above are its single
> task group.
