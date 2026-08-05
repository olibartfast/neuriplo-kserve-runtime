# Pipeline (ensemble) models

A pipeline model runs an ordered graph of steps and serves it as one KServe V2
model. Clients send an encoded image instead of a dense float tensor, and can
get decoded results back instead of raw model outputs.

The wire surface is fixed by the platform ensemble contract, which is shared
with Triton ensembles so an application can point at either server unchanged.
ADR 0011 records why the runtime implements this natively.

## Building

Task-layer preprocess and postprocess steps are opt-in, because they are the one
place the serving runtime depends on the task layer:

```bash
cmake -S . -B build -DNEURIPLO_RUNTIME_ENABLE_TASKS=ON -DNEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO=ON
cmake --build build
```

Built without `NEURIPLO_RUNTIME_ENABLE_TASKS`, model-step chaining still works;
`preprocess` and `postprocess` steps refuse to load with an explicit message
rather than degrading at inference time.

## Loading

Referenced models must already be loaded, because composing the ensemble's
metadata means reading theirs:

```bash
curl -X POST localhost:8080/v2/admin/models/load \
  -d '{"model_name": "yolo", "backend": "onnx_runtime", "model_path": "/models/yolo26s.onnx"}'

curl -X POST localhost:8080/v2/admin/models/load \
  -d '{"model_name": "yolo_ensemble", "backend": "ensemble", "model_path": "deploy/ensemble/yolo-detection.json"}'
```

The graph can also be sent inline as `pipeline_graph` in the load body, instead
of by path.

## Graph format

```json
{
  "steps": [
    {"kind": "preprocess",  "name": "pre",    "task_type": "yolo26"},
    {"kind": "model",       "name": "detect", "model_name": "yolo"},
    {"kind": "postprocess", "name": "post",   "task_type": "yolo26",
     "envelope": "detection"}
  ]
}
```

Steps run in order. Each carries optional `input_map` and `output_map` objects
following Triton's ensemble convention: the key is the step's own tensor name,
the value is the graph tensor name.

```json
{"input_map": {"images": "preprocessed"}, "output_map": {"output0": "raw"}}
```

Omit them when names already line up, as they do above: the preprocess step
emits the model's declared input names directly.

Step fields:

| Field | Kinds | Meaning |
|---|---|---|
| `kind` | all | `model`, `preprocess`, or `postprocess` |
| `name` | all | unique within the graph |
| `model_name` | model | a model loaded in this runtime |
| `model_version` | model | optional; defaults to the model's default version |
| `task_type` | pre/post | neuriplo-tasks model-type string, e.g. `yolo26` |
| `envelope` | postprocess | `detection`, `mask`, or `polygon` |
| `confidence_threshold`, `nms_threshold`, `mask_threshold` | pre/post | task thresholds |

## What the composed model looks like

```text
platform:        ensemble
inputs:          the first step's inputs; IMAGE / UINT8 when it decodes an image
outputs:         the last step's outputs; the decoded envelope after a
                 postprocess step, otherwise the model's own outputs
max_batch_size:  1
```

A graph without a postprocess step is a passthrough ensemble: the server
preprocesses, the client postprocesses exactly as it does for a directly served
model. That is the mode where the client also needs the inner model's metadata,
since the ensemble's own metadata only describes an encoded image.

## Envelope cost

Measured on a 1280x720 frame with ~13 instances, YOLO26-seg on an RTX 3060
through the ONNX Runtime CUDA provider:

| Envelope | Server-side latency |
|---|---|
| `mask` | ~160 ms |
| `polygon` | ~5400 ms |

The gap is not inference, which is identical in both -- it is the polygon
conversion in the task layer, which runs connected components and a convex hull
per instance over a frame-sized mask. Prefer `mask` unless a consumer genuinely
needs vector geometry.

Note also that `polygon` output is convex-hull based, so a polygon is the convex
hull of its instance, not a tight contour around it.

## Rules worth knowing

- **The source image size travels as a tensor.** A preprocess step emits
  `FRAME_SIZE` (INT32, `[2]`, width then height) alongside the model input, and
  a postprocess step consumes it to map boxes back onto the original frame.
  Making it a graph edge is what lets graph validation catch a postprocess step
  wired without it.
- **Model steps resolve per request.** A referenced model can be reloaded
  underneath a loaded pipeline. One that is missing or not ready yields
  `MODEL_NOT_READY` rather than a dangling handle.
- **Pipelines never batch.** `max_batch_size` is 1 by contract; encoded images
  have no common shape. Loading a pipeline with dynamic batching configured
  fails outright.
- **Graphs are validated at load.** A step consuming a tensor no earlier step
  produces fails the load, not the first request.

## GPU preprocessing with DALI

The built-in `preprocess` step runs on the CPU. For GPU preprocessing, load a
serialized DALI pipeline as its own model (neuriplo's `DALI` backend) and chain
it as a `model` step -- see `yolo-seg-dali-tensorrt.json`:

```bash
curl -X POST localhost:8080/v2/admin/models/load -d '{
  "model_name": "yolo_pre", "backend": "dali", "use_gpu": true,
  "model_path": "/models/yolo_pre/1/pipeline.dali",
  "input_sizes": [[3, 640, 640]]}'
```

The DALI model emits the preprocessed tensor plus the source dimensions as
INT64 (height, width), which is both what the built-in `postprocess` step wants
for `FRAME_SIZE` and what the GPU postprocessing operators consume.

For an all-GPU ensemble, chain a second DALI model that postprocesses on the
GPU, so results never touch the host between steps:

```bash
curl -X POST localhost:8080/v2/admin/models/load -d '{
  "model_name": "yolo_post", "backend": "dali", "use_gpu": true,
  "model_path": "/models/yolo_post/1/pipeline.dali|plugin=/models/libyolo26_seg_dali.so|outnames=NUM_DETECTIONS,BOXES,SCORES,CLASSES,MASK_OFFSETS,MASK_DATA",
  "input_sizes": [[300, 38], [2], [32, 160, 160]]}'
```

`input_sizes` follows the pipeline's declared external-input order (query the
loaded model's metadata to see it). Measured on YOLO26m-seg with TensorRT FP16,
server-side: CPU pre+post 144.5 ms, GPU pre + CPU post 119.8 ms, GPU pre+post
69.9 ms. Pipelines are authored offline by
`export/dali/generate_yolo_pipeline.py` in the neuriplo repo -- inference-time
execution is pure C++ through the DALI C API.

Use `scripts/serve-dali-trt.sh` / `scripts/serve-onnx-gpu.sh` to launch with
the correct library paths; each script documents its own dependency story.

