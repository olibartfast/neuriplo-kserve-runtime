# k3d raw-Deployment serving

Manifests for the local k3d `neuriplo` cluster, used when the KServe CRDs are not
installed. With KServe present, prefer `deploy/kserve/`, where
`ClusterServingRuntime` carries the runtime name and `InferenceService` carries
the model name.

For the design behind these manifests -- the role each image plays, the contract
a prepare step has to honour, how to add a backend, and the failure modes -- see
[docs/init-container.md](../../docs/init-container.md). This page is the k3d
deployment; that one is the specification.

## Files

| File | Purpose |
|---|---|
| `runtime-trt.yaml` | TensorRT serving: prepare on the node, serve a model repository tree. Names no model. |
| `runtime-triton.yaml` | The same procedure with stock upstream Triton as the server. |
| `../prepare/prepare-repository.sh` | Builds the repository tree. Server-agnostic. |
| `../prepare/Dockerfile` | The preparer as a standalone init-container image. |
| `../compose/docker-compose.yml` | The same flow without Kubernetes. |

`runtime-trt.yaml` and `runtime-triton.yaml` differ in the server container and
one environment variable (`REPOSITORY_LAYOUT`). The artifact image, the
preparer, the volume layout, and the procedure are identical — which is the
point: the repository tree is the interface, not the runtime.

## Nothing here is model-specific, and nothing requires Kubernetes

Each step lives in the image that owns it, so the manifests only wire volumes:

| Step | Lives in | Not in |
|---|---|---|
| Staging `*.onnx` | the model image's `CMD` | an init-container `command:` |
| ONNX to engine | the serving image's `ENTRYPOINT` | a manifest, hook, or sidecar |
| Model discovery | the runtime's `--models` tree scan | a per-model flag |

Consequences worth relying on: adding or removing a model means rebuilding the
model image and changing no YAML, and the same two images run without a cluster.
`deploy/compose/docker-compose.yml` reproduces the identical flow -- compose's
`service_completed_successfully` is the init container -- and a single container
works too when the models are already on disk:

```bash
docker run --gpus all -p 8080:8080 \
  -v /path/to/onnx:/staging -v /path/to/repo:/models/repo \
  neuriplo-kserve-runtime:trt-gpu --host 0.0.0.0 --port 8080 --use-gpu true
```

No model name appears in any of the three.

## One pod, however many models

The node has a single GPU, so a second pod requesting `nvidia.com/gpu: 1` would
stay Pending indefinitely. Every model therefore shares one runtime process,
which is what repository mode is for: the tree can hold any number of models and
each is addressed by its own `/v2/models/<name>` route.

## TensorRT procedure

1. The init container stages the `*.onnx` files it carries into a shared staging
   volume, using its own default command.
2. The serving container runs `trtexec` on each staged `.onnx`.
3. Each engine is written to a repository tree:
   `/models/repo/<model-name>/<version>/model.plan`.
4. The runtime serves the tree: `neuriplo-kserve-runtime --models=/models/repo`.

Conversion runs in the serving container rather than at image build time because
a TensorRT engine is specific to the GPU, driver, and TensorRT version that built
it. An engine built on the host or in CI is not portable to the cluster node.

Every `*.onnx` in the staging directory is converted and the filename becomes the
model name, so the entrypoint never needs editing.

Engines live on a PVC, not an `emptyDir`. Measured on this node, a cold build
takes ~8.5 min for a 101 MB model and ~2 min for a 21 MB one -- roughly 10
minutes that would otherwise be paid on every pod replacement. Against a warm
claim the entrypoint skips conversion and the pod is ready in well under a
minute. `progressDeadlineSeconds` has to cover the cold case, or the rollout is
marked failed while conversion is still running.

### Static shapes

Do not set `TRT_SHAPES` for a model whose ONNX has fixed input dimensions.
`trtexec` rejects it with `Static model does not take explicit shapes` and the
conversion fails. Both YOLO26 depth exports are static `[1,3,768,768]`, so the
variable stays unset; it exists for models exported with dynamic axes.

### Image

Built by `docker/Dockerfile.tensorrt` as `neuriplo-kserve-runtime:trt-gpu`, on
`nvcr.io/nvidia/tensorrt:25.12-py3` (TensorRT 10.14.1, CUDA 13.1). The runtime is
compiled with `-DNEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO=ON
-DDEFAULT_BACKEND=TENSORRT`, so the binary carries both the `tensorrt` backend
and repository support. The same NGC base is used for the build and runtime
stages because conversion happens at serve time, so `trtexec` has to be in the
serving image.

```bash
docker build -f runtime/docker/Dockerfile.tensorrt \
  -t neuriplo-kserve-runtime:trt-gpu <context>
```

The context needs two sibling directories, `runtime/` (this repo) and
`neuriplo/`.

`neuriplo-kserve-runtime:onnx-gpu` cannot be used for this path. It links only
`libonnxruntime.so.1`, and while it ships
`libonnxruntime_providers_tensorrt.so`, that library cannot load because
`libnvinfer.so.10` is absent. The entrypoint fails loudly rather than silently
serving ONNX; set `TRT_FALLBACK_ONNX=true` to opt into the fallback explicitly.

## Model repository layout

`--models` (alias `--model-repository`, env `MODEL_REPOSITORY`) points at a
Triton-style tree:

```
<root>/<model-name>/<version>/<model file>
<root>/<model-name>/config.pbtxt        # optional I/O name overlay
```

Versions are numeric directories and the highest one is served. The backend is
inferred from the model file extension: `.plan`/`.engine` to `tensorrt`, `.onnx`
to `onnx_runtime`, `.json` to `ensemble`, and so on. When a version directory
holds both an engine and the ONNX it was built from, the engine wins. Every model
in the tree is served, and each appears in `POST /v2/repository/index`.

Single-model mode (`--model-path <file>`) is unchanged and remains the default.
