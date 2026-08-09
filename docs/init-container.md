# Init container specification

How a model repository is assembled before the runtime serves it: the roles, the
contract each role has to honour, how to add a backend, and how to run the whole
thing with and without Kubernetes.

The runtime itself has no opinion about any of this. It is pointed at a
directory and serves what it finds (`src/ModelRepository.cpp`). Everything below
describes how that directory comes to exist.

- [Why there is a preparation step at all](#why-there-is-a-preparation-step-at-all)
- [Roles](#roles)
- [The contract](#the-contract)
- [Writing a prepare step](#writing-a-prepare-step)
- [Managing models](#managing-models)
- [Backends](#backends)
- [Running it](#running-it)
- [Failure modes](#failure-modes)

## Why there is a preparation step at all

Some model formats are portable and can be committed, published, or baked into
an image. Others are compiled artifacts that are only valid on the machine that
built them:

| Format | Portable? | Consequence |
|---|---|---|
| `.onnx`, `.pte`, `.tflite`, `.xml`+`.bin`, `.torchscript` | yes | ship it directly; no prepare step |
| `.plan` / `.engine` (TensorRT) | **no** — tied to GPU model, driver, and TensorRT version | must be built on the serving node, at serve time |
| `.dali` | yes (serialized pipeline) | ship it, but it is produced by a Python export step |

The TensorRT case is what forces the design. An engine built in CI, on a laptop,
or in a `docker build` layer will not load on the cluster node. So the artifact
that gets served cannot always be the artifact that gets shipped, and something
has to sit between them. That something is the preparation step.

Once that step exists, the cheapest thing to do with portable formats is to run
them through the same path as a copy, so there is one procedure rather than two.

## Roles

Three roles, each owned by the image that performs it. This is the load-bearing
part of the design: **no orchestrator describes a step**, it only wires volumes.

```
┌──────────────────┐   /staging   ┌─────────────────────────────┐
│ artifact image   │─────────────>│ serving image               │
│                  │              │                             │
│ CMD: stage its   │              │ ENTRYPOINT: prepare, then   │
│ own files        │              │             exec the server │
└──────────────────┘              └──────────────┬──────────────┘
   deploy/models/Dockerfile                      │ /models/repo
                                                 v
                                    ┌─────────────────────────┐
                                    │ model repository tree   │
                                    │ <name>/<version>/<file> │
                                    └─────────────────────────┘
```

| Role | Owned by | Implemented in |
|---|---|---|
| **Stage** — put raw artifacts where the preparer can see them | the artifact image's `CMD` | `deploy/models/Dockerfile` |
| **Prepare** — compile/copy each artifact into the tree | the serving image's `ENTRYPOINT` | `deploy/k3d/trt-convert-entrypoint.sh` |
| **Serve** — discover and serve the tree | the runtime binary | `src/ModelRepository.cpp` |

### Why staging is not an init-container `command:`

Because then it would only exist in Kubernetes. Putting the copy in the artifact
image's own `CMD` means the identical image is an init container under
Kubernetes, a `service_completed_successfully` dependency under compose, and a
`docker run` on a workstation — with no step description duplicated in three
places and able to drift.

The same argument applies to conversion. It is the serving image's `ENTRYPOINT`,
not a `postStart` hook or a sidecar, so `docker run <serving image>` performs the
whole procedure by itself.

The practical test: **adding or removing a model must not change any YAML.** It
does not. Rebuild the artifact image, restart the pod.

## The contract

### Filesystem

| Path | Written by | Read by | Lifetime |
|---|---|---|---|
| `/artifacts` | artifact image build | artifact image `CMD` | baked into image |
| `$STAGE_DIR` (`/staging`) | artifact image `CMD` | prepare step | may be `emptyDir`; discardable |
| `$MODEL_REPOSITORY` (`/models/repo`) | prepare step | runtime | **must be persistent** — see below |

`$MODEL_REPOSITORY` is a PVC and not an `emptyDir` because engine builds are
expensive. Measured on the k3d node (RTX 3060): **~8.5 min for a 101 MB model,
~2 min for a 21 MB one.** On an `emptyDir` that ~10 minutes is paid on every pod
replacement. Against a warm claim the prepare step short-circuits and the pod is
ready in well under a minute.

`$STAGE_DIR` can be an `emptyDir` because it is re-populated by the init
container on every start.

### Environment

Consumed by the reference prepare step (`deploy/k3d/trt-convert-entrypoint.sh`):

| Variable | Default | Meaning |
|---|---|---|
| `STAGE_DIR` | `/staging` | where staged artifacts are read from |
| `MODEL_REPOSITORY` | `/models/repo` | repository root to build and then serve |
| `MODEL_VERSION` | `1` | version directory to write into |
| `TRT_PRECISION` | `fp16` | `fp16`, `fp32`, or `best` |
| `TRT_SHAPES` | unset | `trtexec` shape spec — **dynamic-axis models only** |
| `TRT_EXTRA_ARGS` | unset | extra `trtexec` arguments, word-split deliberately |
| `TRT_FALLBACK_ONNX` | `false` | serve the ONNX if `trtexec` is missing |

`MODEL_REPOSITORY` is also read directly by the runtime as an alias for
`--models` (`src/RuntimeConfig.cpp`), so the prepare step and the server agree on
the root without the manifest stating it twice.

### Invariants a prepare step must honour

These are what make the step safe to re-run, which it will be on every restart:

1. **Idempotent.** If the output artifact already exists, skip and return
   success. A restart against a warm volume must not rebuild.
2. **Atomic publish.** Write to `<target>.tmp` and `mv` into place. A conversion
   killed by an OOM kill or a failed startup probe must not leave a truncated
   file that the next start mistakes for a finished one.
3. **Fail loudly, never substitute silently.** If the intended backend cannot be
   produced, exit non-zero. Serving a different backend than asked for is a
   different latency and accuracy profile; it is only acceptable behind an
   explicit opt-in (`TRT_FALLBACK_ONNX=true`).
4. **`exec` the server last.** The server must be PID 1's successor so signals
   and exit codes propagate — `exec neuriplo-kserve-runtime "$@"`, and pass
   `"$@"` through so the manifest's `args:` still reach the binary.
5. **Derive the model name from the filename.** Never from a variable the
   manifest sets. This is what keeps the deployment model-agnostic.

## Writing a prepare step

`deploy/k3d/trt-convert-entrypoint.sh` is the reference implementation. Its
shape generalizes:

```sh
set -eu

# 1. Refuse to start on an empty staging dir rather than serving nothing.
#    An empty repository is indistinguishable from a broken volume mount.

for source in "$STAGE_DIR"/*.<ext>; do
    [ -f "$source" ] || continue          # 2. no-match glob is a literal string

    name=$(basename "$source" .<ext>)     # 3. filename becomes the model name
    target_dir="$MODEL_REPOSITORY/$name/$MODEL_VERSION"
    mkdir -p "$target_dir"

    [ -f "$target_dir/model.<out>" ] && continue    # 4. idempotent

    <compile> "$source" -o "$target_dir/model.<out>.tmp"
    mv "$target_dir/model.<out>.tmp" "$target_dir/model.<out>"   # 5. atomic
done

exec neuriplo-kserve-runtime --models="$MODEL_REPOSITORY" "$@"    # 6.
```

The output extension is the only thing that selects the backend. Write
`model.plan` and the model is served by TensorRT; write `model.onnx` and it is
served by ONNX Runtime. No flag, no manifest change, no registry entry — the
extension **is** the declaration (`backendForModelFile`, `src/ModelRepository.cpp`).

### Heterogeneous repositories

A tree may hold models of different backends and different tasks. Nothing in the
design requires them to be uniform: the scanner resolves each model directory
independently.

The reference entrypoint handles exactly one input format (`*.onnx` → `.plan`).
To prepare a mixed staging directory, dispatch on the input extension and let
portable formats fall through as copies:

```sh
case "$source" in
    *.onnx)   trtexec --onnx="$source" --saveEngine="$target/model.plan.tmp" ;;
    *.pte|*.tflite|*.xml|*.bin|*.dali)
              cp "$source" "$target/" ;;   # portable; nothing to compile
    *)        echo "unhandled artifact: $source" >&2; exit 1 ;;
esac
```

Note the last arm. Silently ignoring an unrecognized artifact produces a
repository that is missing a model with no error anywhere — the failure surfaces
much later as a 404 from a client.

### The build must contain the backends the tree needs

A prepare step that writes a `.plan`, a `.pte`, and an `.xml` produces a
repository that a TensorRT-only build cannot serve. `NEURIPLO_BACKENDS` selects
which backends are compiled in (`DEFAULT_BACKEND` sets the default among them);
models whose backend is absent fail to load. This is currently the sharpest
limit on heterogeneous serving — the `trt-gpu` image is a single-backend build.

## Managing models

### Layout

```
<root>/<model-name>/<version>/<model file>
<root>/<model-name>/config.pbtxt        # optional, I/O names only
```

- Version directories are **numeric**; the highest is served. Comparison is
  numeric, not lexical — `10` beats `9`, and leading zeros do not change rank.
- A non-numeric version directory is skipped with a warning, not an error.
- A version directory with no recognized model file is skipped with a warning.
- One malformed model never stops the rest of the repository from serving.

### `config.pbtxt`

Optional, and narrower than Triton's. It overlays **input/output tensor names
and datatypes by index** onto whatever the backend reports
(`applyConfigPbtxt`, `src/RealNeuriploAdapter.cpp`). It exists because some
formats carry no tensor names at all — ExecuTorch `.pte` advertises positional
`input_0`/`output_0` — and hardcoding a model's tensor names into a backend
would make that backend model-specific.

It does **not** select a backend, set instance counts, or configure batching.
If the overlay's tensor count disagrees with the backend's, it is ignored with a
warning rather than applied partially.

### Adding and removing models

```bash
deploy/models/build-model-image.sh --tag neuriplo-models:depth-v2 \
    --k3d-import neuriplo \
    yolo26n-depth yolo26l-depth /path/to/detector.onnx
kubectl -n neuriplo rollout restart deploy/neuriplo-kserve-runtime
```

Sources are either a path to an existing export or an Ultralytics model name,
which is exported on demand. Model files are deliberately not committed, so the
*recipe* is committed instead.

Removing a model from the image does not remove it from a warm PVC — the prepare
step only adds. Delete the model directory from the volume, or recreate the
claim.

### Controlling what is loaded

| Mode | Flag | Behaviour |
|---|---|---|
| `none` (default) | `--model-control-mode none` | every discovered model is loaded at startup |
| `explicit` | `--model-control-mode explicit` | nothing is loaded; the client loads on demand |

Explicit mode requires `--models`. It exists for two reasons: a client that only
needs one model does not pay for loading ten, and — more importantly — **a model
whose backend crashes on load cannot take the server down before anything has
asked for it.** That is not hypothetical; a dynamic-shape TensorRT engine in the
local test repository segfaults the process on load.

Clients drive it through the KServe model-repository extension:

```bash
curl -X POST localhost:8080/v2/repository/index
curl -X POST localhost:8080/v2/repository/models/<name>/load
curl -X POST localhost:8080/v2/repository/models/<name>/unload
```

`index` lists catalogued-but-unloaded models as `UNAVAILABLE`, so a client can
discover a model before loading it. Unloading returns a model to the catalog
rather than erasing it, so it can be loaded again.

## Backends

### What the scanner recognizes

| Extension | Backend | Prepare step |
|---|---|---|
| `.plan`, `.engine` | `tensorrt` | **compile on the serving node** |
| `.onnx` | `onnx_runtime` | copy |
| `.xml` (+ `.bin` beside it) | `openvino` | copy |
| `.pte` | `executorch` | copy |
| `.tflite` | `litert` | copy |
| `.torchscript`, `.pt` | `libtorch` | copy |
| `.pb` | `libtensorflow` | copy |
| `.dali` | `dali` | copy (produced by a Python export) |
| `.json` | `ensemble` | copy |

A `.bin` on its own is not a model — it is an OpenVINO weight blob, and a
version directory holding only that is skipped with a warning.

### When one directory holds several formats

A repository that keeps every export of a model side by side is normal. The
scanner picks by an explicit priority list, so the choice is identical on every
host:

```
tensorrt, onnx_runtime, openvino, executorch, litert,
libtorch, libtensorflow, dali, ensemble
```

Override with `NEURIPLO_REPOSITORY_BACKEND_PRIORITY`. No fixed order can be
right everywhere — an OpenVINO host and a CUDA host want opposite answers from
the same directory, and neither should have to rename files to get it.

The default order is why the TensorRT flow works without cleanup: the `.onnx`
can stay in the version directory next to the `.plan` built from it, and the
engine wins.

### Backends the runtime supports but the scanner cannot discover

`BackendRegistry.cpp` registers `opencv_dnn`, `migraphx`, and the LLM backends
`ggml`, `llamacpp`, and `cactus`. None has an extension mapping, so a model for
one of them cannot be placed in a repository tree — it must be served in
single-model mode (`--model-path`) or loaded through
`/v2/repository/models/<name>/load` with an explicit backend in the request body.

`.pb` maps to `libtensorflow`, but a TensorFlow SavedModel is a *directory*
(`model.savedmodel/`), and the scanner only inspects files inside a version
directory. A frozen-graph `.pb` works; a SavedModel does not.

## Running it

All three run the same two images with the same entrypoints. Only the volume
wiring differs.

### Kubernetes

```bash
kubectl apply -f deploy/k3d/runtime-trt.yaml
```

Two settings are not optional on a cold start:

- `startupProbe` with `failureThreshold: 120`, `periodSeconds: 10` — 20 minutes
  of budget, because the server does not bind until conversion finishes, and the
  liveness probe must not restart the pod mid-build.
- `progressDeadlineSeconds: 1800` — the 600s default marks the rollout failed
  while conversion is still legitimately running.

### Compose

```bash
docker compose -f deploy/compose/docker-compose.yml up
```

`service_completed_successfully` is compose's init container. The healthcheck
carries the same 120-retry budget as the Kubernetes startup probe.

### Plain Docker

With models already on disk, no init container is involved at all:

```bash
docker run --gpus all -p 8080:8080 \
  -v /path/to/onnx:/staging \
  -v /path/to/repo:/models/repo \
  neuriplo-kserve-runtime:trt-gpu --host 0.0.0.0 --port 8080 --use-gpu true
```

### Locally, no container

The runtime takes a repository root directly:

```bash
./build/debug/neuriplo-kserve-runtime --models /path/to/model_repository
```

Single-model mode is unchanged and remains the default:

```bash
./build/debug/neuriplo-kserve-runtime --model-name demo --backend stub --port 8080
```

### Building the serving image

```bash
docker build -f runtime/docker/Dockerfile.tensorrt -t neuriplo-kserve-runtime:trt-gpu <context>
```

The context needs two sibling directories, `runtime/` (this repo) and
`neuriplo/`. Both stages use the same `nvcr.io/nvidia/tensorrt` base, because
conversion happens at serve time and therefore `trtexec` has to be present in
the *runtime* stage, not just the build stage. `libneuriplo.so` must be copied
into the runtime stage and registered with `ldconfig`; without it the binary
dies at startup on a missing shared object.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `Static model does not take explicit shapes` | `TRT_SHAPES` set for an ONNX with fixed dims | unset it; it is for dynamic-axis exports only |
| `error: no .onnx files staged in /staging` | artifact image staged nothing, or the volume is not shared | check both containers mount the same staging volume |
| `trtexec not found in PATH` | serving image has no TensorRT | use `:trt-gpu`; `:onnx-gpu` cannot build engines |
| Rollout reported failed while logs show conversion running | `progressDeadlineSeconds` shorter than the build | raise it above the startup-probe budget |
| Pod restarts every few minutes during first start | liveness probe firing before the server binds | that window belongs to `startupProbe` |
| `cannot open shared object file: libneuriplo.so` | build stage copied the binary but not the library | `COPY --from=build` the `.so` and run `ldconfig` |
| Model missing from `/v2/repository/index` | its backend is not compiled into the image, or its extension is unmapped | check `NEURIPLO_BACKENDS`; see the unmapped-backend list above |
| Server exits on startup, no model loads | a backend segfaulted during load | `--model-control-mode explicit` contains the blast radius |

Discovery decisions are logged at startup — one `discovered model <name> version
<v> backend <b> at <path>` line per model, and a warning per skipped directory.
That log is the first place to look when a tree does not serve what was
expected.

## See also

- [deploy/k3d/README.md](../deploy/k3d/README.md) — the k3d deployment itself
- [deploy/ensemble/README.md](../deploy/ensemble/README.md) — ensemble graphs in a repository
- [deploy/kserve/README.md](../deploy/kserve/README.md) — `ClusterServingRuntime` / `InferenceService`
- [docs/errors.md](errors.md) — error taxonomy and HTTP status mapping
