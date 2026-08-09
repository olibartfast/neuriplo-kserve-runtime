# Init container specification

How a model repository is assembled before a server serves it: the roles, the
contract each role has to honour, how to add a backend, and how to run the whole
thing with and without Kubernetes.

**Nothing here is specific to this runtime.** The output is a versioned model
repository tree — `<model>/<version>/<file>` — which is the layout Triton,
OpenVINO Model Server, and `neuriplo-kserve-runtime` all consume. The preparer
builds that tree and exits; which server mounts the volume next is not its
business. `deploy/k3d/runtime-triton.yaml` is the same procedure with stock
upstream Triton as the server, and differs from the neuriplo deployment only in
the server container and one environment variable.

The runtime itself has no opinion about any of this either. It is pointed at a
directory and serves what it finds (`src/ModelRepository.cpp`). Everything below
describes how that directory comes to exist.

- [Why there is a preparation step at all](#why-there-is-a-preparation-step-at-all)
- [Roles](#roles)
- [Serving it with something else](#serving-it-with-something-else)
- [The contract](#the-contract)
- [Staging shapes](#staging-shapes)
- [The dispatch](#the-dispatch)
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
┌──────────────────┐  /staging  ┌──────────────────┐
│ artifact image   │───────────>│ preparer image   │
│                  │            │                  │
│ CMD: stage its   │            │ compile / copy / │
│ own files        │            │ rename, then exit│
└──────────────────┘            └────────┬─────────┘
 deploy/models/Dockerfile                │ /models/repo
                                         v
                          ┌─────────────────────────┐
                          │ model repository tree   │
                          │ <name>/<version>/<file> │
                          └────────────┬────────────┘
                                       │  mounted by
                    ┌──────────────────┼──────────────────┐
                    v                  v                  v
            ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
            │ neuriplo      │  │ tritonserver  │  │ ovms          │
            └───────────────┘  └───────────────┘  └───────────────┘
```

| Role | Owned by | Implemented in |
|---|---|---|
| **Stage** — put raw artifacts where the preparer can see them | the artifact image's `CMD` | `deploy/models/Dockerfile` |
| **Prepare** — compile/copy/rename each artifact into the tree | the preparer image's `ENTRYPOINT` | `deploy/prepare/prepare-repository.sh`, `deploy/prepare/Dockerfile` |
| **Serve** — discover and serve the tree | whichever server mounts the volume | `src/ModelRepository.cpp` for this runtime |

The third column of the last row is the only one that changes between
deployments. Everything to the left of the tree is shared.

### Why staging is not an init-container `command:`

Because then it would only exist in Kubernetes. Putting the copy in the artifact
image's own `CMD` means the identical image is an init container under
Kubernetes, a `service_completed_successfully` dependency under compose, and a
`docker run` on a workstation — with no step description duplicated in three
places and able to drift.

The same argument applies to conversion. It is the preparer image's `ENTRYPOINT`,
not a `postStart` hook or a sidecar, so `docker run <preparer image>` performs
the whole procedure by itself — and, because it is a step rather than a wrapper,
it composes in front of a server nobody modified.

The practical test: **adding or removing a model must not change any YAML.** It
does not. Rebuild the artifact image, restart the pod.

## Serving it with something else

The preparer has two forms, and which one you want depends on whether the build
needs the serving container's own hardware.

| | `PREPARE_ONLY=true` | wrap (default) |
|---|---|---|
| What it does | builds the tree, exits 0 | builds the tree, then `exec`s `SERVER_EXEC` |
| Runs as | a real init container | the serving image's `ENTRYPOINT` |
| Server image | untouched, stock upstream | must contain the preparer |
| Use when | anything — this is the general form | the build must happen inside the serving container |

`PREPARE_ONLY` is the reusable form and the one to reach for. `deploy/prepare/Dockerfile`
builds it as a standalone image that starts no server at all:

```bash
docker build -f deploy/prepare/Dockerfile -t neuriplo-prepare:trt .
```

Wrap form exists because the serving image already had TensorRT in it, which
made it convenient — not because it is better. It is the narrower option.

### Layouts

Servers agree on `<model>/<version>/` and disagree on what the file inside is
called. `REPOSITORY_LAYOUT` selects the convention:

| Staged | `neuriplo` | `triton` | `ovms` |
|---|---|---|---|
| TensorRT engine | `model.plan` | `model.plan` | *refused* |
| ONNX | `model.onnx` | `model.onnx` | `model.onnx` |
| OpenVINO IR | `model.xml` + `model.bin` | `model.xml` + `model.bin` | `model.xml` + `model.bin` |
| TorchScript | `model.torchscript` | `model.pt` | *refused* |
| TensorFlow frozen graph | `model.pb` | `model.graphdef` | `model.pb` |
| TFLite | `model.tflite` | *refused* | *refused* |
| ExecuTorch | `model.pte` | *refused* | *refused* |
| DALI | `model.dali` | `model.dali` | *refused* |
| Ensemble graph | `model.json` | *refused* | *refused* |

"Refused" is a hard startup error naming the layout and the format, raised
*before* any conversion runs. The alternative — writing the file anyway — costs
minutes of engine build and produces a model the server never mentions, since a
filename it does not recognize is one it silently ignores.

Triton's ensembles are declared in `config.pbtxt` with `platform: "ensemble"`
rather than as a graph file, so a `.json` ensemble does not carry across. Stage
a Triton ensemble in tree form instead.

### Using it in front of Triton

`deploy/k3d/runtime-triton.yaml`, in full:

```yaml
initContainers:
  - name: stage-models          # unchanged artifact image
    image: neuriplo-models:depth-v1
  - name: prepare-repo
    image: neuriplo-prepare:trt
    env:
      - {name: PREPARE_ONLY, value: "true"}
      - {name: REPOSITORY_LAYOUT, value: triton}
    resources: {limits: {nvidia.com/gpu: 1}}
containers:
  - name: tritonserver          # stock upstream, nothing added
    image: nvcr.io/nvidia/tritonserver:25.12-py3
    args: [tritonserver, --model-repository=/models/repo]
```

Two constraints that are easy to miss:

- **The preparer needs the GPU.** A TensorRT engine is built against a real
  device. Init containers run to completion before the main container starts, so
  a single-GPU node is not contended — Kubernetes schedules on
  `max(init, main)`, not their sum.
- **TensorRT versions must match** between the preparer and the server. Triton
  25.12 and the preparer's NGC base both ship TensorRT 10.14. A mismatch is
  reported as a version error when the server loads the engine, well after the
  build succeeded.

Neither applies to a repository with no TensorRT in it, where preparation is
only copying and renaming and a `busybox` base is enough:

```bash
docker build -f deploy/prepare/Dockerfile --build-arg BASE_IMAGE=busybox:glibc .
```

### Wrapping a different server

If the build genuinely has to happen inside the serving container, `SERVER_EXEC`
takes any command; the container's own arguments are appended to it.

```yaml
env:
  - name: SERVER_EXEC
    value: "tritonserver --model-repository=/models/repo"
```

Default is `neuriplo-kserve-runtime --models=$MODEL_REPOSITORY`.

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

Consumed by `deploy/prepare/prepare-repository.sh`:

| Variable | Default | Meaning |
|---|---|---|
| `STAGE_DIR` | `/staging` | where staged artifacts are read from |
| `MODEL_REPOSITORY` | `/models/repo` | repository root to build |
| `MODEL_VERSION` | `1` | version directory for flat-form artifacts |
| `REPOSITORY_LAYOUT` | `neuriplo` | filename convention of the server that will read the tree: `neuriplo`, `triton`, `ovms` |
| `PREPARE_ONLY` | `false` | build and exit instead of starting a server |
| `SERVER_EXEC` | `neuriplo-kserve-runtime --models=$MODEL_REPOSITORY` | command to `exec` in wrap form |
| `ONNX_BACKEND` | `tensorrt` | what a staged `.onnx` becomes: `tensorrt` (compile) or `onnx_runtime` (copy) |
| `TRT_PRECISION` | `fp16` | `fp16`, `fp32`, or `best` |
| `TRT_SHAPES` | unset | `trtexec` shape spec — **dynamic-axis models only** |
| `TRT_EXTRA_ARGS` | unset | extra `trtexec` arguments, word-split deliberately |
| `TRT_FALLBACK_ONNX` | `false` | serve the ONNX if `trtexec` is missing |
| `PREPARE_IGNORE_UNKNOWN` | `false` | skip unrecognized staged files instead of failing |

`MODEL_REPOSITORY` is also read directly by the runtime as an alias for
`--models` (`src/RuntimeConfig.cpp`), so the prepare step and the server agree on
the root without the manifest stating it twice.

**Per-model overrides.** A heterogeneous repository can hold one static and one
dynamic model, which a single global `TRT_SHAPES` cannot express — set it and the
static model's conversion fails outright. Append the model name uppercased, with
non-alphanumerics replaced by underscores:

```yaml
env:
  - name: TRT_SHAPES_RAFT_LARGE      # raft-large.onnx, dynamic axes
    value: "input:1x3x480x640"
  - name: TRT_PRECISION_YOLO26N_DEPTH
    value: fp32
```

`TRT_SHAPES`, `TRT_PRECISION`, and `TRT_EXTRA_ARGS` all take the suffix. A
per-model value wins over the global; the global applies to everything else.
Still no model name in any *manifest structure* — only in a variable whose value
is the operator's business.

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

## Staging shapes

`deploy/prepare/prepare-repository.sh` accepts two, and a staging directory may
mix them freely.

### Flat

One file per model. The filename is the model name; the extension selects what
happens to it.

```
/staging/detector.onnx        ->  detector/1/model.plan       (compiled)
/staging/ecdet.pte            ->  ecdet/1/model.pte           (copied)
/staging/segmenter.xml        ->  segmenter/1/model.xml       (copied)
/staging/segmenter.bin        ->  segmenter/1/model.bin       (companion)
/staging/ecdet.pbtxt          ->  ecdet/config.pbtxt          (overlay)
/staging/yolo_ensemble.json   ->  yolo_ensemble/1/model.json  (copied)
```

The version is `MODEL_VERSION` for everything.

### Tree

Already repository-shaped, and copied through without interpretation.

```
/staging/raft/3/model.plan    ->  raft/3/model.plan
/staging/raft/3/labels.txt    ->  raft/3/labels.txt
/staging/raft/config.pbtxt    ->  raft/config.pbtxt
```

This is the escape hatch. A model that needs a version other than
`MODEL_VERSION`, extra files beside the model, or several versions at once
expresses it directly, rather than this script growing a config language to
describe the same thing indirectly.

## The dispatch

| Staged | Becomes | Action |
|---|---|---|
| `.onnx` | `model.plan` | `trtexec` compile (or `model.onnx` when `ONNX_BACKEND=onnx_runtime`) |
| `.plan`, `.engine` | `model.plan` | copy — a prebuilt engine, valid only if built on this node |
| `.xml` | `model.xml` + `model.bin` | copy both; **the `.bin` is renamed to match**, because OpenVINO resolves weights by basename |
| `.pte`, `.tflite`, `.torchscript`, `.pt`, `.pb`, `.dali`, `.json` | `model.<ext>` | copy |
| `.bin` | — | consumed with its `.xml`; an orphan is an error |
| `.pbtxt` | `<model>/config.pbtxt` | copy beside the version directory |
| anything else | — | **error**, unless `PREPARE_IGNORE_UNKNOWN=true` |

The last row is deliberate. Silently ignoring an unrecognized artifact produces a
repository quietly missing a model, and the failure then surfaces as a 404 from a
client long after the cause is out of sight.

The output extension is the only thing that selects the backend. Write
`model.plan` and the model is served by TensorRT; write `model.onnx` and it is
served by ONNX Runtime. No flag, no manifest change, no registry entry — the
extension **is** the declaration (`backendForModelFile`, `src/ModelRepository.cpp`).

### Atomicity across multi-file models

Each model version is built in `<model>/.prepare-tmp.<version>.<pid>/` and
published by renaming that directory into place. A per-file `.tmp` rename would
be enough for a single-file model but not for OpenVINO's `.xml` + `.bin`, where a
crash between the two renames leaves a model that loads without weights. Stale
temp directories are removed before anything is scanned, so a run killed partway
through can never have its leftovers published by a later pass.

### Adding a format

Add an arm to the `case` in `prepare_flat`, and an arm to the recognized-extension
list in the main loop. If the format needs no compilation, that is one line each.
Then add a case to `scripts/test-prepare-repository.sh`.

### Testing it

```bash
scripts/test-prepare-repository.sh
```

No GPU, no TensorRT, no runtime binary: `trtexec` and `neuriplo-kserve-runtime`
are replaced by stubs on `PATH` that record how they were called. What is under
test is the dispatch and the tree it produces, so it runs on an ordinary CI
runner. It covers heterogeneous staging, idempotency across restarts, atomicity
under a failed conversion, stale temp directories, orphaned weights, unknown
artifacts, per-model overrides, and refusal to serve an empty repository.

### The build must contain the backends the tree needs

A prepare step that writes a `.plan`, a `.pte`, and an `.xml` produces a
repository that a TensorRT-only build cannot serve. `NEURIPLO_BACKENDS` selects
which backends are compiled in (`DEFAULT_BACKEND` sets the default among them).
This is currently the sharpest limit on heterogeneous serving — the `trt-gpu`
image is a single-backend build.

Discovery does **not** check whether a backend is compiled in; it only maps the
extension. So a model whose backend is missing is still discovered, logged, and
added to the catalog, and the failure arrives later:

```
POST /v2/repository/models/<name>/load
409  {"error":{"code":"UNAVAILABLE","message":"real neuriplo support is not enabled; ..."}}
```

Worth knowing because it means the index is a statement about the *repository*,
not about what this binary can actually run. A model listed `UNAVAILABLE` may be
perfectly well-formed and simply have no backend here.

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
| `error: no servable artifacts staged in /staging` | artifact image staged nothing, or the volume is not shared | check both containers mount the same staging volume |
| `error: unrecognized staged file` | an artifact with no dispatch arm | add an arm, or set `PREPARE_IGNORE_UNKNOWN=true` |
| `error: <f>.bin ... no matching .xml` | OpenVINO weights staged without the model | stage both, with the same basename |
| `error: <f>.pbtxt has no matching model` | overlay staged for a model that is not in the tree | check the basenames agree |
| `trtexec not found in PATH` | serving image has no TensorRT | use `:trt-gpu`, or set `ONNX_BACKEND=onnx_runtime` |
| Rollout reported failed while logs show conversion running | `progressDeadlineSeconds` shorter than the build | raise it above the startup-probe budget |
| Pod restarts every few minutes during first start | liveness probe firing before the server binds | that window belongs to `startupProbe` |
| `cannot open shared object file: libneuriplo.so` | build stage copied the binary but not the library | `COPY --from=build` the `.so` and run `ldconfig` |
| Model absent from `/v2/repository/index` entirely | its extension is unmapped, or the version directory holds nothing recognized — discovery skipped it | startup logs `no recognized model file under <dir>; skipping`; see the unmapped-backend list above |
| Model listed `UNAVAILABLE`, and `load` returns 409 | discovery maps by extension and never checks whether the backend is compiled in, so the model is catalogued and only fails when something asks for it | check `NEURIPLO_BACKENDS` / `DEFAULT_BACKEND`; the 409 body names the missing support |
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
