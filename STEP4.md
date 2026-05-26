# Step 4: Atomic neuriplo Integration

This snapshot records the Step 4 implementation state. The runtime now has an
executor-facing tensor request path, a `neuriplo` adapter boundary, a
`NeuriploExecutor`, backend selection for known real backend ids, and optional
CMake wiring for a sibling `neuriplo` checkout.

## Implemented Scope

### Step 4.1: Preserve KServe Input Tensors

- `ExecutionRequest` now carries input tensor `name`, `datatype`, `shape`, and
  dense JSON numeric `data`.
- `KServeV2Codec` validates KServe V2 inputs against model metadata and
  preserves the parsed tensors.
- The codec rejects duplicate inputs and requests that omit required model
  inputs.
- `KServeRuntime` passes parsed inputs through to the executor.
- Stub executor behavior remains unchanged.

### Step 4.2: Define A neuriplo Adapter Boundary

- Added `NeuriploAdapter` as the internal runtime boundary.
- Route, HTTP, codec, and registry code continue to depend only on
  `Executor`.
- Unit tests can inject fake adapter behavior without linking `neuriplo`.

### Step 4.3: Add NeuriploExecutor Skeleton

- Added `NeuriploExecutor` implementing `Executor`.
- Model metadata is loaded from the adapter during construction.
- Adapter load failures are converted to registry failed-state behavior.

### Step 4.4: Wire Backend Selection

- `stub` still maps to `StubExecutor`.
- Known real backend ids map to `NeuriploExecutor`.
- Unknown backend ids fail predictably during startup load.
- When the default build lacks real `neuriplo` support, known real backend ids
  fail with a stable load error and readiness remains false.

### Step 4.5: Convert Request Inputs

- `NeuriploExecutor` accepts dense JSON `FP32` tensors.
- Backend-bound tensor validation rejects unsupported datatypes and data length
  mismatches before calling the adapter.
- Direct executor calls are validated against model metadata for unknown,
  duplicate, missing, wrong-datatype, and wrong-shape inputs.
- The real adapter converts `FP32` tensor values to byte buffers for
  `InferenceInterface::get_infer_results`.

### Step 4.6: Convert Backend Outputs

- Adapter outputs are converted into `ExecutionResponse` tensors.
- Backend inputs are passed to adapters in model metadata input order, not
  client JSON order.
- Requested-output filtering is applied in `NeuriploExecutor`.
- KServe response JSON shape stays compatible with existing tests.

### Step 4.7: Link Real neuriplo

- Added `NEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO`, default `OFF`.
- Added `NEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR` for a local sibling checkout.
- Default stub builds still work without `neuriplo`.
- Real-link configure and executable build were validated against
  `/home/oli/repos/neuriplo` with the default OpenCV DNN backend.

### Step 4.8: Smoke Model Validation

- Added `real_neuriplo_onnx_identity_golden_comparison`.
- The smoke test writes a tiny ONNX identity model at runtime, loads it through
  the real `neuriplo` ONNX Runtime backend, runs one inference request through
  `ModelRegistry` and `NeuriploExecutor`, and compares every output value
  against the input golden vector.
- Added a `real-onnx` CMake preset to make real-backend smoke validation
  repeatable.

## Validation

Validated commands:

```bash
cmake --preset debug
cmake --build --preset debug
scripts/check-format.sh
ctest --preset debug
cmake -S . -B build/real-neuriplo \
  -DNEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO=ON \
  -DNEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR=/home/oli/repos/neuriplo \
  -DBUILD_INFERENCE_ENGINE_TESTS=OFF
cmake --build build/real-neuriplo --target neuriplo-kserve-runtime
cmake --preset real-onnx
cmake --build --preset real-onnx
ctest --preset real-onnx
```

## Remaining Gaps

The real adapter currently assumes dense `FP32` inputs and dense tensor outputs
because that is the minimum safe conversion surface exposed by the current
sibling `neuriplo` API.
