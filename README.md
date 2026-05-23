# neuriplo-kserve-runtime

KServe-compatible C++ inference runtime for [neuriplo] backends.

This repository is the serving data plane for [neuriplo]. It owns the KServe
V2 protocol surface, request admission, scheduling, batching, model lifecycle,
and operational endpoints. The actual backend execution remains owned by
[neuriplo].

## Initial Scope

- HTTP KServe V2 health and metadata endpoints.
- Single-model runtime process.
- Bounded request handling.
- Raw tensor inference API placeholder.
- CMake-based C++ build.

## Planned Flow

1. KServe / client
2. `neuriplo-kserve-runtime`
3. model registry
4. scheduler
5. [neuriplo] backend instance
6. KServe V2 response

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```


## Development

### Configure and build

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

### Unit tests

Unit tests are built as `neuriplo-kserve-runtime-tests` and registered with CTest.

```bash
cmake --build --preset debug
ctest --preset debug
```

### Lint and format

```bash
scripts/check-format.sh
cmake --preset lint
cmake --build --preset lint
```

### Debugger

VS Code launch configurations are provided for the runtime and unit tests. Build the
`debug` preset before launching, or let the configured pre-launch task run it.

### Sanitizers and memory checks

```bash
cmake --preset asan && cmake --build --preset asan && ctest --preset asan
cmake --preset ubsan && cmake --build --preset ubsan && ctest --preset ubsan
cmake --preset tsan && cmake --build --preset tsan && ctest --preset tsan
valgrind --leak-check=full --error-exitcode=1 ./build/debug/neuriplo-kserve-runtime-tests
```

## Run

```bash
./build/neuriplo-kserve-runtime \
  --model-name demo \
  --model-path /models/model.onnx \
  --backend onnx_runtime \
  --port 8080
```

## Endpoints

```text
GET  /v2
GET  /v2/health/live
GET  /v2/health/ready
GET  /v2/models/{model_name}
GET  /v2/models/{model_name}/ready
POST /v2/models/{model_name}/infer
```

[neuriplo]: https://github.com/olibartfast/neuriplo
