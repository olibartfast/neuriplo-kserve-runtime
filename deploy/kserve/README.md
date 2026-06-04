# KServe Deployment Guide

## ClusterServingRuntime

Deploy the runtime manifest first to register the `neuriplo-kserve-runtime` with KServe:

```bash
kubectl apply -f deploy/kserve/cluster-serving-runtime.yaml
```

## Startup & Readiness Probes

The runtime exposes a `/v2/health/live` endpoint (server liveness) and
`/v2/health/ready` (model readiness). Configure probes in the
`ClusterServingRuntime` spec or InferenceService pod template:

```yaml
spec:
  containers:
    - name: kserve-container
      args:
        - --model-name
        - "{{ .Name }}"
        - --port
        - "8080"
      ports:
        - containerPort: 8080
      startupProbe:
        httpGet:
          path: /v2/health/ready
          port: 8080
        initialDelaySeconds: 2
        periodSeconds: 3
        failureThreshold: 30
      livenessProbe:
        httpGet:
          path: /v2/health/live
          port: 8080
        periodSeconds: 10
      readinessProbe:
        httpGet:
          path: /v2/health/ready
          port: 8080
        periodSeconds: 5
```

### Probe Behavior by Model State

| State | `/v2/health/live` | `/v2/health/ready` |
|---|---|---|
| Unloaded | 503 | 503 |
| Loading | 503 | 503 |
| Ready | 200 | 200 |
| Failed | 503 | 503 |
| Unavailable | 503 | 503 |
| Unloading | 503 | 503 |

The readiness endpoint returns HTTP 200 only when the model is in Ready state
with a non-null scheduler. All other states return 503 with a JSON error body
indicating the current state.

### Startup Timing

Model load time depends on the backend and model size. Typical ranges:

| Backend | Small model (<100MB) | Large model (>1GB) |
|---|---|---|
| Stub (test) | <100ms | <100ms |
| ONNX Runtime | 1-5s | 10-60s |
| llama.cpp | 2-8s | 30-120s |

Set `startupProbe.failureThreshold` high enough to cover the expected load time
plus a 2x safety margin. For large models, `failureThreshold: 60` with
`periodSeconds: 3` gives a 3-minute window.

## InferenceService

Deploy a model-backed inference service:

```bash
kubectl apply -f deploy/kserve/inferenceservice.yaml
```

### Canary Rollout

```bash
kubectl apply -f deploy/kserve/canary-inferenceservice.yaml
```

## InferenceGraph

```bash
kubectl apply -f deploy/kserve/inferencegraph.yaml
```
