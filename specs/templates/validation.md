# Feature Validation — <feature name>

> Copy to `specs/YYYY-MM-DD-<feature-name>/validation.md`. Write this *before*
> implementation, so the criteria cannot be fitted to whatever gets built.
> Execute it from this file — do not summarize it from memory.

## Automated

- [ ] Debug preset configures, builds, and passes:
      `cmake --preset debug && cmake --build --preset debug && ctest --preset debug`
- [ ] Focused tests cover the success path *and* the failure path this feature adds.
- [ ] `scripts/check-format.sh` is clean; `cmake --preset lint` builds clean.
- [ ] Sanitizer presets pass if the change touches request handling, threading,
      parsing, or ownership-sensitive code.
- [ ] Real-neuriplo presets this feature could break still pass — list the ones
      that apply (`real-onnx`, `real-multi`, `real-plugin`, `real-onnx-grpc`).
- [ ] Relevant smoke scripts still pass (`scripts/e2e-stub.sh`,
      `scripts/e2e-yolo.sh`, `scripts/e2e-multi-backend.sh`).
- [ ] k3d cluster path still deploys if `deploy/` or the image changed.

## Manual

- [ ] The primary invocation behaves as `requirements.md` specifies — record the
      exact command and the observed output.
- [ ] Invalid, empty, and unsupported-combination inputs fail fast with a stable
      error from the documented taxonomy.
- [ ] `/metrics` output carries the model label where applicable.
- [ ] Every hyperlink added to docs resolves (`ls` for relative, `curl -sI` for absolute).

## Definition of Done

- [ ] Every requirement is implemented or explicitly deferred in `requirements.md`.
- [ ] Nothing in *Out of Scope* was implemented anyway.
- [ ] Deviations from this file are recorded here, honestly, with what was run instead.
- [ ] `CHANGELOG.md` describes the change; `../roadmap.md` phase status is updated.
- [ ] Spec, code, changelog, and roadmap tell the same story in one branch.
