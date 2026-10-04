# Repository Guidelines

## Project Structure & Module Organization

This repository is a C++17 KServe-compatible runtime. Runtime sources live in `src/`,
with `main.cpp` wiring configuration, model registry, KServe routing, and the HTTP
server. Unit tests live in `tests/` and are registered through CTest. Build presets are
defined in `CMakePresets.json`; CI is defined in `.github/workflows/ci.yml`. Local editor
debug tasks are under `.vscode/`.

Read `specs/roadmap.md` for current project status and the active work track. Steps 0–14
and the multi-backend track are complete; see the roadmap for the Step 15 raw output path
and later phases. Completed step snapshots live in `specs/history/steps/STEP0.md` through
`STEP14.md` (historical; do not add new STEP files, new work gets a dated packet under
`specs/`). `specs/history/steps/STEP0.md` remains useful context for the original scaffold
assumptions.

Treat `specs/history/target-design.md` as the original target design and step snapshots as
the implementation record. For architecture work, read `specs/architecture.md` for patterns
in use today and the "Architecture And Design Pattern Evolution" section in
`specs/history/target-design.md` for planned patterns. Prefer extending existing Strategy/factory/adapter boundaries over adding new
frameworks unless the roadmap calls for them.

## Specs And Planning Entry Point

`specs/` is the project constitution (`mission.md`, `tech-stack.md`, `roadmap.md`) and the
planning entry point; start there. The step snapshots and original target design are under `specs/history/`.
Active work that is multi-phase, changes public behavior or architecture, or has low
reversibility needs a dated `specs/YYYY-MM-DD-feature-name/` packet (requirements, plan,
validation) before implementation; cross-repo work uses a neuriplo-platform packet.

## MANDATORY: Agent Guide Maintenance

**Agents must keep this file current.** When your task changes any item below, update the
matching `AGENTS.md` section in the same PR/commit — do not wait for the user to ask.

Triggers:

- New dated `specs/` packets, or material edits to `specs/roadmap.md`
- Build, test, lint, or CI command/preset changes
- New `.cursor/rules/*.mdc` or other mandatory workflow rules
- Repo layout, module boundaries, or default runtime invocation changes
- New cross-cutting architectural patterns (also update `specs/architecture.md`)

Keep `AGENTS.md` as stable conventions and pointers. Do not duplicate full roadmap or step
snapshot content here.

## Build, Test, and Development Commands

Use CMake presets for repeatable local builds:

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

`debug` builds the runtime and unit test binary. `release` creates an optimized build.
`lint` enables clang-tidy. `asan`, `ubsan`, and `tsan` build with sanitizer
instrumentation. Release metadata lives in `VERSION` (read by CMake) and
`CHANGELOG.md` (Keep a Changelog format). Run `scripts/release-patch.sh` to
execute a GitFlow patch release (bumps `VERSION`, updates `CHANGELOG.md`,
merges to `master`, tags `vX.Y.Z`, and merges back to `develop`). Use
`--dry-run` to preview. Real-neuriplo presets
auto-clone neuriplo from GitHub (tag pinned in `versions.env`) into
`build/<preset>/_deps/neurip-src/`: `real-onnx` / `real-onnx-grpc` (single built-in
ONNX Runtime), `real-multi` (built-in OpenCV DNN + ONNX Runtime in one binary), and
`real-plugin` (OpenCV DNN built-in plus ONNX Runtime as a dlopen plugin; its ctest
preset sets `NEURIPLO_PLUGIN_DIR` to the build's `plugins/` directory). To iterate on
neurip itself, override with `-DNEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR=/path/to/neuriplo`.
`scripts/e2e-stub.sh` smoke-tests the stub HTTP surface;
`scripts/e2e-multi-backend.sh` exercises two backends (ONNX Runtime built-in +
TensorRT plugin) in one server on a local GPU machine — see its header for the
required build. Run the runtime locally with:

```bash
./build/debug/neuriplo-kserve-runtime --model-name demo --backend stub --port 8080
```

## MANDATORY: Pre-Push Lint And Format

**Agents must run formatting and lint checks before every `git push`.** Do not push
if either command fails.

```bash
scripts/check-format.sh
cmake --preset lint
cmake --build --preset lint --parallel
```

When C++ behavior changes, also run `ctest --preset debug` before pushing.

Quick format fix for tracked sources:

```bash
git ls-files 'src/*.cpp' 'src/*.hpp' 'tests/*.cpp' 'tests/*.hpp' \
  | xargs clang-format -i
scripts/check-format.sh
```

## Hyperlink verification

When editing documentation (`README.md`, `specs/**/*.md`) with hyperlinks:
- Verify all relative links resolve to existing files in the repo.
- Verify absolute GitHub URLs are reachable.
- Prefer absolute GitHub blob/tree URLs over fragile cross-repo relative paths.

## Coding Style & Naming Conventions

Use 4-space indentation and the repository `.clang-format` profile. Types and classes
use `PascalCase` (`ModelRegistry`, `HttpResponse`); functions and variables use
`camelCase` or lower snake case where already established. Keep headers in `src/*.hpp`
and implementation in matching `src/*.cpp` files. Hide concrete scheduler/backend
types behind interfaces and factory functions when touching scheduling or execution
code (`Scheduler`, `Executor`, `NeuriploAdapter`). Run formatting before submitting:

```bash
scripts/check-format.sh
```

## Testing Guidelines

Tests use the lightweight in-repo harness in `tests/Test.hpp` and run through CTest.
Name test files after the module under test, for example `RuntimeConfigTest.cpp`. Add
focused `TEST_CASE(...)` coverage for route handling, config parsing, and failure paths.
Sanitizer presets should pass for changes touching request handling, threading, parsing,
or ownership-sensitive code.

## MANDATORY: GitFlow Workflow

Follow the [Atlassian GitFlow workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).

Branch mapping for this repository:

- `master` — production (`main` in GitFlow)
- `develop` — integration branch for features
- `feature/*` or `feat/*` — branch from `develop`, merge to `develop` via PR
- `release/*` — branch from `develop`; merge to `master` (tagged) and `develop`;
  delete locally and on `origin`
- `hotfix/*` — branch from `master`; merge to `master` and `develop`; delete
  locally and on `origin`

Do not commit feature work directly to `master`. Use PRs for merges into `develop` and
`master`. If `develop` does not exist yet, create it from `master` before starting new
feature branches.

## Commit & Pull Request Guidelines

The current history uses short imperative commit messages, for example `Initial commit`.
Keep future messages concise and action-oriented, such as `Add runtime config tests`.
Pull requests should summarize the change, list validation commands run, and link related
issues when available. Include API examples or endpoint output when behavior changes.
Feature PRs target `develop`; release and hotfix PRs target `master` (and back-merge to
`develop` when applicable).

## Security & Configuration Tips

Do not commit model files, secrets, tokens, or generated `build*/` directories. Keep
runtime defaults safe for local development, and document any new network-facing options
in `README.md` and tests. Update `specs/architecture.md` when introducing a new
cross-cutting architectural pattern, update `specs/roadmap.md` when changing planned
architecture direction, and sync this file per "Agent Guide Maintenance" above.
