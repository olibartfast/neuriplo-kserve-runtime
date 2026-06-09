# Repository Guidelines

## Project Structure & Module Organization

This repository is a C++17 KServe-compatible runtime. Runtime sources live in `src/`,
with `main.cpp` wiring configuration, model registry, KServe routing, and the HTTP
server. Unit tests live in `tests/` and are registered through CTest. Build presets are
defined in `CMakePresets.json`; CI is defined in `.github/workflows/ci.yml`. Local editor
debug tasks are under `.vscode/`.

Read `plan/STEP0.md` before assuming integration readiness. It documents the current
implemented scaffold and the missing pieces that still block real `neuriplo`,
`vision-core`, and `vision-inference` integration. Completed step snapshots live in
`plan/STEP1.md` through `plan/STEP6.md`; use `plan/STEP<N>_WIP.md` only for in-progress step work.

Treat `plan/ROADMAP.md` as the target roadmap and step snapshots as the current-state
record. For architecture work, read `plan/DESIGN_PATTERNS.md` for patterns in use today and
the "Architecture And Design Pattern Evolution" section in `plan/ROADMAP.md` for planned
patterns. Prefer extending existing Strategy/factory/adapter boundaries over adding new
frameworks unless the roadmap calls for them.

## Build, Test, and Development Commands

Use CMake presets for repeatable local builds:

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

`debug` builds the runtime and unit test binary. `release` creates an optimized build.
`lint` enables clang-tidy. `asan`, `ubsan`, and `tsan` build with sanitizer
instrumentation. Run the runtime locally with:

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
- `release/*` — branch from `develop`; merge to `master` (tagged) and `develop`
- `hotfix/*` — branch from `master`; merge to `master` and `develop`

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
in `README.md` and tests. Update `plan/DESIGN_PATTERNS.md` when introducing a new
cross-cutting architectural pattern, and update `plan/ROADMAP.md` when changing planned
architecture direction.
